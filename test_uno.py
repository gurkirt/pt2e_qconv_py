import pdb
import torch
import torch.nn as nn
import torch.quantization
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer, get_default_x86_inductor_quantization_config


# Define a simple model with one Conv2D layer
class SingleConvModel(nn.Module):
    def __init__(self):
        super(SingleConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        return self.conv(x)



# Quantized 2D convolution with per-channel quantization and precomputed quantized bias
def quantized_conv2d(input_uint8, kernel, kernel_zero_points, kernel_scales, bias_q,
                           input_scale, input_zero_point, 
                           output_scale, output_zero_point, 
                           bias_quanitzed=False, relu=False):
    
    # Input and kernel sizes
    batch_size, in_channels, input_height, input_width = input_uint8.shape
    output_channels, in_channels, kernel_height, kernel_width = kernel.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    

    output = torch.zeros((batch_size, output_channels, output_height, output_width), dtype=torch.float32)

    # Perform the convolution operation
    for n in range(batch_size):  # Loop over batch size
        for oc in range(output_channels):  # Loop over output channels
            for h in range(output_height):
                for w in range(output_width):
                    # Initialize accumulator for the current output pixel
                    acc = 0
                    # Sum across input channels
                    for ic in range(in_channels):
                        # Perform convolution for this patch
                        for kh in range(kernel_height):
                            for kw in range(kernel_width):

                                input_val = input_uint8[n, ic, h + kh, w + kw] - input_zero_point
                                kernel_val = kernel[oc, ic, kh, kw] - kernel_zero_points[oc]
                                acc += input_val * kernel_val
                    
                    fraction = kernel_scales[oc] * input_scale
                    if bias_quanitzed:
                        acc += (bias_q[oc]/fraction).to(torch.int32)
                        acc = float(acc)
                        acc *= fraction/output_scale
                    else:
                        acc = float(acc)
                        acc *= fraction
                        acc += bias_q[oc]
                        acc /= output_scale
                    
                    output[n, oc, h, w] = acc + output_zero_point

                    if relu and output[n, oc, h, w] < 0:
                        output[n, oc, h, w] = 0

    return output


# Precompute the quantized bias
def compute_quantized_bias_torch(bias, input_scale, kernel_scales):
    # Compute bias_q for each output channel
    bias_q = torch.round(bias / (input_scale * kernel_scales)).to(torch.int32)
    return bias_q


# Instantiate the model
model = SingleConvModel()

example_inputs = torch.randn(1, 1, 8, 8)
model = capture_pre_autograd_graph(model, example_inputs)
quantizer = X86InductorQuantizer()
quantizer.set_global(get_default_x86_inductor_quantization_config())
model = prepare_pt2e(model, quantizer)

for _ in range(10):
    #### Generate some random input data
    input_data = torch.randn(1, 1, 8, 8)
    input_data = (input_data*255.).type(torch.uint8)
    #### Calibrate the model with random input (or real input in practice)
    model(input_data/255.)


model = convert_pt2e(model)
model.print_readable()
ep = torch.export.export(model, (example_inputs,))
torch.export.save(ep, 'quantized_model.qt')

pytorch_output = model(input_data/255.0)

sate_dict = model.state_dict()


code = model.code
srchstr = 'per_tensor.default(arg0_1, ' 
st = code.find(srchstr) + len(srchstr)
code1 = code[st:]
et = code1.find(', 0, 255')
code2 = code1[:et]
mid = code2.find(', ')
input_scale_0 = float(code2[:mid])
input_zero_point_0 = int(code2[mid+2:])
output_scale_0 = 1
output_zero_point_0 = 0

conv_bias_q = sate_dict['conv_bias']

quantize_bias = False
manual_output_wo_qb = quantized_conv2d(input_data, sate_dict['_frozen_param0'], sate_dict['conv_zero_point_0'], 
                                       sate_dict['conv_scale_0'], sate_dict['conv_bias'],
                                       input_scale_0, input_zero_point_0, output_scale_0, output_zero_point_0,
                                       bias_quanitzed=quantize_bias)

quantize_bias = True
manual_output_w_qb = quantized_conv2d(input_data, sate_dict['_frozen_param0'], sate_dict['conv_zero_point_0'], 
                                       sate_dict['conv_scale_0'], sate_dict['conv_bias'],
                                       input_scale_0, input_zero_point_0, output_scale_0, output_zero_point_0,
                                       bias_quanitzed=quantize_bias)


diff_wo = torch.abs(manual_output_wo_qb-pytorch_output)
diff_w = torch.abs(manual_output_w_qb-pytorch_output)
dsum, dsum_w = torch.sum(diff_wo), torch.sum(diff_w)
dmean, dmean_w = torch.mean(diff_wo), torch.mean(diff_w)
dmax, dmax_w = torch.max(diff_wo), torch.max(diff_w)
dmin, dmin_w = torch.min(diff_wo), torch.min(diff_w)

print(f"manual quantization execution outputs are on sum: {dsum:0.6f} mean: {dmean:0.6f} max: {dmax:0.6f} min: {dmin:0.6f} different per output pixel with out bias quantization")
print(f"manual quantization execution outputs are on sum: {dsum_w:0.6f} mean: {dmean_w:0.6f} max: {dmax_w:0.6f} min: {dmin_w:0.6f} different per output pixel with bias quantization")

