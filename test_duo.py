import os
import pdb
import torch
import torch.nn as nn
import torch.quantization
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer, get_default_x86_inductor_quantization_config
from test_uno import quantized_conv2d

class DuelConvModel(nn.Module):
    def __init__(self):
        super(DuelConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.conv1(self.relu(self.conv(x)))
    

def get_scales_zero_points(code):
    srchstr = 'per_tensor.default(arg0_1, ' 
    st = code.find(srchstr) + len(srchstr)
    code1 = code[st:]
    et = code1.find(', 0, 255')
    code2 = code1[:et]
    mid = code2.find(', ')
    input_scale_0 = float(code2[:mid])
    input_zero_point_0 = int(code2[mid+2:])
    srchstr='per_tensor.default(quantize_per_tensor_default_1, '
    code = code[st+et:]
    st = code.find(srchstr) + len(srchstr) 
    code1 = code[st:]
    et = code1.find(', 0, 255')
    code2 = code1[:et]
    mid = code2.find(', ')
    output_scale_0 = input_scale_1 = float(code2[:mid])
    output_zero_point_0 = input_zero_point_1 = int(code2[mid+2:])
    output_scale_1 = 1
    output_zero_point_1 = 0
    return [input_scale_0, input_zero_point_0, output_scale_0, output_zero_point_0, input_scale_1, input_zero_point_1, output_scale_1, output_zero_point_1]


def main():
    # Instantiate the model
    model = DuelConvModel()
    example_inputs = torch.randn(1, 1, 8, 8)
    model = capture_pre_autograd_graph(model, example_inputs)
    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())
    model = prepare_pt2e(model, quantizer)

    for i in range(10):
        input_data = torch.randn(1, 1, 8, 8)
        input_data = (input_data*255.).to(torch.uint8)
        model(input_data/255.)


    model = convert_pt2e(model)
    model.print_readable()
    ep = torch.export.export(model, (example_inputs,))
    torch.export.save(ep, 'quantized_model.qt')

    pytorch_output = model(input_data/255.0)

    sate_dict = model.state_dict()


    scales_and_zeros = get_scales_zero_points(model.code)
    input_scale_0, input_zero_point_0, output_scale_0, output_zero_point_0 = scales_and_zeros[:4]
    input_scale_1, input_zero_point_1, output_scale_1, output_zero_point_1 = scales_and_zeros[4:]

    print('scales 0', input_scale_0, input_zero_point_0, output_scale_0, output_zero_point_0)
    print('scales 1', input_scale_1, input_zero_point_1, output_scale_1, output_zero_point_1)
    # # Run manual quantized convolution
    quantize_bias = False
    manual_output_0 = quantized_conv2d(input_data, sate_dict['_frozen_param0'], sate_dict['conv_zero_point_0'], sate_dict['conv_scale_0'], 
                                            sate_dict['conv_bias'], input_scale_0, input_zero_point_0, output_scale_0, output_zero_point_0, quantize_bias, relu=True)

    
    # manual_output_0 = torch.round(manual_output_0).to(torch.uint8)
    manual_output_wo_qb = quantized_conv2d(manual_output_0, sate_dict['_frozen_param1'], sate_dict['conv1_zero_point_0'], sate_dict['conv1_scale_0'], 
                                            sate_dict['conv1_bias'], input_scale_1, input_zero_point_1, output_scale_1, output_zero_point_1, quantize_bias)

    quantize_bias = True
    manual_output_0 = quantized_conv2d(input_data, sate_dict['_frozen_param0'], sate_dict['conv_zero_point_0'], sate_dict['conv_scale_0'], 
                                            sate_dict['conv_bias'], input_scale_0, input_zero_point_0, output_scale_0, output_zero_point_0, quantize_bias, relu=True)

    
    # manual_output_0 = torch.round(manual_output_0).to(torch.uint8)
    manual_output_w_qb = quantized_conv2d(manual_output_0, sate_dict['_frozen_param1'], sate_dict['conv1_zero_point_0'], sate_dict['conv1_scale_0'], 
                                            sate_dict['conv1_bias'], input_scale_1, input_zero_point_1, output_scale_1, output_zero_point_1, quantize_bias)




    diff_wo = torch.abs(manual_output_wo_qb-pytorch_output)
    diff_w = torch.abs(manual_output_w_qb-pytorch_output)
    dsum, dsum_w = torch.sum(diff_wo), torch.sum(diff_w)
    dmean, dmean_w = torch.mean(diff_wo), torch.mean(diff_w)
    dmax, dmax_w = torch.max(diff_wo), torch.max(diff_w)
    dmin, dmin_w = torch.min(diff_wo), torch.min(diff_w)

    print(f"manual quantization execution outputs are on sum: {dsum:0.6f} mean: {dmean:0.6f} max: {dmax:0.6f} min: {dmin:0.6f} different per output pixel with out bias quantization")
    print(f"manual quantization execution outputs are on sum: {dsum_w:0.6f} mean: {dmean_w:0.6f} max: {dmax_w:0.6f} min: {dmin_w:0.6f} different per output pixel with bias quantization")



if __name__ == '__main__':
    main()