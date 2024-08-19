# pt2e_qconv_py

### Overview
Pytorch 2.0's export feature is incredible, especially for creating highly efficient quantized CNNs. However, deploying these models can be tricky, particularly when working with quantized bias, as PyTorch currently offers this in `float32`. Understanding the underlying implementation in PyTorch, C++, and backend libraries like FBGEMM, OneDNN, or XNNPACK can be challenging, due to the layers of wrappers in the code.

This repository was created to help demystify how PyTorch backends handle quantized convolutions, enabling me to implement it on my device. While exploring these backend codes, I found them difficult to follow due to the extensive wrapping. Hence, this repository offers a more straightforward way to understand and experiment with quantization, especially focusing on bias quantization (which is typically kept as a float).

**Note:** This project requires PyTorch 2.4.0 or later.

### Purpose
The main goal of this repository is to assess the impact of quantized bias on performance. Spoiler alert: The impact is minimal. However, I encourage you to test it on your own applications and hardware to see how it performs.

### Usage

This repository provides two test files to explore the effects of quantized bias on CNN performance.

- **Single Convolution Test:** Run a simple model with a single convolution.
    ```bash
    python test_uno.py
    ```

- **Complex Convolution Test:** Run a more complex model with two convolutions and a ReLU layer in between, along with more channels.
    ```bash
    python test_duo.py
    ```

### Expected Output

Each test will output results comparing manual quantization with and without bias quantization. Example outputs might look like this:

```
Manual quantization execution outputs (without bias quantization):
Sum: 0.508191, Mean: 0.000529, Max: 0.002213, Min: 0.000000

Manual quantization execution outputs (with bias quantization):
Sum: 0.508172, Mean: 0.000529, Max: 0.002206, Min: 0.000000
```

### Conclusion
The repository serves as an exploratory tool for understanding the effects of bias quantization in quantized CNNs. While the impact on performance may be minimal, it provides a way to dive deeper into the inner workings of PyTorch's backend quantization mechanics.


