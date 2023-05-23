# conv-NumPy
## Implementation of 2D convolution using only NumPy, (being made) compatible with PyTorch `torch.nn.Conv2D`

### Description
* This code performs different types of 2D convolutions given the input and kernel parameters. It currently supports `Conv2D` and `DepthwiseConv2D` with stride, kernel dilation and image padding options
* The class implementation will be modified along the lines of `torch.nn.Conv2D` in order to enable comparison on this NumPy only implementation with that of PyTorch one, and seamless testing
* Can expect extensive refactoring of the existing code in the days to come
* As part of refactoring, some code will be de-modularized"

### How to use
* :warning: Please note that this code is under development
* Create an instance of `Conv2D` or `DepthwiseConv2D` class with the following parameters: `inp_n`, `inp_c`, `inp_h`, `inp_w`, `ker_c`, `ker_h`, `ker_h`, `ker_w` and `num_ker`
* `stride`, `dilation` and `padding` are optional parameters with default values as follows: `stride = 1`, `dilation = 1` (no dilation) and `padding=0` (no padding)
* Call the methods `create_input_img()`, `create_kernels()` and `create_output()` in the order

### Specifics
* `create_input_img()` creates a random input of shape `[inp_n, inp_c, inp_h, inp_w]`
* `create_kernels()` creates random kernels of shape `[num_ker, ker_c, ker_h, ker_w]`
* `create_output()` performs the convolution and gives output of shape `[inp_n, num_ker, (inp_h + 2*padding - ker_h)/stride + 1, (inp_w + 2*padding - ker_w)/stride + 1]`, where `inp_h`, `inp_w` are updated based on `padding` and `ker_h`, `ker_w` are updated based on `dilation`
* Note that `ker_c = 1` and `inp_c = num_ker = out_c` always, for `DepthwiseConv2D`
### Development
* Enable verbose and debug modes by setting `verbose=True` and `debug=True` respectively

### Future work
* Optimize code
* Add code for different types of convolutions

### Acknowledgements
This work is being done during my summer internship at DeGirum Corp., Santa Clara.
