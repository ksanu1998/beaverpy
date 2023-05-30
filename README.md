# conv-NumPy
## An implementation of 2D convolution using only NumPy, (being made) compatible with PyTorch `torch.nn.Conv2d`

### Description
* This code performs 2D convolution with the given input parameters
* It currently supports `stride`, `padding`, `dilation`, and `groups` options
* Tested for correctness with `torch.nn.functional.conv2d` and code included in the notebook
* Take a glance through this code to understand how convolution with different options works

### How to use
:warning: Please note that this code is under development <br><br>
Following is an example to use `Conv2D`, similar to `torch.nn.Conv2d`: <br>
#### Define input parameters 
``` python

in_channels = 6 # input channels
out_channels = 4 # output channels
kernel_size = (2, 2) # kernel size

padding = (1, 3) # padding (optional)
stride = (2, 1) # stride (optional)
dilation = (2, 3) # dilation factor (optional)
groups = 2 # groups (optional)

in_batches = 2 # input batches
in_h = 4 # input height
in_w = 4 # input weight

```
#### Create a random input using the input parameters
``` python

_input = np.random.rand(in_batches, in_channels, in_h, in_w) # define a random image based on the input parameters

```

#### Call an instance of `Conv2D` with the input parameters
``` python

conv2d = Conv2D(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups)

```

#### Perform convolution

``` python

_output = conv2d.forward(_input) # perform convolution

``` 
In case you wish to provide your own kernel, then define the same and pass it as an argument to `forward()` :

``` python

kernels = []
for k in range(out_channels):
    kernel = np.random.rand(int(in_channels / groups), kernel_size[0], kernel_size[1]) # define a random kernel based on the kernel parameters
    kernels.append(kernel)
_output = conv2d.forward(_input, kernels) # perform convolution

```

### Specifics
* `[n, c, h, w]` format is used
* For a description of the input parameters, refer to <a href="https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">PyTorch documentation</a> of `torch.nn.Conv2d`

### Future work
* Replace `torch.round()` with `np.allclose()` for tests
* Implement `padding = 'same'` mode
* Implement other features and caveats offered by `nn.torch.Conv2d` (e.g., uniform sampling of kernel weights, `bias` etc.)
* Implement other operators such as `MaxPool2d`, `ReLU` etc.
* Provide code insights
* Optimize code

### Acknowledgements
This work is being done during my summer internship at <a href="https://www.degirum.ai/">DeGirum Corp.</a>, Santa Clara. 

### Using this code for your projects
* If you are using this code, please make sure to cite this repository and the author
* If you find bugs, create a pull request with a description of the bug and the proposed changes (code optimization requests will not be entertained for now, for reasons that will be provided soon)
* Do have a look at the <a href="https://ksanu1998.github.io/">author's webpage</a> for other interesting works!
