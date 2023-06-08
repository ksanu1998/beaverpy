# beaverpy
## An implementation of some PyTorch operators using only NumPy
### Provides `Conv2D`, `MaxPool2D`, `Linear`, `MSELoss`, `CosineSimilarity`, `ReLU`, `Sigmoid`, and `Softmax`, that are compatible with `torch.nn.Conv2d`, `torch.nn.MaxPool2d`, `torch.nn.Linear`, `torch.nn.MSELoss`, `torch.nn.CosineSimilarity`, `torch.nn.ReLU`, `torch.nn.Sigmoid`, and `torch.nn.Softmax` respectively

### Description
* This repo is organized into `.ipynb` notebooks and `.py` modules - users can run the notebooks directly or call classes implemented in the modules
* `Conv2D` currently supports `stride`, `padding`, `dilation`, and `groups` options
* `Conv2D` is tested for correctness against `torch.nn.functional.conv2d`
* `MaxPool2D` currently supports `stride`, `padding`, `dilation`, and `return_indices` options
* `MaxPool2D` is tested for correctness against `torch.nn.MaxPool2d`
* `Linear` currently supports `bias` option
* `Linear` is tested for correctness against `torch.nn.functional.linear`
* `MSELoss` currently supports `reduction` option
* `MSELoss` is tested for correctness against `torch.nn.MSELoss`
* `CosineSimilarity` currently supports `dim`, and `eps` options
* `CosineSimilarity` is tested for correctness against `torch.nn.CosineSimilarity`
* `ReLU` is tested for correctness against `torch.nn.ReLU`
* `Sigmoid` is tested for correctness against `torch.nn.Sigmoid`
* `Softmax` currently supports `dim` option
* `Softmax` is tested for correctness against `torch.nn.Softmax`
* Test code that checks for correctness of the implementation is included in the respective notebooks and also available as standalone `Pytest` scripts
* Users can take a glance through the notebooks to gain an overview of the logic - code is not optimized

### How to use
:warning: Please note that this code is under development <br>
#### Following is an example to use `Conv2D`, similar to `torch.nn.Conv2d`: <br>
##### Define input parameters 
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
##### Create a random input using the input parameters
``` python

_input = np.random.rand(in_batches, in_channels, in_h, in_w)

```

##### Call an instance of `Conv2D` with the input parameters
``` python

conv2d = Conv2D(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups)

```

##### Perform convolution

``` python

_output = conv2d.forward(_input) # perform convolution

``` 
##### In case you wish to provide your own kernel, then define the same and pass it as an argument to `forward()` :

``` python

kernels = []
for k in range(out_channels):
    kernel = np.random.rand(int(in_channels / groups), kernel_size[0], kernel_size[1]) # define a random kernel based on the kernel parameters
    kernels.append(kernel)
_output = conv2d.forward(_input, kernels) # perform convolution

```

#### Following is an example to use `MaxPool2D`, similar to `torch.nn.MaxPool2d`: <br>
##### Define input parameters 
``` python

in_channels = 3 # input channels
kernel_size = (6, 6) # kernel size

padding = (1, 2) # padding (optional)
stride = (1, 5) # stride (optional)
dilation = (2, 1) # dilation factor (optional)
return_indices = True # return max indices (optional)

in_batches = 3 # input batches
in_h = 11 # input height
in_w = 8 # input weight

```
##### Create a random input using the input parameters
``` python

_input = np.random.rand(in_batches, in_channels, in_h, in_w)

```

##### Call an instance of `MaxPool2D` with the input parameters
``` python

maxpool2d = MaxPool2D(kernel_size, stride = stride, padding = padding, dilation = dilation)

```

##### Perform maxpooling

``` python

_output = maxpool2d.forward(_input)

``` 

#### Following is an example to use `Linear`, similar to `torch.nn.Linear`: <br>
##### Define input parameters 
``` python

in_samples = 128 # input samples
in_features = 20 # input features
out_features = 30 # output features

```
##### Create a random input using the input parameters
``` python

_input = np.random.rand(in_samples, in_features)

```

##### Call an instance of `Linear` with the input parameters
``` python

linear = Linear(in_features, out_features)

```

##### Apply a linear transformation

``` python

_output = linear.forward(_input)

``` 
##### In case you wish to provide your own weights and bias, then define the same and pass them as arguments to `forward()` :

``` python

_weights = np.random.rand(out_features, in_features) # define random weights
_bias = np.random.rand(out_features) # define random bias
_output = linear.forward(_input, weights = _weights, bias_weights = _bias) # apply linear transformation

```

#### Following is an example to use `MSELoss`, similar to `torch.nn.MSELoss`: <br>
##### Create a random input and target
``` python

dimension = np.random.randint(500) # dimension of the input and target
_input = np.random.rand(dimension) # define a random input of the above dimension
_target= np.random.rand(dimension) # define a random target of the above dimension

```
##### Call an instance of `MSELoss` with the input parameters
``` python

mseloss = MSELoss()

```

##### Compue MSE loss

``` python

_output = mseloss.forward(_input, _target)

``` 

#### Following is an example to use `CosineSimilarity`, similar to `torch.nn.CosineSimilarity`: <br>
##### Create random input
``` python

num_dim = np.random.randint(6) + 1 # number of input dimensions
shape = tuple(np.random.randint(5) + 1 for _ in range(num_dim)) # shape of input
_input1 = np.random.rand(*shape) # generate an input based on the dimensions and shape
_input2 = np.random.rand(*shape) # generate another input based on the dimensions and shape
_dim = np.random.randint(num_dim) # dimension along which CosineSimilarity is to be computed
_eps = np.random.uniform(low = 1e-10, high = 1e-6)
        
```
##### Call an instance of `CosineSimilarity` with the input parameters
``` python

cosinesimilarity = CosineSimilarity(dim = _dim, eps = _eps)

```

##### Compue CosineSimilarity

``` python

_output = cosinesimilarity.forward(_input1, _input2)

``` 

#### Following is an example to use `ReLU`, similar to `torch.nn.ReLU`: <br>
##### Create a random input
``` python

_input = np.random.rand(10, 20, 3)

```
##### Call an instance of `ReLU` with the input parameters
``` python

relu = ReLU()

```

##### Apply ReLU activation

``` python

_output = relu.forward(_input)

``` 

#### Following is an example to use `Sigmoid`, similar to `torch.nn.Sigmoid`: <br>
##### Create a random input
``` python

_input = np.random.rand(10, 20, 3)

```
##### Call an instance of `Sigmoid` with the input parameters
``` python

sigmoid = Sigmoid()

```

##### Apply Sigmoid activation

``` python

_output = sigmoid.forward(_input)

``` 

#### Following is an example to use `Softmax`, similar to `torch.nn.Softmax`: <br>
##### Create a random input and dimension 
``` python

_input = np.random.rand(1, 2, 1, 3, 4)
_dim = np.random.randint(len(_input))

```
##### Call an instance of `Softmax` with the input parameters
``` python

softmax = Softmax(dim = _dim)

```

##### Apply Softmax activation

``` python

_output = softmax.forward(_input)

``` 

### Specifics
* `[n, c, h, w]` format is used
* For a description of the input parameters, refer to PyTorch documentation of <a href="https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">`torch.nn.Conv2d`</a>, <a href="https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html">`torch.nn.MaxPool2d`</a>, <a href="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear">`torch.nn.Linear`</a>, <a href="https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html">`torch.nn.MSELoss`</a>, <a href="https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html">`torch.nn.CosineSimilarity`</a>, <a href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html">`torch.nn.ReLU`</a>, <a href="https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid">`torch.nn.Sigmoid`</a>, and <a href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html">`torch.nn.Softmax`</a>

### Future work
* Replace `torch.round()` with `np.allclose()` for tests
* Implement other operators
* Optimize code
* Implement `padding = 'same'` mode for `Conv2D`
* Implement `ceil_mode` for `MaxPool2D`
* Provide code insights

### Acknowledgements
This work is being done during my summer internship at <a href="https://www.degirum.ai/">DeGirum Corp.</a>, Santa Clara. 

### Using this code for your projects
* If you are using this code, please make sure to cite this repository and the author
* If you find bugs, create a pull request with a description of the bug and the proposed changes (code optimization requests will not be entertained for now, for reasons that will be provided soon)
* Do have a look at the <a href="https://ksanu1998.github.io/">author's webpage</a> for other interesting works!

`README` last updated on 06/08/2023
