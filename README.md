# beaverpy :beaver:
### 

### Description
`beaverpy` is an implementation of PyTorch operators using only NumPy. <br>
Implemented operators (their PyTorch equivalents) include the following:
* Layers
   - `Conv2D` ([`torch.nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html))
   - `MaxPool2D` ([`torch.nn.MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)) 
   - `Linear` ([`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html))
* Loss/Distance Functions
   - `MSELoss` ([`torch.nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html))
   - `CosineSimilarity` ([`torch.nn.CosineSimilarity`](https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html))
* Activations
   - `ReLU` ([`torch.nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.htm))
   - `Sigmoid` ([`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)) 
   - `Softmax` ([`torch.nn.Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html))
> **Note 1:** `[n, c, h, w]` format is used

> **Note 2:** Test code that checks for correctness of the implementation is included in respective notebooks and is also available as standalone `pytest` scripts

### Optional parameters supported
* `Conv2D` &mdash; `stride`, `padding`, `dilation`, `groups`
* `MaxPool2D` &mdash; `stride`, `padding`, `dilation`, `return_indices`
* `Linear` &mdash; `bias`
* `MSELoss` &mdash; `reduction`
* `CosineSimilarity` &mdash; `dim`, `eps`
* `Softmax` &mdash; `dim`

### How to install?
```console

pip3 install beaverpy

```
### How to use?
<!-- :warning: Please note that this code is under development <br> -->

#### Import `beaverpy` and `numpy`
``` python

import beaverpy as bp
import numpy as np

```

#### Following is an example to use `Conv2D`: <br>

##### Define input parameters 
``` python

in_channels = 6 # input channels
out_channels = 4 # output channels
kernel_size = (2, 2) # kernel size

_stride = (2, 1) # stride (optional)
_padding = (1, 3) # padding (optional)
_dilation = (2, 3) # dilation factor (optional)
_groups = 2 # groups (optional)

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

conv2d = bp.Conv2D(in_channels, out_channels, kernel_size, stride = _stride, padding = _padding, dilation = _dilation, groups = _groups)

```
##### Perform convolution

``` python

_output = conv2d.forward(_input) # perform convolution

``` 
##### In case you wish to provide your own kernel, then define the same and pass it as an argument to `forward()` :

``` python

kernels = []
for k in range(out_channels):
    kernel = np.random.rand(int(in_channels / _groups), kernel_size[0], kernel_size[1]) # define a random kernel based on the kernel parameters
    kernels.append(kernel)
_output = conv2d.forward(_input, kernels) # perform convolution

```

#### Following is an example to use `MaxPool2D`: <br>

##### Define input parameters 
``` python

in_channels = 3 # input channels
kernel_size = (6, 6) # kernel size

_stride = (1, 5) # stride (optional)
_padding = (1, 2) # padding (optional)
_dilation = (2, 1) # dilation factor (optional)
_return_indices = True # return max indices (optional)

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

maxpool2d = bp.MaxPool2D(kernel_size, stride = _stride, padding = _padding, dilation = _dilation, return_indices = _return_indices)

```

##### Perform maxpooling

``` python

_output = maxpool2d.forward(_input)

``` 

#### Following is an example to use `Linear`: <br>

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

linear = bp.Linear(in_features, out_features)

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

#### Following is an example to use `MSELoss`: <br>

##### Create a random input and target
``` python

dimension = np.random.randint(500) # dimension of the input and target
_input = np.random.rand(dimension) # define a random input of the above dimension
_target= np.random.rand(dimension) # define a random target of the above dimension

```
##### Call an instance of `MSELoss` with the input parameters
``` python

mseloss = bp.MSELoss()

```
##### Compue MSE loss

``` python

_output = mseloss.forward(_input, _target)

``` 

#### Following is an example to use `CosineSimilarity`: <br>

##### Create random input
``` python

num_dim = np.random.randint(6) + 1 # number of input dimensions
shape = tuple(np.random.randint(5) + 1 for _ in range(num_dim)) # shape of input
_input1 = np.random.rand(*shape) # generate an input based on the dimensions and shape
_input2 = np.random.rand(*shape) # generate another input based on the dimensions and shape
_dim = np.random.randint(num_dim) # dimension along which CosineSimilarity is to be computed (optional)
_eps = np.random.uniform(low = 1e-10, high = 1e-6) # (optional)
        
```
##### Call an instance of `CosineSimilarity` with the input parameters
``` python

cosinesimilarity = bp.CosineSimilarity(dim = _dim, eps = _eps)

```
##### Compue CosineSimilarity

``` python

_output = cosinesimilarity.forward(_input1, _input2)

``` 

#### Following is an example to use `ReLU`: <br>

##### Create a random input
``` python

_input = np.random.rand(10, 20, 3)

```
##### Call an instance of `ReLU` with the input parameters
``` python

relu = bp.ReLU()

```

##### Apply ReLU activation

``` python

_output = relu.forward(_input)

``` 

#### Following is an example to use `Sigmoid`: <br>

##### Create a random input
``` python

_input = np.random.rand(10, 20, 3)

```
##### Call an instance of `Sigmoid` with the input parameters
``` python

sigmoid = bp.Sigmoid()

```
##### Apply Sigmoid activation

``` python

_output = sigmoid.forward(_input)

``` 

#### Following is an example to use `Softmax`: <br>

##### Create a random input and dimension 
``` python

_input = np.random.rand(1, 2, 1, 3, 4)
_dim = np.random.randint(len(_input)) # (optional)

```
##### Call an instance of `Softmax` with the input parameters
``` python

softmax = bp.Softmax(dim = _dim)

```
##### Apply Softmax activation

``` python

_output = softmax.forward(_input)

``` 

### Future work
* Replace `torch.round()` with `np.allclose()` for tests
* Implement other operators
* Optimize code

### Acknowledgements
This work is being done during my summer internship at <a href="https://www.degirum.ai/">DeGirum Corp.</a>, Santa Clara. 

### Using this code in your projects
* If you are using this code in your projects, please make sure to cite this repository and the author
* If you find bugs, create a pull request with a description of the bug and the proposed changes
* Do have a look at the <a href="https://ksanu1998.github.io/">author's webpage</a> for other interesting works!

`README` last updated on 06/08/2023
