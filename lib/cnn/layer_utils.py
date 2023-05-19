from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pprint import pprint

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                param = self.params[n]
                grad = (param > 0).astype(np.float32) - (param < 0).astype(np.float32)
                self.grads[n] += lam * grad

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] += lam * self.params[n]


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
#     def __init__(self, input_channels, kernel_size, number_filters, stride=1, padding=0, init_scale=.02, name="conv"):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv", dilation=1):
        # add dilation parameter to the method definition
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding
        # add self.dilation = dilation
        self.dilation = dilation
        
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        # add the dilation code here -> take the initial kernel, dilate it, and rewrite into self.params[self.w_name]
        # cross-check the original kernel with dilation = 1 -> both of them should be the same
        self.dilate_kernel()
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    # method to dilate a kernel
    def dilate_kernel(self):
#         print(self.params[self.w_name])
#         print(self.params[self.w_name].shape)
        dil_kernel_size = self.dilation*(self.kernel_size-1)+1
        dil_kernel = np.zeros((dil_kernel_size,dil_kernel_size),dtype=self.params[self.w_name].dtype)
#         print(self.params[self.w_name].reshape(self.kernel_size, -1).shape)
        kernels = self.params[self.w_name].reshape(-1, self.kernel_size, self.kernel_size)
        print("original kernels")
        print(kernels)
        # self.dilation*x_pos is the new location of x_pos
        dilated_kernels = []
        for kernel in kernels:
            dilated_kernel = np.zeros((dil_kernel_size,dil_kernel_size),dtype=self.params[self.w_name].dtype)
            for row in range(len(kernel)):
                for col in range(len(kernel[0])):
                    dilated_kernel[self.dilation*row][self.dilation*col] = kernel[row][col]
            dilated_kernels.append(dilated_kernel)
        dilated_kernels = np.asarray(dilated_kernels, dtype=np.float64).reshape(dil_kernel_size, dil_kernel_size, self.input_channels, self.number_filters)
        print("dilated kernels")
        print(dilated_kernels)
        print(dilated_kernels.shape)
#         print(kernels == dilated_kernels)
        self.params[self.w_name] = dilated_kernels
        
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        output_shape[0] = input_size[0]
#         output_shape[1] = int(((input_size[1] + 2 * self.padding - self.kernel_size)/self.stride)+1)
#         output_shape[2] = int(((input_size[2] + 2 * self.padding - self.kernel_size)/self.stride)+1)
        # kernel size changes to dilated kernel size
        output_shape[1] = int(((input_size[1] + 2 * self.padding - (self.dilation*(self.kernel_size-1)+1))/self.stride)+1)
        output_shape[2] = int(((input_size[2] + 2 * self.padding - (self.dilation*(self.kernel_size-1)+1))/self.stride)+1)
        
        output_shape[3] = self.number_filters
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)

        output_shape = self.get_output_size(img.shape)
        _ , input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        # print statistics
        '''
        print("input_height, input_width ", input_height, input_width)
        print("output_height, output_width ", output_height, output_width)
        print("self.kernel_size ", self.kernel_size)
        print("self.stride ", self.stride)
        print("self.padding ", self.padding)
        print("num_batches ", len(img))
        print("self.input_channels ", self.input_channels)
        '''
        input_img = np.pad(img, ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0))) # padding the image first
        output = np.zeros(output_shape) # creating a dummy output image
        
        # an efficient implementation of convolution without using for loops (but works for only input_channels = 1)!
        # source of idea: https://medium.com/analytics-vidhya/implementing-convolution-without-for-loops-in-numpy-ce111322a7cd
        
        # idea in a nutshell
        # say the image is as follows:
        # a00 a01 a02 a03
        # a10 a11 a12 a13
        # a20 a21 a22 a23
        # a30 a31 a32 a33
        # and the kernel is as follows:
        # k00 k01 k02
        # k10 k11 k12
        # k20 k21 k22
        # assume stride = 1
        # we first create a matrix where squash each receptive field of the image
        # corresponding to one pass kernel into a single column, as shown below:
        # a00 a01 a10 a11
        # a01 a02 a11 a12
        # a02 a03 a12 a13
        # a10 a11 a20 a21
        # a11 a12 a21 a22
        # a12 a13 a22 a23
        # a20 a21 a30 a31
        # a21 a22 a31 a32
        # a22 a21 a32 a33
        # the above is achieved by first creating the matrix indices 00, 01, ... and the slicing the image based on these indices
        # we next squash the kernel into a single column as follows:
        # k00 
        # k01
        # k02
        # k10
        # k11
        # k12
        # k20
        # k21
        # k22
        # we then take the matrix product of the above two to get the required convolution and reshape it back to required shape
        
        if self.input_channels == 1:
            i0=np.repeat(np.arange(self.kernel_size), self.kernel_size)
            i1=np.repeat(np.arange(output_height), output_height)
            j0=np.tile(np.arange(self.kernel_size), self.kernel_size)
            j1=np.tile(np.arange(output_height), output_width)
            i=i0.reshape(-1,1)+i1.reshape(1,-1)
            j=j0.reshape(-1,1)+j1.reshape(1,-1)
            input_img_slice=input_img[:,i,j,:,np.newaxis].squeeze()\
            .reshape(self.kernel_size*self.kernel_size*self.input_channels,-1)
            weights=self.params[self.w_name].reshape(self.kernel_size*self.kernel_size*self.input_channels, -1) 
            convolve=input_img_slice.T@weights
            output = convolve.reshape(output_shape)
            output += self.params[self.b_name]
        
        else:
            # a not-so-efficient implementation of convolution using two for-loops (but works for all input_channels)
            for h in range(output_height):
                for w in range(output_width):
                    # vertical start index of the input image slice
                    v_begin = self.stride * h 
                    # vertical end index of the input image slice (vertical start index + kernel_size)
                    v_end = v_begin + self.kernel_size 
                    # horizontal start index of the input_image slice
                    h_begin = self.stride * w 
                    # horizontal end index of the input_image slice (horizontal start index + kernel_size)
                    h_end = h_begin + self.kernel_size 
                    # creating input image slice
                    input_img_slice = input_img[:,v_begin:v_end, h_begin:h_end,:, np.newaxis] 
                    # taking hadamard product of the input image slice with the kernel and
                    # summing them up to get a pixel value for the output
                    convol = np.sum(input_img_slice * self.params[self.w_name][np.newaxis:,:,:], axis=(1,2,3)) 
                    output[:,h,w,:] = convol
            output += self.params[self.b_name]
      
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        return output


    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        num_batches, num_height, num_width, num_channels = dprev.shape
        # creating a dummy gradient of the shape of kernel weights
        self.grads[self.w_name] = np.zeros((self.kernel_size,self.kernel_size,self.input_channels,self.number_filters))
        # creating a dummy gradient of the shape of kernel biases
        self.grads[self.b_name] = np.zeros(self.number_filters)
        # create a dummy delta img
        dimg = np.zeros_like(img)
        # pad delta img
        dimg_pad = np.pad(dimg,((0,0),(self.padding,self.padding),(self.padding,self.padding),\
                                (0,0)),mode='constant',constant_values=(0,0))
        # pad original image
        img_pad = np.pad(img,((0,0),(self.padding,self.padding),(self.padding,self.padding),\
                              (0,0)),mode='constant',constant_values=(0,0))
        # computing gradient w.r.t. bias term
        self.grads[self.b_name] = np.sum(dprev,axis=(0,1,2)) 
        for h in range(num_height):
            for w in range(num_width):
                # vertical start index of the input image slice
                v_begin = self.stride * h 
                # vertical end index of the input image slice (vertical start index + kernel_size)
                v_end = v_begin + self.kernel_size 
                # horizontal start index of the input_image slice
                h_begin = self.stride * w 
                # horizontal end index of the input_image slice (horizontal start index + kernel_size)
                h_end = h_begin + self.kernel_size 
                # creating input image slice
                input_img_slice = img_pad[:,v_begin:v_end, h_begin:h_end,:, np.newaxis] 
                # taking hadamard product of the input image slice (gradient of the kernel) with gradient of the previous
                # layer and summing them up to get a pixel value for the output
                self.grads[self.w_name] += np.sum(input_img_slice*dprev[:,h:h+1,w:w+1,np.newaxis,:],axis=0)
                dimg_pad[:,v_begin:v_end, h_begin:h_end,:] += np.sum(self.params[self.w_name] \
                                                                   [np.newaxis,:,:,:,:]*dprev[:,h:h+1,w:w+1,np.newaxis,:],axis=4)
            dimg = dimg_pad[:,self.padding:img.shape[1]+self.padding,self.padding:img.shape[1]+self.padding,:]
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        
        input_size = img.shape
        def get_output_size(input_size):   
            output_shape = [None, None, None, None]
            output_shape[0] = input_size[0]
            output_shape[1] = int(((input_size[1] - self.pool_size)/self.stride)+1)
            output_shape[2] = int(((input_size[2] - self.pool_size)/self.stride)+1)
            output_shape[3] = input_size[3]
            return output_shape
        
        # get input image shape
        _ , input_height, input_width, _ = img.shape
        # compute and get maxpooled image shape
        output_shape = get_output_size(input_size)
        _, output_height, output_width, _ = output_shape
        
        output = np.zeros(output_shape) # creating a dummy output maxpooled image of shape output_shape
        # this method will be almost the same as convolution except for:
        # 1) there's no covolution, we rather take max of the slice
        # 2) there's no bias term
        for h in range(output_height):
            for w in range(output_width):
                # vertical start index of the input image slice
                v_begin = self.stride * h 
                # vertical end index of the input image slice (vertical start index + kernel_size)
                v_end = v_begin + self.pool_size 
                # horizontal start index of the input_image slice
                h_begin = self.stride * w 
                # horizontal end index of the input_image slice (horizontal start index + kernel_size)
                h_end = h_begin + self.pool_size 
                # creating input image slice
                input_img_slice = img[:,v_begin:v_end, h_begin:h_end,:] 
                # taking max of the slice
                maxpool = np.max(input_img_slice, axis=(1,2)) 
                output[:,h,w,:] = maxpool
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        return output

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out, _ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size

        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        
        for h in range(h_out):
            for w in range(w_out):
                # vertical start index of the input image slice
                v_begin = self.stride * h 
                # vertical end index of the input image slice (vertical start index + pooling filter size)
                v_end = v_begin + h_pool
                # horizontal start index of the input_image slice
                h_begin = self.stride * w 
                # horizontal end index of the input_image slice (horizontal start index + pooling filter size)
                h_end = h_begin + w_pool
                # creating input image slice
                input_img_slice = img[:,v_begin:v_end, h_begin:h_end,:] 
                # compute maxpooled image of the shape of the gradient of the previous layer
                maxpool_val = np.argmax(input_img_slice.reshape(dprev.shape[0],h_pool * w_pool, dprev.shape[-1]),axis=1)
                # create a mask so that it corresponds to only that slice of the image that has been maxpooled
                # and take hadamard product of the mask with the gradient of the previous layer (no weights learnt 
                # in this layer so no gradients of this layer!)
                mask = np.zeros_like(input_img_slice)
                n , c = np.indices((input_img_slice.shape[0],input_img_slice.shape[-1]))
                mask.reshape(dprev.shape[0],h_pool * w_pool, dprev.shape[-1])[n,maxpool_val,c] = 1
                dimg[:,v_begin:v_end, h_begin:h_end,:] += mask * dprev[:,h:h+1,w:w+1,:]
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
