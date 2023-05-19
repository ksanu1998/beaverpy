from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
                ######## TODO ########
                layer.grads[n] = v + lam*np.sign(layer.params[n]) 
                # as gradient of l1 norm of a vector is the absolute value of each component  
                # i.e., d(|w1|+...+|wn|)/dx = (w1/|w1|,...,wn/|wn|)
                # also note that v is gradient whereas layer.params[n] gives the weight vector itself 
                pass
                ######## END  ########
    
    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                layer.grads[n] = v + 2*lam*layer.params[n] 
                pass
                ######## END  ########


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


class flatten(object):
    def __init__(self, name="flatten"):
        """
        - name: the name of current layer
        - meta: to store the forward pass activations for computing backpropagation
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of a flatten layer.                      #
        # You need to reshape (flatten) the input features.                         #
        # Store the results in the variable self.meta provided above.               #
        #############################################################################
        # flattens features batchwise (3 batches) into (7*6*4)
        ''' # contains one for loop per batch
        flattened_list_batchwise = []
        for batch in feat:
            flattened_list_batchwise.append([inner for outer in batch for middle in outer for inner in middle])
        feat = np.array(flattened_list_batchwise)
        output = feat
        '''
        # better code
        output = feat.reshape(feat.shape[0], np.prod(feat.shape[1:]))
        ''' # sanity checks
        print("Number of batches in flattened features: ", len(feat))
        print("Number of params in flattened features: ", len(feat[0]))
        '''
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of a flatten layer.                     #
        # You need to reshape (flatten) the input gradients and return.             #
        # Store the results in the variable dfeat provided above.                   #
        #############################################################################
        # just reshape the partial derivatives of the previous layer
        dfeat = dprev.reshape(feat.shape)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat


class fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.002, name="fc"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def forward(self, feat):
        output = None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        #############################################################################
        # TODO: Implement the forward pass of a single fully connected layer.       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        # perform wx+b here for every flattened batch of features
        ''' # sanity checks
        print("features", feat[0].shape)
        print("weights", self.params[self.w_name].shape)
        print("biases", self.params[self.b_name])
        print("features*weights", self.params[self.w_name].T@feat[0].T+self.params[self.b_name])
        '''
        ''' # contains one for loop iterating over each batch
        output = []
        for batch in feat:
            output.append(np.array(self.params[self.w_name].T @ batch.T + self.params[self.b_name]))
        output = np.array(output)
        ''' 
        # better code
        output = np.array(feat @ self.params[self.w_name] + self.params[self.b_name])
        '''
        print(output)
        '''
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        assert len(feat.shape) == 2 and feat.shape[-1] == self.input_dim, \
            "But got {} and {}".format(feat.shape, self.input_dim)
        assert len(dprev.shape) == 2 and dprev.shape[-1] == self.output_dim, \
            "But got {} and {}".format(dprev.shape, self.output_dim)
        #############################################################################
        # TODO: Implement the backward pass of a single fully connected layer.      #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        # fc(w,b,x) = wx+b
        # dfc(w,b,x)/dw = x
        # dfc(w,b,x)/db = 1
        # dfc(w,b,x)/dx = w
        # dfc(w,b,x)/dw, dfc(w,b,x)/dw and dfc(w,b,x)/dx should be multiplied by the corresponding
        # partial derivatives of the prev (next) layer
        self.grads[self.w_name] = np.array(feat.T@dprev)
        self.grads[self.b_name] = np.array(np.ones((1, feat.shape[0]))@dprev).flatten()
        dfeat = np.array(dprev@self.params[self.w_name].T)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat

class gelu(object):
    def __init__(self, name="gelu"):
        """
        - name: the name of current layer
        - meta:  to store the forward pass activations for computing backpropagation
        Notes: params and grads should be just empty dicts here, do not update them
        """
        self.name = name 
        self.params = {}
        self.grads = {}
        self.meta = None 
    
    def forward(self, feat):
        output = None
        #############################################################################
        # TODO: Implement the forward pass of GeLU                                  #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        # gelu(x) = 0.5x(1+tanh(g(x))) where g(x)=√2/√𝜋(x+0.044715x^3)
        output = 0.5*feat*(1+np.tanh(np.sqrt(2/np.pi)*(feat+(0.044715*(feat**3)))))
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = feat
        return output
    
    def backward(self, dprev):
        """ You can use the approximate gradient for GeLU activations """
        feat = self.meta
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        dfeat = None
        #############################################################################
        # TODO: Implement the backward pass of GeLU                                 #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        # splitting the function and its derivative into components for clarity
        # gelu(x) = 0.5x(1+tanh(g(x))) where g(x)=√2/√𝜋(x+0.044715x^3)
        const_x3 = 0.044715
        f_tanhx = np.tanh(np.sqrt(2./np.pi)*(feat+(const_x3*feat**3))) # tanh(g(x))
        f_sech2x = 1.-(f_tanhx**2) # sech()^2 = 1-tanh()^2
        del_f_tanhx =f_sech2x*np.sqrt(2./np.pi)*(1.+3*const_x3*(feat**2)) # d/dx(tanh(g(x))) = sech(g(x))^2*g'(x)
        dfeat = (1.+f_tanhx)+(feat*(del_f_tanhx))
        dfeat = 0.5*dfeat*dprev
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = None
        return dfeat



class dropout(object):
    def __init__(self, keep_prob, seed=None, name="dropout"):
        """
        - name: the name of current layer
        - keep_prob: probability that each element is kept.
        - meta: to store the forward pass activations for computing backpropagation
        - kept: the mask for dropping out the neurons
        - is_training: dropout behaves differently during training and testing, use
                       this to indicate which phase is the current one
        - rng: numpy random number generator using the given seed
        Note: params and grads should be just empty dicts here, do not update them
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.keep_prob = keep_prob
        self.meta = None
        self.kept = None
        self.is_training = False
        self.rng = np.random.RandomState(seed)
        assert keep_prob >= 0 and keep_prob <= 1, "Keep Prob = {} is not within [0, 1]".format(keep_prob)

    def forward(self, feat, is_training=True, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        kept = None
        output = None
        #############################################################################
        # TODO: Implement the forward pass of Dropout.                              #
        # Remember if the keep_prob = 0, there is no dropout.                       #
        # Use self.rng to generate random numbers.                                  #
        # During training, need to scale values with (1 / keep_prob).               #
        # Store the mask in the variable kept provided above.                       #
        # Store the results in the variable output provided above.                  #
        #############################################################################
        # consider doing the below only during training and when keep probability is not 0 (as keep 
        # probability = 0 => no dropout)
        if is_training and self.keep_prob: 
            # if random number generated is less than the keep probability, 
            # then mask that element as 0, or otherwise 1
            kept = (self.rng.rand(feat.shape[0], feat.shape[1]) <= self.keep_prob).astype(float) 
            kept *=  1.0/self.keep_prob # scale the retained elements with inverse of keep probability
        else:
            # no dropout => mask every element as 1
            kept = np.ones(feat.shape).astype(float)   
        # get the Hadamard product of the mask and the features
        output = np.multiply(kept, feat)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.kept = kept
        self.is_training = is_training
        self.meta = feat
        return output

    def backward(self, dprev):
        feat = self.meta
        dfeat = None
        if feat is None:
            raise ValueError("No forward function called before for this module!")
        #############################################################################
        # TODO: Implement the backward pass of Dropout                              #
        # Select gradients only from selected activations.                          #
        # Store the output gradients in the variable dfeat provided above.          #
        #############################################################################
        dfeat = self.kept*dprev
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.is_training = False
        self.meta = None
        return dfeat


class cross_entropy(object):
    def __init__(self, size_average=True):
        """
        - size_average: if dividing by the batch size or not
        - logit: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.size_average = size_average
        self.logit = None
        self.label = None

    def forward(self, feat, label):
        logit = softmax(feat)
        loss = None
        #############################################################################
        # TODO: Implement the forward pass of an CE Loss                            #
        # Store the loss in the variable loss provided above.                       #
        #############################################################################
        # celoss(yi=j,xi) = -\sum_{i=1}^{N}\log\sigma(x_{ij})
        loss = 0
        for true,row in zip(label,logit):
            loss += -np.log(row[true])
        if self.size_average:
            loss /= label.shape[0]
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = logit
        self.label = label
        return loss

    def backward(self):
        logit = self.logit
        label = self.label
        if logit is None:
            raise ValueError("No forward function called before for this module!")
        dlogit = None
        #############################################################################
        # TODO: Implement the backward pass of an CE Loss                           #
        # Store the output gradients in the variable dlogit provided above.         #
        #############################################################################
        # celoss(yi=j,xi) = -\sum_{i=1}^{N}\log\sigma(x_{ij})
        # \nabla celoss(yi=j,xi) = \sigma(x_{ij}) - 1
        dlogit = []
        for true,row in zip(label,logit):
            row[true] -= 1
            dlogit.append(row)
        dlogit = np.array(dlogit)
        if self.size_average:
            dlogit /= label.shape[0]
        
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.logit = None
        self.label = None
        return dlogit


def softmax(feat):
    scores = None

    #############################################################################
    # TODO: Implement the forward pass of a softmax function                    #
    # Return softmax values over the last dimension of feat.                    #
    #############################################################################
    # \sigma(a) = \dfrac{e^{a_{j}-\max{a}}}{\sum_{i=1}^{N}e^{a_{j}-\max{a}}}
    # subtracting \max{a} for stability of computation as exponentiation may lead to overflow
    scores = []
    for row in feat:
        row_max = max(row)
        row = np.array([np.exp(ele-row_max) for ele in row])
        row /= np.sum(row)
        scores.append(row)
    scores = np.array(scores)
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return scores

def reset_seed(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
