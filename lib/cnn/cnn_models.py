from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            # here the params should correspond to the definition below:
            # B, H, W, iC = 4, 8, 8, 3 #batch, height, width, in_channels
            # k = 3 #kernel size
            # oC, Hi, O = 3, 27, 5 # out channels, Hidden Layer input, Output size
#             ConvLayer2D(input_channels=3,kernel_size=3,number_filters=3,name='TestCNN_conv'),
            ConvLayer2D(input_channels=3,kernel_size=3,number_filters=3,name='TestCNN_conv',dilation=2),
            MaxPoolingLayer(2,2,'TestCNN_maxpool'),
            flatten("TestCNN_flatten"),
            fc(27,5,0.02,'TestCNN_fc')
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            # option #1 - problem: reaching just short of 0.4 validation accuracy,
            # so need to increase number of filters in the 2nd convolutional layer
            
#             ConvLayer2D(input_channels=3,kernel_size=3,padding=1,number_filters=3,name='SCN_conv1'),
#             gelu(name='SCN_gelu1'),
#             ConvLayer2D(input_channels=3,kernel_size=3,stride=2,number_filters=3,name='SCN_conv2'),
#             gelu(name='SCN_gelu2'),
#             MaxPoolingLayer(2,2,'SCN_maxpool'),
#             flatten("SCN_flatten"),
#             fc(147,100,5e-2,'SCN_fc1'),
#             gelu(name='SCN_gelu3'),
#             fc(100,20,5e-2,'SCN_fc2')

            # option #2 - addresses the drawback in option #1    
            ConvLayer2D(input_channels=3,kernel_size=3,padding=1,number_filters=3,name='SCN_conv1'),
            gelu(name='SCN_gelu1'),
            ConvLayer2D(input_channels=3,kernel_size=3,stride=2,number_filters=9,name='SCN_conv2'),
            gelu(name='SCN_gelu2'),
            MaxPoolingLayer(2,2,'SCN_maxpool'),
            flatten("SCN_flatten"),
            fc(441,100,5e-2,'SCN_fc1'),
            gelu(name='SCN_gelu3'),
            fc(100,20,5e-2,'SCN_fc2')
            
            ########### END ###########
        )