#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:38:31 2019

@author: berend
"""


from keras.layers import Conv2D,Conv2DTranspose, BatchNormalization, Input, add, ReLU

from keras.models import Model



    
    
    
def block_1(x, use_bn = False):
    
    kernel_initializer = 'glorot_uniform'
    
    x = Conv2D(32, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)
    if use_bn:
        x = BatchNormalization(axis = -1)(x)
    x = ReLU()(x)

    x = Conv2D(32, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)
    if use_bn:
        x = BatchNormalization(axis = -1)(x)
    x = ReLU()(x)

    return x
    
    
    
def res_block(x, channels, downsample = False, expand_channels = 4, use_bn = False):
    #residual:
    res = x
    kernel_initializer = 'glorot_uniform'

    #first convolution
    x = Conv2D(channels, kernel_size = (1,1), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)
    if use_bn:
        x = BatchNormalization(axis = -1)(x)
    x = ReLU()(x)


    #second: with or without downsampling
    if downsample:
        x = Conv2D(channels, kernel_size = (3,3), strides=(2,2), padding = 'same', kernel_initializer = kernel_initializer)(x)
        res = Conv2D(channels*expand_channels, kernel_size = (1,1), strides=(2,2), padding = 'same', kernel_initializer = kernel_initializer)(res)
        if use_bn:
            res = BatchNormalization(axis = -1)(res)
    else:
        x = Conv2D(channels, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)
    if use_bn:
        x = BatchNormalization(axis = -1)(x)
    x = ReLU()(x)
    
    
    
    #last convolution
    x = Conv2D(channels*expand_channels, kernel_size = (1,1), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)
    if use_bn:
        x = BatchNormalization(axis = -1)(x)
        
    #add the residual and return:
    return ReLU()(add([x,res]))  


def make_block(x, res_blocks, channels, use_bn = False, expand_channels = 4):
    
    #first res block:
    x = res_block(x, channels, downsample = True, expand_channels = expand_channels, use_bn = use_bn)

    for i in range(1, res_blocks):
        x = res_block(x, channels, downsample = False, expand_channels = expand_channels, use_bn = use_bn)
        
    return x



def latteral_and_output(c3, c4, c5):

    kernel_initializer = 'glorot_uniform'
    latlayer1 = Conv2D(196, kernel_size = (1,1), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)
    latlayer2 = Conv2D(128, kernel_size = (1,1), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)
    latlayer3 = Conv2D(96, kernel_size = (1,1), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)
    
    
    deconv1 = Conv2DTranspose(128, kernel_size = (3,3), strides=(2,2), padding = 'same', output_padding = (1,1))
    deconv2 = Conv2DTranspose(96, kernel_size = (3,3), strides=(2,2), padding = 'same', output_padding = (1,1))


    l5 = latlayer1(c5)
    l4 = latlayer2(c4)
    p5 = add([l4,deconv1(l5)])
    l3 = latlayer3(c3)
    p4 = add([l3,deconv2(p5)])  
    
    return p4



def make_backbone(x, num_res_per_block, channels_per_block, use_bn = False, expand_channels = 4):
    
    c1 = block_1(x, use_bn = use_bn)

    c2 = make_block(c1, num_res_per_block[0], channels_per_block[0], use_bn = use_bn, expand_channels = expand_channels)
    c3 = make_block(c2, num_res_per_block[1], channels_per_block[1], use_bn = use_bn, expand_channels = expand_channels)
    c4 = make_block(c3, num_res_per_block[2], channels_per_block[2], use_bn = use_bn, expand_channels = expand_channels)
    c5 = make_block(c4, num_res_per_block[3], channels_per_block[3], use_bn = use_bn, expand_channels = expand_channels)
        
    p4 = latteral_and_output(c3,c4,c5)

    return p4
    

def make_cls_head(x, n_classes):

    kernel_initializer = 'glorot_uniform'
    
    cls_pred = Conv2D(n_classes, 
               kernel_size = (3,3), 
               strides=(1,1), padding = 'same', 
               kernel_initializer = kernel_initializer,
               activation = 'softmax',
               name='class_output')(x)    
    
    return cls_pred

def make_reg_head(x):

    kernel_initializer = 'glorot_uniform'
    
    x = Conv2D(8, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer,name='reg_output')(x) 
    
    return x

def make_header(x, n_classes, use_bn = False):

    kernel_initializer = 'glorot_uniform'
    #1    
    x = Conv2D(96, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)    
    if use_bn:
        x = BatchNormalization(axis = -1)(x)

    #2    
    x = Conv2D(96, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)    
    if use_bn:
        x = BatchNormalization(axis = -1)(x)
        
    #3    
    x = Conv2D(96, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)    
    if use_bn:
        x = BatchNormalization(axis = -1)(x)
        
    #4    
    x = Conv2D(96, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)    
    if use_bn:
        x = BatchNormalization(axis = -1)(x)        
        
    #5
    x = Conv2D(96, kernel_size = (3,3), strides=(1,1), padding = 'same', kernel_initializer = kernel_initializer)(x)    
    if use_bn:
        x = BatchNormalization(axis = -1)(x) 
        

    cls_pred = make_cls_head(x, n_classes)
    
    reg = make_reg_head(x)

    return cls_pred, reg
    



def make_network(input_shape,n_classes, use_bn = False, expand_channels = 4):
    
    inp = Input(shape = input_shape)

    p4 = make_backbone(inp,
                       [3, 6, 6, 3],
                       [24, 48, 64, 96],
                       use_bn = use_bn,
                       expand_channels = expand_channels
                       )
    
    cls_pred, reg = make_header(p4, n_classes, use_bn = use_bn)
    
    return inp, cls_pred, reg
    
    
def get_model(input_shape,n_classes, use_bn = False, expand_channels = 4):
    
    
    inp, cls_pred, reg = make_network(input_shape,n_classes, use_bn = use_bn, expand_channels = expand_channels)
    
    model = Model(inp,[cls_pred,reg])
    
    #note: need to update this to a more appropriate lossfunction:
    # basically the regression loss is calculated where the category is non-zero, so only where an object is
    
    model.compile(
                  optimizer='adam',
                  loss={'class_output': 'categorical_crossentropy', 'reg_output': 'logcosh'},
                  loss_weights={'class_output': 1., 'reg_output': 1.})
    
    return model
    


## THere is backbone, which consists of a bottleneck
##

##  Of 3, 6, 6, 3  res_blocks
##    24, 48, 64, 96  output channels
        
# block1: 
        # conv13x3(36,32)
        # if bn: bn
        # relu
        # conv2_3x3(32,32)
        #if bn: bn
        # relu
        
# block2:
        #Bottleneck, 24, 3
        # 1 block(in_planes, planes,stride =2, downsample)
        #
        #then (3 - 1) normal blocks:
        # block(in_planes, planes, stride =1 
        
# block3-5: see block 2
        
        
#where Bottleneck:
        # res = x
        # conv_1x1
        # (bn)
        # relu
        # conv_3x3 (stride = 2), bn, relu
        # conv_1x1(in, expansion*in), bn
        # out = out + res  (if stride = 2, then downsample res)
        # downsample is done using a 2d convolution, 
        #with kernel size 1, proper in/out channels, and optionally batch norm
        
        #so every bottleneck has n_channels
        # and has a series n_channels, n_channels, n_channels*4
        # the first bottleneck has a downsample

        # so in terms of tensor shapes, after block1, youd have:
        #
        # [x, y, 32]
        #
        # into block2:
        # [x  , y  , 24] (conv1x1) #bottleneck 1
        # [x/2, y/2, 24] (conv3x3)
        # [x/2, y/2, 96] (conv1x1)
        # [x/2, y/2, 24] (conv1x1) #bottleneck2
        # [x/2, y/2, 24] (conv3x3)
        # [x/2, y/2, 96] (conv1x1)
        # [x/2, y/2, 24] (conv1x1)#bottleneck3
        # [x/2, y/2, 24] (conv3x3)
        # [x/2, y/2, 96] (conv1x1)
        
        #into block3:
        # [x/2, x/2, 48] (conv1x1) #bottleneck1
        # [x/4, x/4, 48] (conv3x3) 
        # etc
        
        
        
        
#Then you have 
    
    
    
    

    
