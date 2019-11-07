#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:38:31 2019

@author: berend
"""


from keras.layers import Conv2D,Conv2DTranspose, BatchNormalization, Input, add, ReLU, Concatenate

from keras.models import Model

from keras.losses import logcosh, categorical_crossentropy, binary_crossentropy
import keras.backend as K
    

    
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
    
    activation = 'sigmoid'
    
    cls_pred = Conv2D(n_classes, 
               kernel_size = (3,3), 
               strides=(1,1), padding = 'same', 
               kernel_initializer = kernel_initializer,
               activation = activation,
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

    return reg, cls_pred


def get_loss(n_classes, cls_weight, reg_weight):
    
    crossentropy = binary_crossentropy
    
    size_reg = [-1,-1,-1,8]
    start_reg = [0,0,0,0]


    size_cls = [-1,-1,-1,n_classes]
    start_cls = [0,0,0,8]
    
    def cls_loss_func(y_true, y_pred):

        cls_true = K.slice(y_true, start_cls, size_cls)
        cls_pred = K.slice(y_pred, start_cls, size_cls)
        
        cls_loss = crossentropy(cls_true, cls_pred)
        
        return cls_loss*cls_weight 
    
        
    def reg_loss_func(y_true, y_pred):
        
        reg_true = K.slice(y_true, start_reg, size_reg)
        reg_pred = K.slice(y_pred, start_reg, size_reg)

        cls_true = K.slice(y_true, start_cls, size_cls)
                
        reg_mask = K.sum(cls_true, axis = -1, keepdims = True)
        
        reg_loss = logcosh(reg_true, reg_mask*reg_pred)    
        
        return reg_loss*reg_weight
    
    
    def loss_func(y_true, y_pred):
        
        #y[:,:,0:8] is reg
        #y[:,:,8:]  is classes

        reg_true = K.slice(y_true, start_reg, size_reg)
        reg_pred = K.slice(y_pred, start_reg, size_reg)


        cls_true = K.slice(y_true, start_cls, size_cls)
        cls_pred = K.slice(y_pred, start_cls, size_cls)
        
        
        cls_loss = crossentropy(cls_true, cls_pred)
        
        
        
#        reg_mask = obj_true
        reg_mask = K.sum(cls_true, axis = -1, keepdims = True)
        
        reg_loss = logcosh(reg_true, reg_mask*reg_pred)
        
        return reg_loss*reg_weight + cls_weight*cls_loss
    
    return loss_func, cls_loss_func, reg_loss_func




def make_network(input_shape,n_classes, use_bn = False, expand_channels = 4):
    
    inp = Input(shape = input_shape)

    p4 = make_backbone(inp,
                       [3, 6, 6, 3],
                       [24, 48, 64, 96],
                       use_bn = use_bn,
                       expand_channels = expand_channels
                       )
    
    reg, cls_pred = make_header(p4, n_classes, use_bn = use_bn)
    
    out = Concatenate(axis=-1)([reg,cls_pred])
    
    return inp, out
    
    
    
def get_model(input_shape,n_classes, use_bn = False, expand_channels = 4, 
              cls_weight = 10., reg_weight = 5., optimizer = 'adam'):
    
    
    inp, out = make_network(input_shape,n_classes, use_bn = use_bn, expand_channels = expand_channels)
    
    model = Model(inp,out)
    
    #note: need to update this to a more appropriate lossfunction:
    # basically the regression loss is calculated where the category is non-zero, so only where an object is
    
    loss, cls_loss, reg_loss =  get_loss(n_classes, cls_weight, reg_weight)
    
    model.compile(
                  optimizer='adam',
                  loss=loss,
                metrics = [cls_loss, reg_loss])
    
    return model



if __name__ == '__main__':
    model = get_model((512,512,6), 8, use_bn = False, expand_channels = 4, cls_weight = 1.)
    model.summary()
#    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    
    import numpy as np
    
    X = np.random.uniform(size = (12,512,512,6))
    y = np.random.uniform(size = (12,128,128,16))
    
    
    model.fit(X,y, batch_size = 2, epochs = 2)

    
