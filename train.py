#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:39:17 2019

@author: berend
"""

from network import get_model, make_network, get_loss
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import Callback

from data_generator import train_generator,data_generator

import numpy as np

def train(lyftdata, train, config):
    
        
    use_bn = config['use_bn'] if 'use_bn' in config else False
    bn_level = config['bn_level'] if 'bn_level' in config else 0
    load_path = config['load_path'] if 'load_path' in config else ''
    save_path = config['save_path'] if 'save_path' in config else ''
    lr = config['lr'] if 'lr' in config else 1e-4
    lr_decay_per_epoch = config['lr_decay_per_epoch'] if 'lr_decay_per_epoch' in config else 0.2
    epochs = config['epochs'] if 'epochs' in config else 3
    cls_weight = config['cls_weight'] if 'cls_weight' in config else 10.
    reg_weight = config['reg_weight'] if 'reg_weight' in config else 50.

    train_split = config['train_split'] if 'train_split' in config else 0.9
    seed = config['seed'] if 'seed' in config else 0
    workers = config['workers'] if 'workers' in config else 1
    
    if not 'callbacks' in config:
        histories = Histories()
        callbacks = [histories]
    else:
        callbacks = config['callbacks']
    
    gen = data_generator(train, lyftdata)
    np.random.seed(seed)
    perm = np.random.permutation(gen.train.shape[0])
    train_idx = perm[0:int(train_split*gen.train.shape[0])]
    #val_idx = perm[int(train_split*gen.train.shape[0]):]
    
    train_gen = train_generator(train_idx, gen, batch_size=4, shuffle=True, seed = None)
    #val_gen = train_generator(val_idx, gen, batch_size=4, shuffle=True, seed = None)    
    
    if load_path != '':
        model = load_model(load_path)
        loss, cls_loss, reg_loss =  get_loss(len(gen.inc_classes), cls_weight, reg_weight)
        model.compile(
                      optimizer=Adam(lr),
                      loss=loss,
                    metrics = [cls_loss, reg_loss])
    else:
        model = get_model(gen.shape,len(gen.inc_classes), use_bn = use_bn, bn_level = bn_level, 
                          expand_channels = 4, cls_weight = 10., reg_weight = 5., optimizer = Adam(lr))
    
    for i in range(epochs):
        model.fit_generator(train_gen, epochs = 1, use_multiprocessing = False, workers = workers, callbacks=callbacks)
        lr = lr*lr_decay_per_epoch
        
        
        if save_path != '':
            model.save(save_path + '_epoch_{}'.format(i))
    
    

class Histories(Callback):

    def on_train_begin(self,logs={}):
        self.losses = []
        self.cls_losses = []
        self.reg_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.reg_losses.append(logs.get('reg_loss_func'))
        self.cls_losses.append(logs.get('cls_loss_func'))    
    
    
    