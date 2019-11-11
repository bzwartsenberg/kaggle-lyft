#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:39:17 2019

@author: berend
"""

from network import get_model, make_network, get_loss
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import Callback,LearningRateScheduler

from data_generator import train_generator,data_generator,evaluation_generator

import numpy as np
import json
import os

def run_train(lyftdata, train, config):
    
        
    use_bn = config['use_bn'] if 'use_bn' in config else False
    bn_level = config['bn_level'] if 'bn_level' in config else 0
    load_path = config['load_path'] if 'load_path' in config else ''
    save_path = config['save_path'] if 'save_path' in config else ''
    sub_path = config['sub_path'] if 'sub_path' in config else ''
    save_name = config['save_name'] if 'save_name' in config else 'save'
    lr = config['lr'] if 'lr' in config else 1e-4
    lr_decay_per_epoch = config['lr_decay_per_epoch'] if 'lr_decay_per_epoch' in config else 0.2
    epochs = config['epochs'] if 'epochs' in config else 3
    cls_weight = config['cls_weight'] if 'cls_weight' in config else 10.
    reg_weight = config['reg_weight'] if 'reg_weight' in config else 50.

    train_split = config['train_split'] if 'train_split' in config else 0.9
    seed = config['seed'] if 'seed' in config else 0
    workers = config['workers'] if 'workers' in config else 1
    use_multiprocessing = config['use_multiprocessing'] if 'use_multiprocessing' in config else False
    
    if not save_path[-1] == '/':
        save_path += '/'
    
    if sub_path == '':
        i = 0
        while os.path.exists(save_path + '{:05d}/'.format(i)):
            i += 1
        sub_path = '{:05d}/'.format(i)
        os.mkdir(save_path + sub_path)
        
    save_path += (sub_path + save_name)
    
    
    histories = Histories()
    if not 'callbacks' in config:
        callbacks = [histories]
    else:
        callbacks = config['callbacks'] + [histories]
    
    gen = data_generator(train, lyftdata, config=config)
    np.random.seed(seed)
    perm = np.random.permutation(gen.train.shape[0])
    train_idx = perm[0:int(train_split*gen.train.shape[0])]
    val_idx = perm[int(train_split*gen.train.shape[0]):]
    
    train_gen = train_generator(train_idx, gen, batch_size=4, shuffle=True, seed = None)
    val_gen = evaluation_generator(val_idx, gen, batch_size=4)    
    
    if load_path != '':
        model = load_model(load_path)
        loss, cls_loss, reg_loss =  get_loss(len(gen.inc_classes), cls_weight, reg_weight)
        model.compile(
                      optimizer=Adam(lr),
                      loss=loss,
                    metrics = [cls_loss, reg_loss])
    else:
        model = get_model(gen.shape,len(gen.inc_classes), use_bn = use_bn, bn_level = bn_level, 
                          expand_channels = 4, cls_weight = cls_weight, reg_weight = reg_weight, optimizer = Adam(lr))
    
    def scheduler(epoch, lr):
        if epoch == 0:
            return lr
        else:
            return lr*lr_decay_per_epoch
    
    lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
    callbacks.append(lr_scheduler)
    callbacks.append(SaveCheckPoints(frequency = 1000, path = save_path))
        
    hist = model.fit_generator(train_gen, epochs = epochs, use_multiprocessing = use_multiprocessing, 
                        workers = workers, 
                        callbacks=callbacks,
                        validation_data = val_gen)
    
    with open(save_path + '_config.json','w') as f:
        json.dump(config,f)
    
    return model, histories,hist
    
    

class SaveCheckPoints(Callback):
    
    def __init__(self, frequency = 1000, path = ''):
        
        #keep track of number of batches
        self.i = 0
        self.j = 0
        
        self.path = path
        self.frequency = frequency
        
    def on_epoch_end(self, epoch, logs={}):
        self.i = 0
        self.j += 1
        self.model.save(self.path + '_{}_final.h5'.format(self.j), include_optimizer = False)
        
    def on_batch_end(self, batch, logs={}):
        if (self.i != 0) and (self.i % self.frequency == 0):
            self.model.save(self.path + '_{}_{}.h5'.format(self.j,self.i), include_optimizer = False)
        self.i += 1
        
        


class Histories(Callback):
    def __init__(self, save_path = ''):
        self.save_path = save_path
        
    def on_train_begin(self,logs={}):
        self.losses = []
        self.cls_losses = []
        self.reg_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.reg_losses.append(logs.get('reg_loss_func'))
        self.cls_losses.append(logs.get('cls_loss_func'))
        
    def on_epoch_end(self, epoch, logs={}):
        if self.save_path != '':
            np.save_txt(np.array(self.losses),self.save_path + 'losses.txt')
            np.save_txt(np.array(self.reg_losses),self.save_path + 'reg_losses.txt')
            np.save_txt(np.array(self.cls_losses),self.save_path + 'cls_losses.txt')
            
        

    