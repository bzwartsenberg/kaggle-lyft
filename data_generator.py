#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:18:10 2019

@author: berend
"""

#data generator
import pdb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from util_funcs import decode_predictionstring, get_points_in_a_rotated_box
import os

import cv2


# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

from scipy.sparse import csr_matrix

from keras.utils import Sequence



class data_generator():
    
    def __init__(self, train, lyftdata, config = {}):
        """
        Args:
            train: train dataframe
            lyftdata: the data sdk
        """
        
        self.train = train
        self.lyftdata = lyftdata
        
        #params for input maps:
        self.xlim = config['xlim'] if 'xlim' in config else (-102.4,102.4)
        self.ylim = config['ylim'] if 'ylim' in config else  (-51.2,51.2)
        self.delta = config['delta'] if 'delta' in config else  0.2
        self.zlim = config['zlim'] if 'zlim' in config else  (0,3)
        self.delta_z = config['delta_z'] if 'delta_z' in config else  0.5
        
        self.normalize = config['normalize'] if 'normalize' in config else True
        
        #no intensity, so no extra channels
        #may want to have an extra channel for the roadmap that lyft provides
        self.shape = tuple(map(int, [(self.xlim[1]-self.xlim[0])/self.delta, (self.ylim[1]-self.ylim[0])/self.delta, (self.zlim[1]-self.zlim[0])/self.delta_z])) 
        
        #limits for pixels (size shape + 1)
        self.xaxis_lim = np.linspace(*self.xlim, self.shape[0]+1)
        self.yaxis_lim = np.linspace(*self.ylim, self.shape[1]+1)
        self.zaxis_lim = np.linspace(*self.zlim, self.shape[2]+1)   
        
        self.xaxis = (self.xaxis_lim[1:] + self.xaxis_lim[:-1])/2
        self.yaxis = (self.yaxis_lim[1:] + self.yaxis_lim[:-1])/2
        self.zaxis = (self.zaxis_lim[1:] + self.zaxis_lim[:-1])/2
                
        
        self.categories = ['car',
                     'pedestrian',
                     'animal',
                     'other_vehicle',
                     'bus',
                     'motorcycle',
                     'truck',
                     'emergency_vehicle',
                     'bicycle']
    
        #classes to use:
        self.inc_classes = config['inc_classes'] if 'inc_classes' in config else self.categories
        self.n_classes = len(self.inc_classes)
    
        #classmap:
        self.cat_to_num = {cat : i for i,cat in enumerate(self.inc_classes)}
        #inverse:
        self.num_to_cat = {v : k for k,v in self.cat_to_num.items()}
        
        #params for output maps:
        self.output_scale = config['output_scale'] if 'output_scale' in config else 4
        self.o_delta = self.delta*self.output_scale
        
        #output features: x,y,z,dx,dy,dz,cos(th),sin(th),n_classes     
        self.o_shape = tuple(map(int, [(self.xlim[1]-self.xlim[0])/self.o_delta, (self.ylim[1]-self.ylim[0])/self.o_delta, 8 + self.n_classes])) 
        
        
        self.o_xaxis_lim = np.linspace(*self.xlim, self.o_shape[0]+1)
        self.o_yaxis_lim = np.linspace(*self.ylim, self.o_shape[1]+1)
        self.o_zaxis_lim = np.linspace(*self.zlim, self.o_shape[2]+1)   
        
        self.o_xaxis = (self.o_xaxis_lim[1:] + self.o_xaxis_lim[:-1])/2
        self.o_yaxis = (self.o_yaxis_lim[1:] + self.o_yaxis_lim[:-1])/2
        self.o_zaxis = (self.o_zaxis_lim[1:] + self.o_zaxis_lim[:-1])/2
        
    def scale_to_bin(self, pts, as_int = True):
        
        pts_bin = (pts - np.array([[self.xlim[0],self.ylim[0]]]))/np.array([[self.delta,self.delta]])
        
        if as_int:
            pts_bin = pts_bin.astype('int')
            
        return pts_bin

    def o_scale_to_bin(self, pts, as_int = True):
        
        pts_bin = (pts - np.array([[self.xlim[0],self.ylim[0]]]))/np.array([[self.o_delta,self.o_delta]])
        
        if as_int:
            pts_bin = pts_bin.astype('int')
            
        return pts_bin
        
    def bin_to_scale(self, bins):
        
        return bins*np.array([[self.delta,self.delta]]) + np.array([[self.xlim[0],self.ylim[0]]])
        

    def o_bin_to_scale(self, bins):
        
        return bins*np.array([[self.o_delta,self.o_delta]]) + np.array([[self.xlim[0],self.ylim[0]]])
        
    

    def get_lidar_BEV(self,sample_token):
        """Get a lidar feature map
        Args: sample token, token from train to put into lyftdataset
        Returns: featuremap of self.shape"""
        
        #make feature map
        feature_map = np.zeros(self.shape)
        
        sample = self.lyftdata.get('sample', sample_token)
    
        #get lidar points:
        for sensor_name in sample['data'].keys():
            if 'LIDAR' in sensor_name:
                lidar = self.lyftdata.get('sample_data', sample['data'][sensor_name])
                
                try:
                    
                    pc = LidarPointCloud.from_file(Path(lidar['filename']))
                    #transform to vehicle frame:
                    cs_record = self.lyftdata.get("calibrated_sensor", lidar["calibrated_sensor_token"])
                    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
                    pc.translate(np.array(cs_record["translation"]))
    
                    feature_map += np.histogramdd(pc.points[0:3].T, [self.xaxis_lim, self.yaxis_lim, self.zaxis_lim])[0]
                except ValueError as e:
                    print('Ignored ', e)
                    
        if self.normalize:
            feature_map = np.log(1 + feature_map)/3
        #bin points:
        return feature_map
    
    
    #adapted from https://github.com/philip-huang/PIXOR/blob/master/srcs/datagen.py
    def get_corners(self, df_row):
    
        x,y,dx,dy,yaw = df_row[['x','y','dx','dy','yaw_cor']]
    
        
        corners = np.array([[- dx/2, dy/2],
                           [- dx/2,  -dy/2],
                           [ dx/2,  -dy/2],
                           [dx/2,   dy/2]], dtype=np.float32)
        
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]],dtype = 'float32')
        
        corners = np.dot(rot, corners.T).T + np.array([[x,y]])
        
        return corners
    
    
    def update_label_map(self, df_row, out_part_1, out_part_2, class_map):
        #array_x and array_y should be array.shape[0]+1 and array.shape[1]+1 to be used with digitize
        
        corners = self.get_corners(df_row)
    
        
        corner_bins = self.o_scale_to_bin(corners, as_int = True)
        
        corner_bins = corner_bins[:,[1,0]]
                
        reg_target = np.array(df_row[['x','y','z','logdx','logdy','logdz','cos','sin']]).astype('float') #reg target

        cv2.drawContours(out_part_1, contours=[corner_bins], contourIdx=-1, color=reg_target[0:4], thickness=-1)
        cv2.drawContours(out_part_2, contours=[corner_bins], contourIdx=-1, color=reg_target[4:8], thickness=-1)
        cv2.drawContours(class_map, contours=[corner_bins], contourIdx=-1, color=1., thickness=-1)

    
    
    def get_output_map(self,sample_token, pred_str):

        
        pos, obj = decode_predictionstring(pred_str)
    
    
        sample = self.lyftdata.get('sample', sample_token)
        lidar_top = self.lyftdata.get('sample_data', sample['data']['LIDAR_TOP']) 
        ego_pose = self.lyftdata.get('ego_pose',lidar_top['ego_pose_token'])
    
        xyz = pos[:,0:3]-np.array(ego_pose["translation"]).reshape((1,-1))
    
        qt = Quaternion(ego_pose["rotation"]).inverse
    
        xyz = np.dot(qt.rotation_matrix,xyz.T).T
    
        pos[:,0:3] = xyz
    
        df = pd.DataFrame(pos)
        df.columns = ['x','y','z','dy','dx','dz','yaw']
        obj_df = pd.Categorical(obj, categories = self.categories)
        df['cat'] = obj_df
        df['cat_num'] = obj_df.codes
        df['yaw_cor'] = df['yaw'] + np.arctan2(qt.rotation_matrix[1,0],qt.rotation_matrix[0,0])
    
        df['logdx'] = np.log(df['dx'])
        df['logdy'] = np.log(df['dy'])
        df['logdz'] = np.log(df['dz'])
        df['sin'] = np.sin(df['yaw_cor'])
        df['cos'] = np.cos(df['yaw_cor'])
        df['vol'] = df['dx']*df['dy']*df['dz']
        df_a = df[df['cat'].isin(self.inc_classes)].sort_values('vol', ascending = False)
    
    
        output_map = np.zeros(self.o_shape)
        
        #cv2 can only "draw" up to 4 channels at once
        #out_part_1 is x,y,z,logdx
        #out_part_2 is logdy,logdz,cos,sin
        #then has_object, and num_cat other maps
        out_part_1 = np.zeros(self.o_shape[0:2] + (4,))
        
        #precorrect for x/y output correction
        out_part_1[:,:,0] += self.o_xaxis.reshape((-1,1))
        out_part_1[:,:,1] += self.o_yaxis.reshape((1,-1))
        
        out_part_2 = np.zeros(self.o_shape[0:2] + (4,))
        class_maps = {cat : np.zeros(self.o_shape[0:2]) for cat in self.inc_classes}
        
    
        for i in range(df_a.shape[0]):
            
    
            self.update_label_map(df_a.iloc[i], out_part_1, out_part_2, class_maps[df_a['cat'][i]])
        
        output_map[:,:,0:4] = out_part_1
        output_map[:,:,4:8] = out_part_2
        for cat in self.inc_classes:
            output_map[:,:,8 + self.cat_to_num[cat]] = class_maps[cat]
        
        #correct x and y:
        output_map[:,:,0] -= self.o_xaxis.reshape((-1,1))
        output_map[:,:,1] -= self.o_yaxis.reshape((1,-1))
        
    
        return output_map
    

    
class train_generator(Sequence):
    'Generates data for train'
    
    def __init__(self, use_idx, generator, batch_size=4, shuffle=True, seed = None):
        'Initialization'
        self.use_idx = use_idx
        self.gen = generator
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # set seed
        np.random.seed(seed)
        

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.use_idx) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        idxs = self.shuffled_idx[index*self.batch_size:(index+1)*self.batch_size]


        # Generate data
        X, y = self.__data_generation(idxs)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.shuffled_idx = np.random.permutation(self.use_idx)
        else:
            self.shuffled_idx = self.use_idx

    def __data_generation(self, idxs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        
        X = np.empty((self.batch_size, *self.gen.shape))
        y = np.empty((self.batch_size, *self.gen.o_shape))

        # Generate data
        for i, idx in enumerate(idxs):
            # Store sample
            X[i] = self.gen.get_lidar_BEV(self.gen.train['Id'][idx])

            # Store class
            y[i] = self.gen.get_output_map(self.gen.train['Id'][idx], self.gen.train['PredictionString'][idx])

        return X, y    

class evaluation_generator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, use_idx, generator, batch_size=4):
        'Initialization'
        self.use_idx = use_idx
        self.gen = generator
        self.batch_size = batch_size
        
        

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.use_idx) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        #last batch is shorter in length
        idxs = self.use_idx[index*self.batch_size:min((index+1)*self.batch_size,self.use_idx.shape[0])]


        # Generate data
        X, y = self.__data_generation(idxs)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass
    
    
    def __data_generation(self, idxs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        
        X = np.empty((len(idxs), *self.gen.shape))
        y = np.empty((len(idxs), *self.gen.o_shape))

        # Generate data
        for i, idx in enumerate(idxs):
            # Store sample
            X[i] = self.gen.get_lidar_BEV(self.gen.train['Id'][idx])

            # Store class
            y[i] = np.zeros(self.gen.o_shape)
        return X, y    
    
