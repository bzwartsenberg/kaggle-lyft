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


# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix


class data_generator():
    
    def __init__(self, train, lyftdata):
        """
        Args:
            train: train dataframe
            lyftdata: the data sdk
        """
        
        self.train = train
        self.lyftdata = lyftdata
        
        #params for input maps:
        self.xlim = (-100,100)
        self.ylim = (-50,50)
        self.delta = 0.5
        self.zlim = (0,3)
        self.delta_z = 0.5
        
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
        self.inc_classes = ['car']
        self.n_classes = len(self.inc_classes)
    
        #classmap:
        self.cat_to_num = {cat : i for i,cat in enumerate(self.inc_classes)}
        #inverse:
        self.num_to_cat = {v : k for k,v in self.cat_to_num.items()}
        
        #params for output maps:
        self.output_scale = 4
        self.o_delta = self.delta/self.output_scale
        
        #output features: x,y,z,dx,dy,dz,cos(th),sin(th),n_classes        
        self.o_shape = tuple(map(int, [(self.xlim[1]-self.xlim[0])/self.o_delta, (self.ylim[1]-self.ylim[0])/self.o_delta, 8 + self.n_classes])) 
        
        
        self.o_xaxis_lim = np.linspace(*self.xlim, self.o_shape[0]+1)
        self.o_yaxis_lim = np.linspace(*self.ylim, self.o_shape[1]+1)
        self.o_zaxis_lim = np.linspace(*self.zlim, self.o_shape[2]+1)   
        
        self.o_xaxis = (self.o_xaxis_lim[1:] + self.o_xaxis_lim[:-1])/2
        self.o_yaxis = (self.o_yaxis_lim[1:] + self.o_yaxis_lim[:-1])/2
        self.o_zaxis = (self.o_zaxis_lim[1:] + self.o_zaxis_lim[:-1])/2
        
        

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
                pc = LidarPointCloud.from_file(Path(lidar['filename']))
    
                #transform to vehicle frame:
                cs_record = self.lyftdata.get("calibrated_sensor", lidar["calibrated_sensor_token"])
                pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
                pc.translate(np.array(cs_record["translation"]))
    
                x_bins = np.digitize(pc.points[0], self.xaxis_lim)
                y_bins = np.digitize(pc.points[1], self.yaxis_lim)
                z_bins = np.digitize(pc.points[2], self.zaxis_lim)    
    
                usepts = np.argwhere((x_bins > 0) &
                            (x_bins <= self.shape[0]) &
                            (y_bins > 0) &
                            (y_bins <= self.shape[1]) &
                            (z_bins > 0) &
                            (z_bins <= self.shape[2]))    
    
                #ugly for loop:
                for pt in usepts:
                    feature_map[x_bins[pt]-1,y_bins[pt]-1,z_bins[pt]-1] += 1.
        
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
    
    
    def update_label_map(self, df_row, array):
        #array_x and array_y should be array.shape[0]+1 and array.shape[1]+1 to be used with digitize
        
        corners = self.get_corners(df_row)
    
        print('Got corners', corners)
        
        corner_bins = np.zeros_like(corners).astype('int')
        
        #edge case: what if half falls outside the limit? --> it will cut off to n+1 or 0
        corner_bins[:,0] = np.digitize(corners[:,0], self.o_xaxis_lim)
        corner_bins[:,1] = np.digitize(corners[:,1], self.o_yaxis_lim)
        print('with bins', corner_bins)
    
                
        points = get_points_in_a_rotated_box(corner_bins, array.shape[0:2])
        
        for p in points:
            metric_x, metric_y = self.o_xaxis[p[0]],self.o_yaxis[p[1]]
            
            reg_target = np.array(df_row[['x','y','z','logdx','logdy','logdz','cos','sin']]) #x,y,z,dx,dy,dz,cos(th),sin(th),n_classes
            reg_target[0] -= metric_x
            reg_target[1] -= metric_y
            
            class_target = np.zeros(self.n_classes)
            class_target[df_row['cat_num']] = 1.
            array[p[0],p[1],0:8] = reg_target
            array[p[0],p[1],8:] = class_target
    
    
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
        df['yaw_cor'] = df['yaw'] - qt.angle
    
        df['logdx'] = np.log(df['dx'])
        df['logdy'] = np.log(df['dy'])
        df['logdz'] = np.log(df['dz'])
        df['sin'] = np.sin(df['yaw_cor'])
        df['cos'] = np.cos(df['yaw_cor'])
        df['vol'] = df['dx']*df['dy']*df['dz']
        df_a = df[df['cat'].isin(self.inc_classes)].sort_values('vol', ascending = False)
    
    
        output_map = np.zeros(self.o_shape)
    
        for i in range(df_a.shape[0]):
            
            print('Now working on', df_a.iloc[i][['x','y']])
    
            self.update_label_map(df_a.iloc[i], output_map)
    
        return output_map
    
    
    