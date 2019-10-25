#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:19:16 2019

@author: berend
"""


import numpy as np
from shapely.geometry import Polygon,mapping
from util_funcs import decode_predictionstring,get_points_in_a_rotated_box
import pandas as pd
import matplotlib.pyplot as plt

class RotatedBox():
    
    def __init__(self,x,y,z,dx,dy,dz,th, score, objstr):
        
        corners = np.array([[- dx/2, dy/2],
                           [- dx/2,  -dy/2],
                           [ dx/2,  -dy/2],
                           [dx/2,   dy/2]], dtype=np.float32)
        
        rot = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th), np.cos(th)]],dtype = 'float32')
        
        corners = np.dot(rot, corners.T).T + np.array([[x,y]])

        
        self.base_poly = Polygon(corners)
        
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.th = th
        
        self.z = z
        self.dz = dz
        
        self.z_lo = z - dz/2
        self.z_hi = z + dz/2
        
        self.score = score
        self.objstr = objstr
        
        
    def z_intersect(self, box):
        
        z_start_boxes = max(self.z_lo, box.z_lo)
        z_end_boxes = min(self.z_hi, box.z_hi)
        
        overlap = max(0, z_end_boxes - z_start_boxes)
        
        
        return overlap

    def z_union(self, box):
        
        return self.dz + box.dz - self.z_intersect(box)
        

        
    def intersect(self, box):
        
        xy_area_overlap = self.base_poly.intersection(box.base_poly).area
        
        return self.z_intersect(box)*xy_area_overlap

    def union(self, box):
        
        xy_area_overlap = self.base_poly.union(box.base_poly).area
        
        return self.z_union(box)*xy_area_overlap
                
    
    
    def iou(self, box):
        
        return self.intersect(box)/self.union(box)
    
    def get_pred_str(self):
        
        pred_str = '{} {} {} {} {} {} {} {}'.format(self.score, self.x, self.y, self.z, self.dx, self.dy, self.dz, self.objstr)
        
        
        return pred_str


def convert_to_box_objs(pred_array):
    """
    Args:
        pred_array: [N, (x,y,z,logdx,logdy,logdz,costh,sinth)]
    returns: 
        array of RotatedBox classes
    """
    
    dxdydz = np.exp(pred_array[:,3:6])
    th = np.arctan(pred_array[:,7]/pred_array[:,6])  #th = arctan(sin/cos)


    rot_boxes = [RotatedBox(*pred_array[i,0:3], *dxdydz[i], th[i], pred_array[i,8], 'car') for i in range(pred_array.shape[0])]
    
    return np.array(rot_boxes)


def compute_ious(rot_box, with_rot_boxes):

    ious = np.zeros(with_rot_boxes.shape[0])
    for i in range(with_rot_boxes.shape[0]):
        ious[i] = rot_box.iou(with_rot_boxes[i])
    return ious





def non_max_suppression(pred_box_array, nms_iou_threshold):

    assert pred_box_array.shape[0] > 0
    

    box_obj_array = convert_to_box_objs(pred_box_array)
    
    scores = pred_box_array[:, -1]

    # Get indicies of boxes sorted by scores (highest first)
    
    max_pred = min(scores.shape[0], 1e10)
    
    ixs = scores.argsort()[::-1][:max_pred]
    

    picks = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        picks.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_ious(box_obj_array[i], box_obj_array[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > nms_iou_threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(picks, dtype=np.int32),box_obj_array

def filter_pred(pred_image, x_scale, y_scale, cls_threshold, nms_iou_threshold):
    #NOTE THIS ONLY WORKS FOR SINGLE CLASS PREDICTION!!!

    if len(pred_image.shape) == 4:
        pred_image = np.squeeze(pred_image, 0)
        
    pred_image[:,:,0] += x_scale.reshape((-1,1))
    pred_image[:,:,1] += y_scale.reshape((1,-1))
    
    cls_pred = pred_image[:,:,-1]
    
    ##FOR THIS ALGORITHM TO WORK CAN"T USE SOFTMAX
    # HAVE TO USE SIGMOID. OTHERWISE THERE WOULD ALWAYS BE A PRED > cls_threshold
    
    
    idx = np.argwhere(cls_pred > cls_threshold)    
    
    if idx.shape[0] == 0:
        return np.array([]),np.array([])


    #a (n_boxes,9) size array, (flattened and filtered by > cls_threshold of pred image)
    pred_box_array = pred_image[idx[:,0], idx[:,1]]
    
        
    #return numpy array of RotatedBox objects as well, so they can use a method to convert to string
    selected_box_idxs, box_obj_array = non_max_suppression(pred_box_array, nms_iou_threshold)
    
    picked_box_objs = box_obj_array[selected_box_idxs]
    picked_scores = pred_box_array[selected_box_idxs, -1]
    
    return picked_box_objs, picked_scores



def make_prediction_string(box_objs):
    pred_str = ''
    for box_obj in box_objs:
        pred_str += box_obj.get_pred_str()
        pred_str += ' '
        
        
    return pred_str[:-1]
    

class geometry():

    def __init__(self):      

        #params for input maps:
        self.xlim = (-102.4,102.4)
        self.ylim = (-51.2,51.2)
        self.delta = 0.2
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
    
        
        corner_bins = self.o_scale_to_bin(corners, as_int = True)
        
        try:
            points = get_points_in_a_rotated_box(corner_bins)
        except ValueError:
            points = np.array([[-1,-1]])
            
        usepts = np.argwhere((points[:,0] >= 0) &
                    (points[:,0] < self.o_shape[0]) &
                    (points[:,1] >= 0) &
                    (points[:,1] < self.o_shape[1])).flatten()

        for p in points[usepts]:
            array[p[0],p[1]] = 1.

    
    
    def reconstruct_output_map(self, pred_str):

        
        pos, obj = decode_predictionstring(pred_str)
    
    
        df = pd.DataFrame(pos)
        df.columns = ['x','y','z','dy','dx','dz','yaw_cor']
    
    
        output_map = np.zeros(self.o_shape[0:2])
    
        for i in range(df.shape[0]):
            
    
            self.update_label_map(df.iloc[i], output_map)
#        print('all else ', times[1] - times[0])
#        print('For loop ', times[2] - times[1])
        
    
        return output_map    
        
    
def plot_Rotated_box(ax, rotated_box):
    
    coords = np.array(mapping(rotated_box.base_poly)['coordinates'])[0]
    
    ax.plot(coords[:,0],coords[:,1])
    
    
def plot_boxes(box_array, ax = None):
    
    if ax is None:
        fig,ax = plt.subplots(figsize = (5,5))
    
    for box in box_array:
        plot_Rotated_box(ax, box)
    

if __name__ == '__main__':
        
    
    test_y_pred = np.load('/Users/berend/Google Drive/colab_files/test_y_pred.npy')
    test_y_true = np.load('/Users/berend/Google Drive/colab_files/test_y_true.npy')
    
    pred_image = test_y_pred[3]
    pred_image2 = test_y_true[3]
    
    cls_pred = pred_image[:,:,-1]

    cls_threshold = 0.7
    idx = np.argwhere(cls_pred > cls_threshold)    
    pred_box_array = pred_image[idx[:,0], idx[:,1]]
    
    box_obj_array = convert_to_box_objs(pred_box_array)
    
    nms_iou_threshold = 0.2
    
    g = geometry()    
    #limits for pixels (size shape + 1)
    
    
    picked_box_objs, picked_scores = filter_pred(pred_image, g.o_xaxis, g.o_yaxis, cls_threshold, nms_iou_threshold)
    picked_box_objs2, picked_scores2 = filter_pred(pred_image2, g.o_xaxis, g.o_yaxis, cls_threshold, nms_iou_threshold)
    

    pstr = make_prediction_string(picked_box_objs)
    
    out_map = g.reconstruct_output_map(pstr)
    

    fig,ax = plt.subplots(figsize = (6,3))
    ax.pcolormesh(g.o_xaxis_lim, g.o_yaxis_lim, pred_image2[:,:,-1].T)
    plot_boxes(picked_box_objs2, ax = ax)
    
    fig,ax = plt.subplots(figsize = (6,3))
    ax.pcolormesh(g.o_xaxis_lim, g.o_yaxis_lim, pred_image[:,:,-1].T)
    plot_boxes(picked_box_objs, ax = ax)
    
    fig,ax = plt.subplots(figsize = (6,3))
    ax.pcolormesh(g.o_xaxis_lim, g.o_yaxis_lim, pred_image2[:,:,-1].T)
    plot_boxes(picked_box_objs, ax = ax)
    
#    
#training: 
#- continue training the model
#- train from checkpoint
#- remove al big code from colab, put in github and manage on gdrive
#
#postprocessing:
#- correct angles and offsets for predictions
#- clean up postprocessing pipeline
#- 
#
#checking:
#- make a plotter that plots prediction string with ground truth string
#- make a calculator that calculates the mAP of ious as integrated in kaggle
#- check optimal iou_threshold and cls_threshold
#
#submission
#- translate prediction into string
#- submit to kaggle
#
#
#optimizing:
#- make outputmap generator faster
#    
#integrate multiple classes:
#- for predicting multiple classes: maybe have an 'object detected' output, that is used for threshold, then softmax for which class
#- predict multple classes

    
    
    
    
    
    
    