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
from data_generator import evaulation_generator

from lyft_dataset_sdk.lyftdataset import Quaternion
from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions

from multiprocessing import Pool


class RotatedBox():
    
    def __init__(self,x,y,z,dx,dy,dz,th, score, obj_num):
        
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
        self.obj_num = obj_num
        
        
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
    
    def get_array(self):
	
        return np.array([self.x, self.y, self.z, self.dx, self.dy, self.dz, self.th])
		
    def get_obj(self):
        return self.obj_num
		
        

class ValidationPostProcessor():

    def __init__(self, model, data_generator, val_idx, 
                 nms_iou_threshold=0.2, 
                 cls_threshold=0.7,
                 chunk_size=100):
        
        self.gen = data_generator
        self.model = model
        self.val_idx = val_idx
        self.chunk_size=100
		
        self.nms_iou_threshold = nms_iou_threshold
        self.cls_threshold = cls_threshold
        
        self.val_predictions = []

		

    def analyze_chunk(self, idx, workers=4):
        
        
        chunk_gen = evaulation_generator(idx, self.gen, batch_size=4)
        
        pred_im = self.model.predict_generator(chunk_gen, multiprocessing=True, workers=workers)
        
        #list of lists
        predictions_fox_idx = []
        
        def mapping_func(i):
            picked_box_objs, picked_scores = filter_pred(pred_im[i], self.gen.o_xaxis, self.gen.o_yaxis, self.cls_threshold, self.nms_iou_threshold)
            predictions = box_objs_to_lyft_sdk_format(picked_box_objs, 
                                                      picked_scores, 
                                                      self.gen.lyftdata, 
                                                      self.gen.train.iloc[idx[i]]['Id'])
            
            return predictions
        
        p = Pool(workers)
        predictions_fox_idx = p.map(mapping_func, range(idx.shape[0]))
        
        return predictions_fox_idx
    
    
    def make_predictions(self):
        
        num_chunks = np.ceil(self.val_idx.shape[0]/self.chunk_size)
        
        self.val_predictions = []
        
        for i in range(num_chunks):
            idx = self.val_idx[i*self.chunk_size:max((i+1)*self.chunk_size, self.val_idx.shape[0])]
            
            self.val_predictions.append(self.analyze_chunk(idx))
    
    
    def calculate_mAP(self, over_iou_thresh=0.1, over_class_names=None):
        
        if self.val_predictions == []:
            print('Run "make_predictions" first')
            
        if over_class_names is None:
            over_class_names = self.gen.inc_classes
            
        
        if type(over_iou_thresh)!= list:
            over_iou_thresh = list(over_iou_thresh)
            
        precs_over_iou = []
        for iou in over_iou_thresh:
            
            precs = []
        
            for i,idx in enumerate(self.val_idx):
                predictions_gt = prediction_string_to_prediction_dicts_gt(self.gen.train.iloc[idx]['PredictionString'], 
                                                                      self.gen.train.iloc[idx]['Id'])
        
                prec = get_average_precisions(predictions_gt, self.val_predictions[i], over_class_names,iou)

                precs.append(np.array(prec))
            precs_over_iou.append(np.array(precs).mean(axis=0))
            
        
        return np.array(over_iou_thresh),np.array(precs_over_iou)




class PostProcessor():

    def __init__(self, model, data_generator, test, config = {}):
        
        self.gen = data_generator
        self.test = test
        self.model = model
		
        self.nms_iou_threshold = config['nms_iou_threshold'] if 'nms_iou_threshold' in config else 0.3
        self.cls_threshold = config['cls_threshold'] if 'cls_threshold' in config else 0.8
		

    def df_to_pred_str(self, df):
        
        cols = ['score','x','y','z','dx','dy','dz','yaw','cat']
        
        pred_str_list = ['{} {} {} {} {} {} {} {} {}'.format(*df.loc[i][cols]) for i in range(df.shape[0])]
        
        return ' '.join(pred_str_list)
    
    def analyze_val_idx(self, idx):
        
        Id, true_predstr = self.gen.train['Id'][idx], self.gen.train['PredictionString'][idx]
        
        df = self.analyze_sample(Id)
        
        predicted_string = Id + ' ' + self.df_to_pred_str(df)
        
        return df, predicted_string, true_predstr
		
		
    def analyze_sample(self, sample_id):
        
        X = np.expand_dims(self.gen.get_lidar_BEV(self,sample_id),0)
        
        y_pred = self.model.predict(X)
        
        picked_box_objs, picked_scores = filter_pred(y_pred[0], self.gen.o_xaxis, self.gen.o_yaxis, self.cls_threshold, self.nms_iou_threshold)
        
        picked_objstrs = np.array([obj.get_obj() for obj in picked_box_objs])
        picked_pos_array = np.array([obj.get_array() for obj in picked_box_objs])
        
        df = pd.DataFrame(picked_pos_array)
        df.columns = ['x','y','z','dy','dx','dz','yaw']
        df['cat'] = picked_objstrs
        df['scores'] = picked_scores


        #translate and rotate:

        sample = self.gen.lyftdata.get('sample', sample_id)
        lidar_top = self.gen.lyftdata.get('sample_data', sample['data']['LIDAR_TOP']) 
        ego_pose = self.gen.lyftdata.get('ego_pose',lidar_top['ego_pose_token'])
        
        qt = Quaternion(ego_pose["rotation"])
        
        df[['x','y','z']] = np.dot(qt.rotation_matrix,df[['x','y','z']].T).T

        df[['x','y','z']] = df[['x','y','z']] + np.array(ego_pose["translation"]).reshape((1,-1))
        
        df['yaw'] = df['yaw'] - qt.angle ##check this!! the data_preprocessing was -, with "inverse", I think you change only one of the two		
        
        return df
	


	

def convert_to_box_objs(pred_array,cls_pred_array,cls_scores_array):
    """
    Args:
        pred_array: [N, (x,y,z,logdx,logdy,logdz,costh,sinth)]
    returns: 
        array of RotatedBox classes
    """
    
    dxdydz = np.exp(pred_array[:,3:6])
    th = np.arctan(pred_array[:,7]/pred_array[:,6])  #th = arctan(sin/cos)


    rot_boxes = [RotatedBox(*pred_array[i,0:3], *dxdydz[i], th[i], cls_scores_array[i], cls_pred_array[i]) for i in range(pred_array.shape[0])]
    
    return np.array(rot_boxes)


def compute_ious(rot_box, with_rot_boxes):

    ious = np.zeros(with_rot_boxes.shape[0])
    for i in range(with_rot_boxes.shape[0]):
        ious[i] = rot_box.iou(with_rot_boxes[i])
    return ious





def non_max_suppression(pred_box_array,cls_pred_array,cls_scores_array, nms_iou_threshold):

    assert pred_box_array.shape[0] > 0
    

    box_obj_array = convert_to_box_objs(pred_box_array,cls_pred_array,cls_scores_array)
    
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
    #I should be able to improve this for batch type inference, at least the first type

    if len(pred_image.shape) == 4:
        pred_image = np.squeeze(pred_image, 0)
        
    pred_image[:,:,0] += x_scale.reshape((-1,1))
    pred_image[:,:,1] += y_scale.reshape((1,-1))
    
    #this predicts the indices
    cls_pred = np.argmax(pred_image[:,:,8:], axis=2)
    cls_scores = pred_image[:,:,8:].max(axis=2)
    
    idx = np.argwhere(cls_scores > cls_threshold)
    
    if idx.shape[0] == 0:#no boxes found
        return np.array([]),np.array([])
        
    else:
        #a (n_boxes,9) size array, (flattened and filtered by > cls_threshold of pred image)
        pred_box_array = pred_image[idx[:,0], idx[:,1],:8]
        cls_pred_array = cls_pred[idx[:,0], idx[:,1]]
        cls_scores_array = cls_scores[idx[:,0],idx[:,1]]
            
        #return numpy array of RotatedBox objects as well, so they can use a method to convert to string
        selected_box_idxs, box_obj_array = non_max_suppression(pred_box_array,cls_pred_array,cls_scores_array, nms_iou_threshold)
        
        picked_box_objs = box_obj_array[selected_box_idxs]
        picked_scores = pred_box_array[selected_box_idxs, -1]
        
        return picked_box_objs, picked_scores


def box_objs_to_lyft_sdk_format(picked_box_objs, picked_scores,lyftdata, Id, num_to_cat):
        
    predictions = []
    
    picked_objstrs = np.array([num_to_cat[obj.get_obj()] for obj in picked_box_objs])
    picked_pos_array = np.array([obj.get_array() for obj in picked_box_objs])
    
    df = pd.DataFrame(picked_pos_array)
    df.columns = ['x','y','z','dy','dx','dz','yaw']
    df['cat'] = picked_objstrs
    df['score'] = picked_scores


    #translate and rotate:

    sample = lyftdata.get('sample', Id)
    lidar_top = lyftdata.get('sample_data', sample['data']['LIDAR_TOP']) 
    ego_pose = lyftdata.get('ego_pose',lidar_top['ego_pose_token'])
    
    qt = Quaternion(ego_pose["rotation"])
    
    df[['x','y','z']] = np.dot(qt.rotation_matrix,df[['x','y','z']].T).T

    df[['x','y','z']] = df[['x','y','z']] + np.array(ego_pose["translation"]).reshape((1,-1))
    
    df['yaw'] = df['yaw'] - qt.angle ##check this!! the data_preprocessing was -, with "inverse", I think you change only one of the two		
    
    
    for i in range(df.shape[0]):
        predictions.append({
                'sample_token' : Id,
                'name' : df['cat'][i],
                'score' : df['score'][i],
                'translation' : df[['x','y','z']][i],
                'size' : df[['dy','dx','dz']][i],
                'rotation' : [df['yaw'][i],0.,0.,1.],
                })
            
    return predictions
    

def decode_predictionstring_gt(pred_str):
    
    if pred_str[-1] == ' ':
        pred_str = pred_str[:-1]
    
    pred_ar = np.array(pred_str.split(' ')).reshape((-1,8))[:]
    obj = pred_ar[:,-1]
    pos = pred_ar[:,:-1]

    pos = pos.astype('float32')
    return pos, obj


def prediction_string_to_prediction_dicts_gt(pred_str, Id):
    
    pos,obj = decode_predictionstring_gt(pred_str)
    
    predictions = []
    
    for i in range(pos.shape[0]):
        predictions.append({
                'sample_token' : Id,
                'name' : obj[i],
                'score' : 1.,
                'translation' : pos[i,0:3],
                'size' : pos[i,3:6],
                'rotation' : [pos[i,6],0.,0.,1.],
                })
            
    return predictions
    
    
#    
#training: 
#x- continue training the model
#- train from checkpoint
#x- remove al big code from colab, put in github and manage on gdrive
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

    
    
    
    
    
    
    