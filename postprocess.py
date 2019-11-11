#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:19:16 2019

@author: berend
"""


import numpy as np
from shapely.geometry import Polygon,mapping
import pandas as pd
from data_generator import evaluation_generator

from lyft_dataset_sdk.lyftdataset import Quaternion
from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions

from multiprocessing import Pool
from functools import partial
import time

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
    
    def iou_2d(self, box):
        return self.base_poly.intersection(box.base_poly).area/self.base_poly.union(box.base_poly).area
    
    def get_array(self):
	
        return np.array([self.x, self.y, self.z, self.dy, self.dx, self.dz, self.th])
		
    def get_obj(self):
        return self.obj_num
		
        

class PostProcessor():

    def __init__(self, model, data_generator, use_idx, 
                 nms_iou_threshold=0.2, 
                 cls_threshold=0.7,
                 chunk_size=100):
        
        
        ##Note: right now, the data_generator for testing is set up to load the "test" dataframe
        #into the "train" attribute of data_generator
        #this is because for the test set you need to make a new datagenerator anyway with the
        #correct symlinks
        
        self.gen = data_generator
        self.model = model
        self.use_idx = use_idx
        self.chunk_size=100
		
        self.nms_iou_threshold = nms_iou_threshold
        self.cls_threshold = cls_threshold
        
        self.evaluated_predictions = []
        self.evaluated_predictions_string = []

		

    def analyze_chunk(self, idx, save_dir, workers=4):
        
        chunk_gen = evaluation_generator(idx, self.gen, batch_size=4)
        
        print('Running NN')
        pred_image = self.model.predict_generator(chunk_gen, use_multiprocessing=True, workers=workers)
        
        Ids = self.gen.train.iloc[idx]['Id']
        
        print('Writing to disk')
        filter_pred_and_save(pred_image, 
                             self.gen.o_xaxis, self.gen.o_yaxis, 
                             self.gen.lyftdata, self.cls_threshold, 
                             save_dir, Ids)        

    
    
    def make_predictions(self, save_dir, gen_workers = 4, nms_workers = 4):
        
        num_chunks = int(np.ceil(self.use_idx.shape[0]/self.chunk_size))
        
        print('Running predictions and saving')
        for i in range(num_chunks):
            idx = self.use_idx[i*self.chunk_size:min((i+1)*self.chunk_size, self.use_idx.shape[0])]
            self.analyze_chunk(idx, save_dir, workers = gen_workers)
            
        print('Running nms suppression')        
        self.evaluated_predictions = nms_from_save(save_dir, 
                                                   self.gen.train['Id'], 
                                                   self.nms_iou_threshold, 
                                                   workers = nms_workers)

            

    
    
    def calculate_mAP(self, over_iou_thresh=0.1, over_class_names=None):
        """For use as a validation tool"""
        if self.evaluated_predictions == []:
            print('Run "make_predictions" first')
            
        if over_class_names is None:
            over_class_names = self.gen.inc_classes
            
        
        if type(over_iou_thresh)!= list:
            over_iou_thresh = list(over_iou_thresh)
            
        precs_over_iou = []
        for iou in over_iou_thresh:
            
            precs = []
        
            for i,idx in enumerate(self.use_idx):
                predictions_gt = prediction_string_to_prediction_dicts_gt(self.gen.train.iloc[idx]['PredictionString'], 
                                                                      self.gen.train.iloc[idx]['Id'])
        
                prec = get_average_precisions(predictions_gt, self.evaluated_predictions[i], over_class_names,iou)

                precs.append(np.array(prec))
            precs_over_iou.append(np.array(precs).mean(axis=0))
            
        
        return np.array(over_iou_thresh),np.array(precs_over_iou)

    def write_prediction_df(self, save_path = 'out.csv'):
        """For use as a test set prediction tool"""
        
        #note: in this case "train" should be the test set!
        df_out = self.gen.train.copy()
        
        if self.evaluated_predictions == []:
            print('Run "make_predictions" first')
            
        for i in range(df_out.shape[0]):
            df_out.iloc[i]['PredictionString'] = box_objs_to_str(self.evaluated_predictions[i], self.gen.num_to_cat)  
        
        df_out.to_csv(save_path,index=False)
        
    def prediction_to_lyft_sdk(self):
        """For use as a test set prediction tool"""
        
        #note: in this case "train" should be the test set!
        df_out = self.gen.train.copy()
        
        if self.evaluated_predictions == []:
            print('Run "make_predictions" first')
        lyft_objects = []
        for i in range(df_out.shape[0]):
            lyft_objects.append(box_objs_to_lyft_sdk_format(self.evaluated_predictions[i], 
                                                            self.gen.num_to_cat, 
                                                            df_out.iloc[i]['Id']))
        
        return lyft_objects        





def convert_to_box_objs(boxes, cls_pred):
    """
    Args:
        pred_array: [N, (score,x,y,z,dy,dx,dz,yaw)]
    returns: 
        array of RotatedBox classes
    """
    

    rot_boxes = [RotatedBox(*boxes[i,1:8], boxes[i,0], cls_pred[i]) for i in range(boxes.shape[0])]
    
    return np.array(rot_boxes)


def compute_ious(rot_box, with_rot_boxes):

    ious = np.zeros(with_rot_boxes.shape[0])
    for i in range(with_rot_boxes.shape[0]):
        ious[i] = rot_box.iou_2d(with_rot_boxes[i])
    return ious




def non_max_suppression(boxes,cls_pred, nms_iou_threshold):

    if boxes.shape[0] == 0:
        return
    

    box_obj_array = convert_to_box_objs(boxes, cls_pred)
    
    scores = np.array(boxes[:, 0])

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

    picks = np.array(picks, dtype=np.int32)
    return box_obj_array[picks]

def compute_iou(rot_box, with_rot_box):
    return rot_box.iou_2d(with_rot_box)


def nms_from_file(base_path, nms_iou_threshold):
    boxes = np.load(base_path + '_boxes.npy')
    cls_pred = np.load(base_path + '_cls.npy')

    return non_max_suppression(boxes,cls_pred, nms_iou_threshold)
    


def filter_pred_and_save(pred_image, x_scale, y_scale, lyftdata, cls_threshold, save_dir, Ids):
    pred_image[:,:,:,0] += x_scale.reshape((1,-1,1))
    pred_image[:,:,:,1] += y_scale.reshape((1,1,-1))
    
    cls_pred = np.argmax(pred_image[:,:,:,8:], axis=3)
    cls_scores = pred_image[:,:,:,8:].max(axis=3)
    
    def pred_iterator(pred_image, cls_pred, cls_scores, Ids):
        for i in range(pred_image.shape[0]):
            idx = np.argwhere(cls_scores[i] > cls_threshold)
            if idx.shape[0] == 0:
                yield (np.array([]),np.array([]),Ids[i])
            
                sample = lyftdata.get('sample', Id)
                lidar_top = lyftdata.get('sample_data', sample['data']['LIDAR_TOP']) 
                ego_pose = lyftdata.get('ego_pose',lidar_top['ego_pose_token'])
                
                qt = Quaternion(ego_pose["rotation"])
                
                boxes = np.zeros((len(idx), 8))
                
                #rotate and shift
                boxes[:,0] = np.dot(qt.rotation_matrix,pred_image[i,idx[:,0], idx[:,1],:3].T).T + np.array(ego_pose["translation"]).reshape((1,-1))
                
                #dy, dz, dx
                boxes[:,1:4] = np.exp(pred_image[i,idx[:,0], idx[:,1],0:3])
                
                #angle: 
                boxes[:,7] = np.arctan2(pred_image[i,idx[:,0], idx[:,1],7],pred_image[i,idx[:,0], idx[:,1],6]) + np.arctan2(qt.rotation_matrix[1,0],qt.rotation_matrix[0,0])
    

            else:
                yield (boxes,cls_pred[i,idx[:,0], idx[:,1]], Ids[i])
    
    if not save_dir[-1] == '/':
        save_dir += '/'
    
    for pred_box, cls_pred, cls_score, Id in pred_iterator(pred_image, cls_pred, cls_scores):
        np.save(save_dir + Id + '_boxes.npy', pred_box)
        np.save(save_dir + Id + '_cls.npy', cls_pred)
    
    
def nms_from_save(save_dir, Ids, nms_iou_threshold, workers):

    #
    def path_iterator(Ids, save_dir):
        for i,Id in enumerate(Ids):
            if i%100 == 0:
                print(i)
            yield save_dir + Id
    
    with Pool(workers) as p:
        picked_boxes = p.map(partial(nms_from_file, 
                    nms_iou_threshold=nms_iou_threshold), path_iterator(Ids, save_dir))
    
    return picked_boxes
    



def box_objs_to_lyft_sdk_format(picked_box_objs,num_to_cat, Id):
        
    predictions = []
    
    if len(picked_box_objs) == 0:
        return predictions
    
    picked_objstrs = np.array([num_to_cat[obj.get_obj()] for obj in picked_box_objs])
    picked_pos_array = np.array([obj.get_array() for obj in picked_box_objs])
    picked_scores = np.array([obj.score for obj in picked_box_objs])
    df = pd.DataFrame(picked_pos_array)
    df.columns = ['x','y','z','dy','dx','dz','yaw']
    df['cat'] = picked_objstrs
    df['score'] = picked_scores


    #translate and rotate:

    
    
    for i in range(df.shape[0]):
        predictions.append({
                'sample_token' : Id,
                'name' : df['cat'][i],
                'score' : df['score'][i],
                'translation' : df[['x','y','z']].iloc[i],
                'size' : df[['dy','dx','dz']].iloc[i],
                'rotation' : [df['yaw'][i],0.,0.,1.],
                })
            
    return predictions

def box_objs_to_str(picked_box_objs, num_to_cat):
    """Convert a series of box objects and scores to a prediction string"""
    
    if len(picked_box_objs) == 0:
        return ''
    picked_objstrs = np.array([num_to_cat[obj.get_obj()] for obj in picked_box_objs])
    picked_pos_array = np.array([obj.get_array() for obj in picked_box_objs])
    picked_scores = np.array([obj.score for obj in picked_box_objs])
    
    df = pd.DataFrame(picked_pos_array)
    df.columns = ['x','y','z','dy','dx','dz','yaw']
    df['cat'] = picked_objstrs
    df['score'] = picked_scores    
    
    
    cols = ['score','x','y','z','dy','dx','dz','yaw','cat']
    
    return df.to_csv(header=False,index=False,columns=cols, sep=' ', line_terminator=' ')
    


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
    
if __name__=='__main__':
    
#    pred_image = np.load('/Users/berend/Google Drive/colab_files/pred_im.npy')
#    xlim = (-102.4,102.4)
#    ylim = (-51.2,51.2)
#    
#    x_scale = np.linspace(-102,102, 256)
#    y_scale = np.linspace(-50.8,50.8, 128)
#    
#    
#    cls_threshold = 0.7
#    nms_iou_threshold = 0.2
#    workers = 1
#    import os
#    os.environ["OMP_NUM_THREADS"] = "1"
#    
#    pred_image[:,:,:,0] += x_scale.reshape((1,-1,1))
#    pred_image[:,:,:,1] += y_scale.reshape((1,1,-1))
#    
#    #this predicts the indices
#    cls_pred = np.argmax(pred_image[:,:,:,8:], axis=3)
#    cls_scores = pred_image[:,:,:,8:].max(axis=3)
#    
#    
#

    
    
    