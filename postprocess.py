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
	
        return np.array([self.x, self.y, self.z, self.dx, self.dy, self.dz, self.th])
		
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

		

    def analyze_chunk(self, idx, workers=4):
        
        
        chunk_gen = evaluation_generator(idx, self.gen, batch_size=4)
        
        print('Running NN')
        pred_im = self.model.predict_generator(chunk_gen, use_multiprocessing=True, workers=workers)
        
        #list of lists
        predictions_fox_idx = []
        
    
        print('Analyzing predictions')
        
        output_boxes_for_iterator = filter_pred(pred_im, 
                                                self.gen.o_xaxis, 
                                                self.gen.o_yaxis, 
                                                self.cls_threshold, 
                                                self.nms_iou_threshold, 
                                                workers=workers)
        
        print('Converting to lyft type data')
        
        
        for i,picked_box_objs in enumerate(output_boxes_for_iterator):
            if i % 10 == 0:
                print(i)
#                
            predictions = box_objs_to_lyft_sdk_format(picked_box_objs,
                                        self.gen.lyftdata, 
                                        self.gen.train.iloc[idx[i]]['Id'], 
                                        self.gen.num_to_cat)
            predictions_fox_idx.append(predictions)


                
        return predictions_fox_idx
    
    def analyze_chunk_to_string(self, idx, workers=4):
        
        
        chunk_gen = evaluation_generator(idx, self.gen, batch_size=4)
        
        print('Running NN')
        pred_im = self.model.predict_generator(chunk_gen, use_multiprocessing=True, workers=workers)
        
        #list of lists
        
        print(pred_im.shape[0])
        
        output_boxes_for_iterator = filter_pred(pred_im, 
                                                self.gen.o_xaxis, 
                                                self.gen.o_yaxis, 
                                                self.cls_threshold, 
                                                self.nms_iou_threshold, 
                                                workers=workers)
        
        
        predictions_fox_idx = []
        
        print('Analyzing predictions ')        
        for i,picked_box_objs in enumerate(output_boxes_for_iterator):
                
            if len(picked_box_objs) == 0:
                predictions_fox_idx.append('')
            else:
                predictions_fox_idx.append(box_objs_to_str(picked_box_objs, 
                                                           self.gen.num_to_cat, 
                                                           self.gen.lyftdata, 
                                                           self.gen.train.iloc[idx[i]]['Id']))        
        
        return predictions_fox_idx
    
    
    
    def make_predictions(self):
        
        num_chunks = int(np.ceil(self.use_idx.shape[0]/self.chunk_size))
        
        
        self.evaluated_predictions = []
        
        for i in range(num_chunks):
            idx = self.use_idx[i*self.chunk_size:min((i+1)*self.chunk_size, self.use_idx.shape[0])]
            
            self.evaluated_predictions += self.analyze_chunk(idx)
            
    def make_predictions_to_string(self,workers=1):
        
        num_chunks = int(np.ceil(self.use_idx.shape[0]/self.chunk_size))
        
        self.evaluated_predictions_string = []
        t0 = time.time()
        for i in range(num_chunks):
            idx = self.use_idx[i*self.chunk_size:min((i+1)*self.chunk_size, self.use_idx.shape[0])]
            
            self.evaluated_predictions_string += self.analyze_chunk_to_string(idx,workers=workers)
            dt = time.time() - t0
            print('Iteration ',i+1,'out of ', num_chunks)
            print('Time left, ', (num_chunks-i)*dt/(i+1))

    
    
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
        
        
        if self.evaluated_predictions_string == []:
            print('Run "make_predictions_to_string" first')
            
        df_out['PredictionString'] = pd.Series(self.evaluated_predictions_string)
        
        df_out.to_csv(save_path,index=False)





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
        ious[i] = rot_box.iou_2d(with_rot_boxes[i])
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

    picks = np.array(picks, dtype=np.int32)
    return box_obj_array[picks]

def compute_iou(rot_box, with_rot_box):
    return rot_box.iou_2d(with_rot_box)


vectorized_iou = np.vectorize(compute_iou, otypes=[float])


def non_max_suppression_vector(pred_box_array,cls_pred_array,cls_scores_array, nms_iou_threshold):
    ##Faster hopefully
    
    assert pred_box_array.shape[0] > 0
    

    box_obj_array = convert_to_box_objs(pred_box_array,cls_pred_array,cls_scores_array)
    
    scores = pred_box_array[:, -1]

    # Get indicies of boxes sorted by scores (highest first)
    
    max_pred = 1e10
    
    max_pred = min(scores.shape[0], max_pred)
    
    ixs = scores.argsort()[::-1][:max_pred]
    

    picks = []
    while ixs.size > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        picks.append(i)
        # Compute IoU of the picked box with the rest
        iou = vectorized_iou(box_obj_array[i], box_obj_array[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        keep_ixs = np.where(iou <= nms_iou_threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = ixs[keep_ixs]

    picks = np.array(picks, dtype=np.int32)
    return box_obj_array[picks]


def non_max_suppression_from_file(i):
    
    a = np.load('{}_a.npy'.format(i))
    b = np.load('{}_b.npy'.format(i))
    c = np.load('{}_c.npy'.format(i))
    
    
    
    return non_max_suppression(a,b,c, 0.2)


def non_max_suppression_for_multi(args, nms_iou_threshold):
    
    if args is None:
        return np.array([])
    
    pred_box_array,cls_pred_array,cls_scores_array = args
    
    return non_max_suppression(pred_box_array,cls_pred_array,cls_scores_array, nms_iou_threshold)

def filter_pred(pred_image, x_scale, y_scale, cls_threshold, nms_iou_threshold, workers=4):
    #I should be able to improve this for batch type inference, at least the first type
        
    pred_image[:,:,:,0] += x_scale.reshape((1,-1,1))
    pred_image[:,:,:,1] += y_scale.reshape((1,1,-1))
    
    #this predicts the indices
    cls_pred = np.argmax(pred_image[:,:,:,8:], axis=3)
    cls_scores = pred_image[:,:,:,8:].max(axis=3)
    
    
    def pred_iterator(pred_image, cls_pred, cls_scores):
        for i in range(pred_image.shape[0]):
            idx = np.argwhere(cls_scores[i] > cls_threshold)
            if idx.shape[0] == 0:
                yield None
            else:
                yield (pred_image[i,idx[:,0], idx[:,1],:8],cls_pred[i,idx[:,0], idx[:,1]], cls_scores[i,idx[:,0],idx[:,1]])
    
    chunksize = int(np.ceil(pred_image.shape[0]/workers))
    
    
    #alternatively: starmap directly on nms_...
    with Pool(workers) as p:
        output_boxes_for_iterator = p.imap(partial(non_max_suppression_for_multi, 
                                       nms_iou_threshold=nms_iou_threshold),
                                       pred_iterator(pred_image, cls_pred, cls_scores), chunksize=chunksize)
        output_boxes_for_iterator = [b for b in output_boxes_for_iterator]
        
#    output_boxes_for_iterator = []
#    for arg in pred_iterator(pred_image, cls_pred, cls_scores):
#        
#        output_boxes_for_iterator.append(non_max_suppression_for_multi(arg, 0.2))



    return output_boxes_for_iterator


def box_objs_to_lyft_sdk_format(picked_box_objs,lyftdata, Id, num_to_cat):
        
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
                'translation' : df[['x','y','z']].iloc[i],
                'size' : df[['dy','dx','dz']].iloc[i],
                'rotation' : [df['yaw'][i],0.,0.,1.],
                })
            
    return predictions

def box_objs_to_str(picked_box_objs, num_to_cat, lyftdata, Id):
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
    
    
    sample = lyftdata.get('sample', Id)
    lidar_top = lyftdata.get('sample_data', sample['data']['LIDAR_TOP']) 
    ego_pose = lyftdata.get('ego_pose',lidar_top['ego_pose_token'])
    
    qt = Quaternion(ego_pose["rotation"])
    
    df[['x','y','z']] = np.dot(qt.rotation_matrix,df[['x','y','z']].T).T

    df[['x','y','z']] = df[['x','y','z']] + np.array(ego_pose["translation"]).reshape((1,-1))
    
    df['yaw'] = df['yaw'] - qt.angle ##check this!! the data_preprocessing was -, with "inverse", I think you change only one of the two		
    
    
    
    cols = ['score','x','y','z','dx','dy','dz','yaw','cat']
    
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
    
    pred_image = np.load('/Users/berend/Google Drive/colab_files/pred_im.npy')
    xlim = (-102.4,102.4)
    ylim = (-51.2,51.2)
    
    x_scale = np.linspace(-102,102, 256)
    y_scale = np.linspace(-50.8,50.8, 128)
    
    
    cls_threshold = 0.7
    nms_iou_threshold = 0.2
    workers = 1
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    
    pred_image[:,:,:,0] += x_scale.reshape((1,-1,1))
    pred_image[:,:,:,1] += y_scale.reshape((1,1,-1))
    
    #this predicts the indices
    cls_pred = np.argmax(pred_image[:,:,:,8:], axis=3)
    cls_scores = pred_image[:,:,:,8:].max(axis=3)
    
    
    def pred_iterator(pred_image, cls_pred, cls_scores):
        for i in range(pred_image.shape[0]):
            idx = np.argwhere(cls_scores[i] > cls_threshold)
            if idx.shape[0] == 0:
                yield None
            else:
                yield (pred_image[i,idx[:,0], idx[:,1],:8],cls_pred[i,idx[:,0], idx[:,1]], cls_scores[i,idx[:,0],idx[:,1]])
    
    import time
    t0 = time.time()
    for i,args in enumerate(pred_iterator(pred_image, cls_pred, cls_scores)):
        a,b,c = args
        np.save('{}_a.npy'.format(i),a)
        np.save('{}_b.npy'.format(i),b)
        np.save('{}_c.npy'.format(i),c)    
#    box_list = filter_pred(pred_image, x_scale, y_scale, cls_threshold, nms_iou_threshold, workers=4)
    t1 = time.time()
    print(t1-t0)
    
    for i in range(100):
        a = np.load('{}_a.npy'.format(i))
        b = np.load('{}_b.npy'.format(i))
        c = np.load('{}_c.npy'.format(i))
        print(a.max(),b.max(),c.max())
    t2 = time.time()
    print(t2-t1)
    
    t2 = time.time()
    workers = 8
    with Pool(workers) as p:
        output = p.imap_unordered(non_max_suppression_from_file, range(100), chunksize=25)
        output = [o for o in output]
    t3 = time.time()
    print(t3-t2)
        
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

    
    
    
    
    
    
    