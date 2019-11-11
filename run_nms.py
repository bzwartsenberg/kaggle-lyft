#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:33:24 2019

@author: berend
"""

### run nms from dir


from postprocess import nms_from_save, box_objs_to_str
import pandas as pd


def run_nms_from_path(path, test_df,out_path,
                      nms_iou_threshold = 0.1, workers = 4, num_to_cat = None):
    
    if num_to_cat is None:
        inc_classes = ['car',
             'pedestrian',
             'animal',
             'other_vehicle',
             'bus',
             'motorcycle',
             'truck',
             'emergency_vehicle',
             'bicycle']

        cat_to_num = {cat : i for i,cat in enumerate(inc_classes)}
        num_to_cat = {v : k for k,v in cat_to_num.items()}

    

    evaluated_predictions = nms_from_save(path, 
                   test_df['Id'], 
                   nms_iou_threshold, 
                   workers = workers)

    df_out = test_df.copy()
    
        
    pred_strs = [box_objs_to_str(evaluated_predictions[i], num_to_cat) for i in range(df_out.shape[0])]
    
    df_out['PredictionString'] = pd.Series(pred_strs)
    
    df_out.to_csv(out_path,index=False)
    
    
    
if __name__ == '__main__':
    
    zipfile_path = '/Users/berend/Documents/temp/lyft_nms/temp.zip'
    extract_to = '/Users/berend/Documents/temp/lyft_nms/'
    
    import zipfile
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    
    test_csv_path = '/Users/berend/Documents/temp/lyft_nms/sample_submission.csv'
    
    test_df = pd.read_csv(test_csv_path)
    
    out_path = '/Users/berend/Documents/temp/lyft_nms/submission_Nov10th_0.csv'
    
    
    run_nms_from_path(extract_to + '/temp/', test_df,out_path,
                          nms_iou_threshold = 0.1, workers = 4, num_to_cat = None)        

    
    
    
    
    
    