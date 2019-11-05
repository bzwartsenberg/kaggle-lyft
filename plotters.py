#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:19:32 2019

@author: berend
"""

#plotters

import pdb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon,mapping

# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix



def get_box(sample_annotation_token, lyftdata):
    """Instantiates a Box class from a sample annotation record.
    Args:
        sample_annotation_token: Unique sample_annotation identifier.
    Returns:
    """
    record = lyftdata.get("sample_annotation", sample_annotation_token)
    return Box(
        record["translation"],
        record["size"],
        Quaternion(record["rotation"]),
        name=record["category_name"],
        token=record["token"],
    )
    
    
def plot_box(ax, box, lyftdata, view = np.eye(3),normalize = False,linewidth =1):
    
    colors = ['red', 'green','blue','yellow','cyan','magenta','orange','grey','black']
    
    color_dict = {cat_record['name'] : c for cat_record,c in zip(lyftdata.category,colors)}
    
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        ax.plot(
            [corners.T[i][0], corners.T[i + 4][0]],
            [corners.T[i][1], corners.T[i + 4][1]],
            color=color_dict[box.name],
            linewidth=linewidth,
        )

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0])
    draw_rect(corners.T[4:], colors[1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    ax.plot(
        [center_bottom[0], center_bottom_forward[0]],
        [center_bottom[1], center_bottom_forward[1]],
        color=colors[0],
        linewidth=linewidth,
    )    

def plot_pc(ax, pc):
    
    pts = pc.points
        
    ax.scatter(pts[0],pts[1], c = pts[2], cmap = 'viridis', s = 0.5)



def plot_sample(sample, lyftdata, lidar = True, ax = None):
    lidar_top = lyftdata.get('sample_data', sample['data']['LIDAR_TOP']) 


    pc = LidarPointCloud.from_file(Path(lidar_top['filename']))
    
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = lyftdata.get("calibrated_sensor", lidar_top["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

#    # Second step: transform to the global frame.
    poserecord = lyftdata.get("ego_pose", lidar_top["ego_pose_token"])
#    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
#    pc.translate(np.array(poserecord["translation"]))    

    
    if ax is None:
        fig,ax = plt.subplots(figsize = (15,15))
    
    if lidar:
        plot_pc(ax, pc)
    
    for ann in sample['anns']:
        box = get_box(ann,lyftdata)
        
        box.translate(-np.array(poserecord["translation"]))
        box.rotate(Quaternion(poserecord["rotation"]).inverse)
        #box.render(ax)
        plot_box(ax, box,lyftdata)
        

def plot_sample_global(sample, lyftdata, lidar = True, ax = None):
    lidar_top = lyftdata.get('sample_data', sample['data']['LIDAR_TOP']) 


    pc = LidarPointCloud.from_file(Path(lidar_top['filename']))
    
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = lyftdata.get("calibrated_sensor", lidar_top["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

#    # Second step: transform to the global frame.
    poserecord = lyftdata.get("ego_pose", lidar_top["ego_pose_token"])
    pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
    pc.translate(np.array(poserecord["translation"]))    

    
    if ax is None:
        fig,ax = plt.subplots(figsize = (15,15))
    
    if lidar:
        plot_pc(ax, pc)
    
    for ann in sample['anns']:
        box = get_box(ann,lyftdata)
        
        #box.translate(-np.array(poserecord["translation"]))
        #box.rotate(Quaternion(poserecord["rotation"]).inverse)
        #box.render(ax)
        plot_box(ax, box,lyftdata)        
        

def plot_output(generator, output_map, channel = 8, ax = None):
    
    if ax is None:
        fig,ax = plt.subplots(figsize = (10,10))
    
    ax.pcolormesh(generator.o_xaxis_lim, generator.o_yaxis_lim, output_map[:,:,channel].T)
    
    


        
    
def plot_Rotated_box(ax, rotated_box):
    
    coords = np.array(mapping(rotated_box.base_poly)['coordinates'])[0]
    
    ax.plot(coords[:,0],coords[:,1])
    
    
def plot_boxes(box_array, ax = None):
    
    if ax is None:
        fig,ax = plt.subplots(figsize = (5,5))
    
    for box in box_array:
        plot_Rotated_box(ax, box)    