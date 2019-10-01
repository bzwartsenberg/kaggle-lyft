#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:19:24 2019

@author: berend
"""


#utils

import numpy as np


def decode_predictionstring(pred_str):
    pred_ar = np.array(pred_str.split(' ')).reshape((-1,8))[:]
    obj = pred_ar[:,-1]
    pos = pred_ar[:,:-1]

    pos = pos.astype('float32')
    return pos, obj



def points_between_lines_at_x(x, a1, b1, a2, b2):
    
    y1 = x*a1+b1
    y2 = x*a2+b2
    
    y_min = min(y1,y2)
    y_max = max(y1,y2)

    ys = np.arange(np.int(y_min)+1, np.int(y_max)+1)
    
    return np.concatenate([x*np.ones(ys.shape).astype('int'),ys]).reshape((2,-1)).T
    
    
def get_line(p1, p2):
    
    a = (p2[1] - p1[1])/ (p2[0] - p1[0]) #dy/dx
    #a*x + b = y --> -a*x  y = b
    b = p1[1] - a*p1[0]
    
    return a,b
    
    
    
def get_points_in_a_rotated_box(corners):
    
    sorted_corners = corners[np.argsort(corners[:,0])]
    #corners can be floats
    c1x, c2x,c3x,c4x = sorted_corners[:,0]
    c1y, c2y,c3y,c4y = sorted_corners[:,1]
    #easiest is x1 == x2 & x3 == x4:
    pts = []
    if (c1x == c2x) and (c3x == c4x) and (c2x != c3x):
        
        if np.sign(c2y - c1y) != np.sign(c4y - c3y):
            #make sure lines don't cross over:
            c4y,c3y = c3y,c4y
            c4x,c3x = c3x,c4x
        
        a1,b1 = get_line([c1x,c1y], [c3x,c3y])
        a2,b2 = get_line([c2x,c2y], [c4x,c4y])
        
        for x in range(np.int(c1x), np.int(c3x)+1):
            pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))
            
    elif (c2x == c3x):
        for x in range(np.int(c1x), np.int(c4x)+1):
            if x < c3x:
                a1,b1 = get_line([c1x,c1y], [c2x,c2y]) 
                a2,b2 = get_line([c1x,c1y], [c3x,c3y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))
            else:
                a1,b1 = get_line([c2x,c2y], [c4x,c4y]) 
                a2,b2 = get_line([c3x,c3y], [c4x,c4y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))                

    
    elif (c1x == c2x):
        if np.sign(c2y - c1y) != np.sign(c4y - c3y):
            #make sure lines don't cross over, but swap 1 and 2:
            #Note: this is not the right way for all polygons, but for rectangles it should work
            c2y,c1y = c1y,c2y
            c2x,c1x = c1x,c2x
        
        for x in range(np.int(c1x), np.int(c4x)+1):
            if x < c3x:
                a1,b1 = get_line([c1x,c1y], [c3x,c3y]) 
                a2,b2 = get_line([c2x,c2y], [c4x,c4y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))
            else:
                a1,b1 = get_line([c3x,c3y], [c4x,c4y]) 
                a2,b2 = get_line([c2x,c2y], [c4x,c4y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))        
    elif (c3x == c4x):
        if np.sign(c2y - c1y) != np.sign(c4y - c3y):
            #Note: this is not the right way for all polygons, but for rectangles it should work
            c4y,c3y = c3y,c4y
            c4x,c3x = c3x,c4x
        
        for x in range(np.int(c1x), np.int(c4x)+1):
            if x < c2x:
                a1,b1 = get_line([c1x,c1y], [c3x,c3y]) 
                a2,b2 = get_line([c1x,c1y], [c2x,c2y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))
            else:
                a1,b1 = get_line([c1x,c1y], [c3x,c3y]) 
                a2,b2 = get_line([c2x,c2y], [c4x,c4y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))      
    
    elif (c1x < c2x) and (c2x < c3x) and (c3x < c4x):
        
        for x in range(np.int(c1x), np.int(c4x)+1):
            if x < c2x:
                a1,b1 = get_line([c1x,c1y], [c3x,c3y]) 
                a2,b2 = get_line([c1x,c1y], [c2x,c2y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))
            elif x < c3x:
                a1,b1 = get_line([c1x,c1y], [c3x,c3y]) 
                a2,b2 = get_line([c2x,c2y], [c4x,c4y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))
            else:
                a1,b1 = get_line([c3x,c3y], [c4x,c4y]) 
                a2,b2 = get_line([c2x,c2y], [c4x,c4y])
                pts.append(points_between_lines_at_x(x, a1, b1, a2, b2))                
        
        
    
    else:
        raise RuntimeError('Not a valid rectangle')

    return np.concatenate(pts, axis = 0)
    
