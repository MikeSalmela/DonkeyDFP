#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:03:05 2019

@author: mike
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import math
from skimage.morphology import skeletonize

def splitImage(image, border=0.5, top=False):
    height, width = image.shape[:2]
    if(not top):
        start_row, start_col = int(height * border), int(0)
        end_row, end_col = int(height), int(width)
        cropped_bot = image[start_row:end_row , start_col:end_col]
        return cropped_bot
    else:
        start_row, start_col = int(0), int(0)
        end_row, end_col = int(height * border), int(width)
        cropped_top = image[start_row:end_row , start_col:end_col]
        return cropped_top


def normalize(x):
    return (x - np.float16(128)) / 128


def tupleInteger(x, bot=False, dimx=640, dimy=108, normalize=True):
    
    if(math.isnan(x[0]) or math.isnan(x[1])):
        return x
    h = 0    
    if (bot) :
        h = dimy/2
    if(normalize):
        x = ((x[1] - dimx/2)/dimx, (x[0] + h - dimy/2)/dimy)
    else:
        x = (int(x[1]), int(x[0]  + h))

    return x

def pickWhite(image):
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    r = r > 200
    g = g > 200
    b = b > 200
    mask = r*g*b
    return mask

def yellowOnly(img):
    img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowYellow = np.array([13, 0, 0])
    highYellow = np.array([25, 255, 255])
    img = cv2.inRange(img, lowYellow, highYellow) 
    return img

def featurePick(img):
    copy = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plt.imshow(img)
    lowYellow = np.array([10, 0, 0])
    highYellow = np.array([25, 255, 255])
    mask = cv2.inRange(img, lowYellow, highYellow) 
    """
    sensitivity = 30
    lowWhite = np.array([0,0,255-sensitivity])
    highWhite = np.array([255,sensitivity,255])    
    mask += cv2.inRange(img, lowWhite, highWhite) 
    """ 
    mask += pickWhite(copy)
    mask = mask.clip(max=1)
    return mask

def useMask(image, mask):
    image[:,:,0] = image[:,:,0]*mask
    image[:,:,1] = image[:,:,1]*mask
    image[:,:,2] = image[:,:,2]*mask
    return image

def linePoints(img, second=False, dimy=240, dimx=640, normalize=True):
    if(not second):
        img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lowYellow = np.array([13, 0, 0])
        highYellow = np.array([25, 255, 255])
        img = cv2.inRange(img, lowYellow, highYellow) 
        img = img.clip(max=1)
        
        img = skeletonize(img)
        
    bot = splitImage(img)
    top = splitImage(img, top=True)
    if(second):
        if(np.sum(bot) == 0 or  np.sum(top) == 0):
            return (0,0), (0,0)
    
    
    bot_center = tupleInteger(center_of_mass(bot), bot=True, dimy=dimy, dimx=dimx, normalize=normalize)
    top_center = tupleInteger(center_of_mass(top), dimy=dimy, dimx=dimx, normalize=normalize)
    
    if ( math.isnan(bot_center[0]) or math.isnan(bot_center[1]) ):
        p = linePoints(top, second=True)
    elif ( math.isnan(top_center[0]) or math.isnan(top_center[1]) ):
        p = linePoints(bot, second=True)
    p = [bot_center[0], bot_center[1], top_center[0], top_center[1]]
    return np.array(p)


def lineVec(img):
    img = yellowOnly(img)
    #img = cropImage(img)
    width = float(img.shape[1])
    top = splitImage(img, top=True)
    bot = splitImage(img)
     
    x_1 = (center_of_mass(top)[1] - width/2)/width 
    x_2 = (center_of_mass(bot)[1] - width/2)/width
    return np.array([x_1, x_2])

