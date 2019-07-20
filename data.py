#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:13:15 2018

@author: ck807
"""
import glob
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array

"create test set"
data_file_mask = glob.glob('/home/chaitanya/Documents/odseg/ODSegmentationTrainingSet/grayMask/*.tif')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

data_mask = np.zeros((len(data_file_mask),256,256,1))

for i in range(0, len(data_file_mask)):
    image = cv2.imread(data_file_mask[i])
    gray = rgb2gray(image)
    resized_image = cv2.resize(gray, (256, 256))
    gray_image = img_to_array(resized_image)
    data_mask[i, :, :, :] = gray_image[:, :, :]
    
np.save('dataMask.npy', data_mask)
