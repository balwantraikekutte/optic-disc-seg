import os, glob
import cv2
import numpy as np

pred = np.load('predMask2.npy')

data_file = glob.glob('/home/chaitanya/Documents/odseg/Trainingc/*.jpg')
files = []
data_file_mask = glob.glob('/home/chaitanya/Documents/odseg/ODSegmentationTrainingSet/grayMask/*.tif')

for f in (data_file):
    base = os.path.basename("/home/chaitanya/Documents/odseg/Trainingc/" + f)
    fileName = os.path.splitext(base)[0]
    files.append(fileName)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

for k in (data_file_mask):
    base = os.path.basename("/home/chaitanya/Documents/odseg/ODSegmentationTrainingSet/grayMask/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName[0:8]
    index = files.index(fileName)
    imageMask = cv2.imread(k)
    grayMask = rgb2gray(imageMask)
    height, width = np.shape(grayMask)
    predMask = pred[index,:,:,:]
    predMask = predMask*255
    predMask = cv2.resize(predMask, (width, height))
    predMask = predMask.astype(np.uint8)
    filename = fileName + '.jpg'
    cv2.imwrite(filename, predMask)

    