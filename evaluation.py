import os, glob
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

i=0
data_file = glob.glob('/home/chaitanya/Documents/odseg/*.jpg')
files = []
data_file_mask = glob.glob('/home/chaitanya/Documents/odseg/ODSegmentationTrainingSet/grayMask/*.tif')

predMask = np.zeros((len(data_file),2848,4288,1))
dataMask = np.zeros((len(data_file_mask),2848,4288,1))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

for f in (data_file):
    base = os.path.basename("/home/chaitanya/Documents/odseg/" + f)
    fileName = os.path.splitext(base)[0]
    files.append(fileName)
    a=cv2.imread(f)
    a = rgb2gray(a)
    a = img_to_array(a)
    predMask[i,:,:,:] = a[:,:,:] # 0-255 range already
    i=i+1
    
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def jaccard_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    sum_ = np.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return np.mean(jac)

def perf_measure(y_actual_tensor, y_hat_tensor):
    y_actual = y_actual_tensor.flatten()
    y_hat = y_hat_tensor.flatten()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
    
for k in (data_file_mask):
    base = os.path.basename("/home/chaitanya/Documents/odseg/ODSegmentationTrainingSet/grayMask/" + k)
    fileName = os.path.splitext(base)[0]
    fileName = fileName[0:8]
    index = files.index(fileName)
    image = cv2.imread(k)
    gray = rgb2gray(image)
    gray_image = img_to_array(gray)
    dataMask[index, :, :, :] = gray_image[:, :, :]

dataMask = dataMask.astype('float32')
dataMask /= 255.  # scale masks to [0, 1]
predMask = predMask.astype('float32')
predMask /= 255.  # scale masks to [0, 1]

dice_coefficient = dice_coef(dataMask, predMask)
jaccard_index = jaccard_coef(dataMask, predMask) 

#dataMask = dataMask.astype(np.uint8)
#predMask = dataMask.astype(np.uint8)
                           
[TP, FP, TN, FN] = perf_measure(dataMask, predMask)

sensitivity =TP/(TP+FN)
specificity= TN/(TN+FP)

print('Dice Index', dice_coefficient)
print('Jaccard Index', jaccard_index)
print('Sensitivity', sensitivity)
print('Specificity', specificity)


