import glob
import cv2
import numpy as np

#from keras.preprocessing.image import img_to_array, load_img

def scaleRadius(img,scale):
    x=img[int(img.shape[0]/2),:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

scale = 300
image_length = 256
image_height = 256
num_channels = 3
i = 0

data_file = glob.glob('/home/chaitanya/Documents/odseg/Trainingc/*.jpg')

trainData = np.zeros((len(data_file),image_length, image_height, num_channels))

for f in (data_file):
    a=cv2.imread(f)
    a=scaleRadius(a,scale)
#    b=np.zeros(a.shape)
#    b=b.astype(np.uint8) 
#    cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),int(scale*0.9),(1,1,1),-1,8,0)
#    aa=cv2.addWeighted(a,4,cv2.GaussianBlur(a,(0,0),scale/30),-4,128)*b+128*(1-b)
    resized_image = cv2.resize(a, (image_length, image_height))
    resized_image = resized_image.astype(np.float32)
    trainData[i,:,:,:] = resized_image[:,:,:]
    i += 1
    
np.save('testData.npy',trainData)