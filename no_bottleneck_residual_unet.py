from residual import Residual

from keras import backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, merge, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import flip_axis, random_channel_shift

from augmentation import random_rotation, random_zoom

trainData = np.load('data.npy')
trainMask = np.load('dataMask.npy')

trainData = trainData.astype('float32')
mean = np.mean(trainData)  # mean for data centering
std = np.std(trainData)  # std for data normalization

trainData -= mean
trainData /= std

trainMask = trainMask.astype('float32')
trainMask /= 255.  # scale masks to [0, 1]

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def get_unet():
    inputs = Input((256, 256, 3))
    
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Residual(16, 32, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Residual(32, 32, pool1)
    conv2 = Residual(32, 64, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Residual(64, 64, pool2)
    conv3 = Residual(64, 128, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Residual(128, 128, pool3)
    conv4 = Residual(128, 256, conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Residual(256, 256, pool4)
    conv5 = Residual(256, 128, conv5)
    drop5 = Dropout(0.2)(conv5)
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Residual(384, 128, merge6)
    conv6 = Residual(128, 64, conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Residual(192, 64, merge7)
    conv7 = Residual(64, 32, conv7)
    
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Residual(96, 32, merge8)
    conv8 = Residual(32, 16, conv8)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Residual(48, 16, merge9)
    conv9 = Residual(16, 2, conv9)
    conv9 = Residual(2, 1, conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(input = inputs, output = conv10)

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef, jaccard_coef, 'accuracy'])
#    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

model = get_unet()

def Augmentation(X, Y):
        print('Augmentation model...')
        total = len(X)
        x_train, y_train = [], []
        
        for i in range(total):
            x, y = X[i], Y[i]
            #standart
            x_train.append(x)
            y_train.append(y)
        
#            for _ in xrange(1):
#                _x, _y = elastic_transform(x[0], y[0], 100, 20)
#                x_train.append(_x.reshape((1,) + _x.shape))
#                y_train.append(_y.reshape((1,) + _y.shape))
            
            #flip x
            x_train.append(flip_axis(x, 2))
            y_train.append(flip_axis(y, 2))
            #flip y
            x_train.append(flip_axis(x, 1))
            y_train.append(flip_axis(y, 1))
            #continue
            #zoom
            for _ in range(5):
                _x, _y = random_zoom(x, y, (0.9, 1.1))
                x_train.append(_x)
                y_train.append(_y)
            for _ in range(8):
                _x, _y = random_rotation(x, y, 5)
                x_train.append(_x)
                y_train.append(_y)
            #intentsity
            for _ in range(10):
                _x = random_channel_shift(x, 5.0)
                x_train.append(_x)
                y_train.append(y)
    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train
    
def AugmentationValidation(X, Y):
        print('Augmentation model...')
        total = len(X)
        x_train, y_train = [], []
        
        for i in range(total):
            x, y = X[i], Y[i]
            #standart
            x_train.append(x)
            y_train.append(y)
        
#            for _ in xrange(1):
#                _x, _y = elastic_transform(x[0], y[0], 100, 20)
#                x_train.append(_x.reshape((1,) + _x.shape))
#                y_train.append(_y.reshape((1,) + _y.shape))
            
            #flip x
            x_train.append(flip_axis(x, 2))
            y_train.append(flip_axis(y, 2))
            #flip y
            x_train.append(flip_axis(x, 1))
            y_train.append(flip_axis(y, 1))
            #continue
            #zoom
            for _ in range(1):
                _x, _y = random_zoom(x, y, (0.9, 1.1))
                x_train.append(_x)
                y_train.append(_y)
            for _ in range(0):
                _x, _y = random_rotation(x, y, 5)
                x_train.append(_x)
                y_train.append(_y)
            #intentsity
            for _ in range(1):
                _x = random_channel_shift(x, 5.0)
                x_train.append(_x)
                y_train.append(y)
    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train
    
x_train, y_train = Augmentation(trainData[0:40,:,:,:], trainMask[0:40,:,:,:])
x_validation, y_validation = AugmentationValidation(trainData[40:54,:,:,:], trainMask[40:54,:,:,:])

def schedule(epoch):
    if epoch<=20:
        return 1e-4
    elif epoch<=40:
        return 1e-5
    else:
        return 1e-6

lr_schedule= LearningRateScheduler(schedule)

model.fit(x_train, y_train,
          batch_size=16, nb_epoch=50,
          verbose=1, validation_data=(x_validation, y_validation), shuffle=True, callbacks=[lr_schedule])

model.save('unet_residual_no_bottleneck_final.h5')