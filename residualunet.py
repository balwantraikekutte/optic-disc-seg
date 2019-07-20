from keras import backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, merge, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import flip_axis, random_channel_shift

from keras.layers import BatchNormalization, Dense, Flatten, Lambda, Convolution2D
from keras.layers.advanced_activations import ELU, LeakyReLU

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
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.2)(conv5)
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(input = inputs, output = conv10)

    adam = Adam(lr=1e-5)
    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef, jaccard_coef, 'accuracy'])
#    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

model = get_unet()

#def _shortcut(_input, residual):
#    stride_width = _input._keras_shape[1] / residual._keras_shape[1]
#    stride_height = _input._keras_shape[2] / residual._keras_shape[2]
#    equal_channels = residual._keras_shape[3] == _input._keras_shape[3]
#
#    shortcut = _input
#    # 1 X 1 conv if shape is different. Else identity.
#    if stride_width > 1 or stride_height > 1 or not equal_channels:
#        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
#                                 subsample=(stride_width, stride_height),
#                                 init="he_normal", border_mode="valid")(_input)
#
#    return merge([shortcut, residual], mode="sum")
#
#def inception_block(inputs, depth, batch_mode=0, splitted=False, activation='relu'):
#    assert depth % 16 == 0
#    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
#    
#    c1_1 = Convolution2D(int(depth/4), (1, 1), init='he_normal', border_mode='same')(inputs)
#    
#    c2_1 = Convolution2D(int(depth/8*3), (1, 1), init='he_normal', border_mode='same')(inputs)
#    c2_1 = actv()(c2_1)
#    if splitted:
#        c2_2 = Convolution2D(int(depth/2), (1, 3), init='he_normal', border_mode='same')(c2_1)
#        c2_2 = BatchNormalization(axis=1)(c2_2)
#        c2_2 = actv()(c2_2)
#        c2_3 = Convolution2D(int(depth/2), (3, 1), init='he_normal', border_mode='same')(c2_2)
#    else:
#        c2_3 = Convolution2D(int(depth/2), (3, 3), init='he_normal', border_mode='same')(c2_1)
#    
#    c3_1 = Convolution2D(int(depth/16), (1, 1), init='he_normal', border_mode='same')(inputs)
#    #missed batch norm
#    c3_1 = actv()(c3_1)
#    if splitted:
#        c3_2 = Convolution2D(int(depth/8), (1, 5), init='he_normal', border_mode='same')(c3_1)
#        c3_2 = BatchNormalization(axis=1)(c3_2)
#        c3_2 = actv()(c3_2)
#        c3_3 = Convolution2D(int(depth/8), (5, 1), init='he_normal', border_mode='same')(c3_2)
#    else:
#        c3_3 = Convolution2D(int(depth/8), (5, 5), init='he_normal', border_mode='same')(c3_1)
#    
#    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), border_mode='same')(inputs)
#    c4_2 = Convolution2D(int(depth/8), 1, 1, init='he_normal', border_mode='same')(p4_1)
#    
#    res = merge([c1_1, c2_3, c3_3, c4_2], mode='concat', concat_axis=1)
#    res = BatchNormalization(axis=1)(res)
#    res = actv()(res)
#    return res
#
#def rblock(inputs, num, depth, scale=0.1):    
#    residual = Convolution2D(depth, (num, num), border_mode='same')(inputs)
#    residual = BatchNormalization(axis=1)(residual)
#    residual = Lambda(lambda x: x*scale)(residual)
#    res = _shortcut(inputs, residual)
#    return ELU()(res) 
#
#def NConvolution2D(nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
#    def f(_input):
#        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
#                              border_mode=border_mode)(_input)
#        norm = BatchNormalization(axis=1)(conv)
#        return ELU()(norm)
#
#    return f
#
#def BNA(_input):
#    inputs_norm = BatchNormalization(axis=1)(_input)
#    return ELU()(inputs_norm)
#
#def reduction_a(inputs, k=64, l=64, m=96, n=96):
#    "35x35 -> 17x17"
#    inputs_norm = BNA(inputs)
#    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)
#    
#    conv2 = Convolution2D(n, (3, 3), subsample=(2,2), border_mode='same')(inputs_norm)
#    
#    conv3_1 = NConvolution2D(k, (1, 1), subsample=(1,1), border_mode='same')(inputs_norm)
#    conv3_2 = NConvolution2D(l, (3, 3), subsample=(1,1), border_mode='same')(conv3_1)
#    conv3_2 = Convolution2D(m, (3, 3), subsample=(2,2), border_mode='same')(conv3_2)
#    
#    res = merge([pool1, conv2, conv3_2], mode='concat', concat_axis=1)
#    return res
#
#def reduction_b(inputs):
#    "17x17 -> 8x8"
#    inputs_norm = BNA(inputs)
#    pool1 = MaxPooling2D((3,3), strides=(2,2), border_mode='same')(inputs_norm)
#    #
#    conv2_1 = NConvolution2D(64, (1, 1), subsample=(1,1), border_mode='same')(inputs_norm)
#    conv2_2 = Convolution2D(96, (3, 3), subsample=(2,2), border_mode='same')(conv2_1)
#    #
#    conv3_1 = NConvolution2D(64, (1, 1), subsample=(1,1), border_mode='same')(inputs_norm)
#    conv3_2 = Convolution2D(72, (3, 3), subsample=(2,2), border_mode='same')(conv3_1)
#    #
#    conv4_1 = NConvolution2D(64, (1, 1), subsample=(1,1), border_mode='same')(inputs_norm)
#    conv4_2 = NConvolution2D(72, (3, 3), subsample=(1,1), border_mode='same')(conv4_1)
#    conv4_3 = Convolution2D(80, (3, 3), subsample=(2,2), border_mode='same')(conv4_2)
#    #
#    res = merge([pool1, conv2_2, conv3_2, conv4_3], mode='concat', concat_axis=1)
#    return res
#    
#def get_unet_inception_2head():
#    splitted = True
#    act = 'elu'
#    
#    inputs = Input((128, 128, 1), name='main_input')
#    conv1 = inception_block(inputs, 32, batch_mode=2, splitted=splitted, activation=act)
#    #conv1 = inception_block(conv1, 32, batch_mode=2, splitted=splitted, activation=act)
#    
#    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#    pool1 = NConvolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
#    pool1 = Dropout(0.5)(pool1)
#    
#    conv2 = inception_block(pool1, 64, batch_mode=2, splitted=splitted, activation=act)
#    #pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#    pool2 = NConvolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
#    pool2 = Dropout(0.5)(pool2)
#    
#    conv3 = inception_block(pool2, 128, batch_mode=2, splitted=splitted, activation=act)
#    #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#    pool3 = NConvolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
#    pool3 = Dropout(0.5)(pool3)
#     
#    conv4 = inception_block(pool3, 256, batch_mode=2, splitted=splitted, activation=act)
#    #pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#    pool4 = NConvolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
#    pool4 = Dropout(0.5)(pool4)
#    
#    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
#    #conv5 = inception_block(conv5, 512, batch_mode=2, splitted=splitted, activation=act)
#    conv5 = Dropout(0.5)(conv5)
#    
#    after_conv4 = rblock(conv4, 1, 256)
#    up6 = merge([UpSampling2D(size=(2, 2))(conv5), after_conv4], mode='concat', concat_axis=1)
#    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)
#    conv6 = Dropout(0.5)(conv6)
#    
#    after_conv3 = rblock(conv3, 1, 128)
#    up7 = merge([UpSampling2D(size=(2, 2))(conv6), after_conv3], mode='concat', concat_axis=1)
#    conv7 = inception_block(up7, 128, batch_mode=2, splitted=splitted, activation=act)
#    conv7 = Dropout(0.5)(conv7)
#    
#    after_conv2 = rblock(conv2, 1, 64)
#    up8 = merge([UpSampling2D(size=(2, 2))(conv7), after_conv2], mode='concat', concat_axis=1)
#    conv8 = inception_block(up8, 64, batch_mode=2, splitted=splitted, activation=act)
#    conv8 = Dropout(0.5)(conv8)
#    
#    after_conv1 = rblock(conv1, 1, 32)
#    up9 = merge([UpSampling2D(size=(2, 2))(conv8), after_conv1], mode='concat', concat_axis=1)
#    conv9 = inception_block(up9, 32, batch_mode=2, splitted=splitted, activation=act)
#    #conv9 = inception_block(conv9, 32, batch_mode=2, splitted=splitted, activation=act)
#    conv9 = Dropout(0.5)(conv9)
#
#    conv10 = Convolution2D(1, (1, 1), init='he_normal', activation='sigmoid', name='main_output')(conv9)
#    #print conv10._keras_shape
#
#    model = Model(input=inputs, output=[conv10])
#    
#    adam = Adam(lr=1e-5)
#    model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef, jaccard_coef])
#
#    return model
#
#
#model = get_unet_inception_2head()

#datagen = ImageDataGenerator(rotation_range=0.2,
#                             width_shift_range=0.05,
#                             height_shift_range=0.05,
#                             shear_range=0.05,
#                             zoom_range=0.05,
#                             horizontal_flip=True,
#                             fill_mode='nearest')
#
#datagen.fit(trainData)
#
#
#def schedule(epoch):
#    if epoch<=5:
#        return 1e-5
#    elif epoch<=10:
#        return 5e-4
#    elif epoch<=25:
#        return 2e-4
#    elif epoch<=40:
#        return 1e-3
#    else:
#        return 5e-4
#
#lr_schedule= LearningRateScheduler(schedule)
#        
#model.fit_generator(datagen.flow(trainData, trainMask, batch_size=16), steps_per_epoch=500, epochs=50, verbose=1, callbacks=[lr_schedule])


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
    
x_train, y_train = Augmentation(trainData, trainMask)
x_validation, y_validation = AugmentationValidation(trainData, trainMask)

def schedule(epoch):
    if epoch<=20:
        return 1e-5
    elif epoch<=40:
        return 1e-4
    else:
        return 1e-5

lr_schedule= LearningRateScheduler(schedule)

model.fit(x_train, y_train,
          batch_size=16, nb_epoch=50,
          verbose=1, validation_data=(x_validation, y_validation), shuffle=True, callbacks=[lr_schedule])

model.save('unet_vanilla_final.h5')