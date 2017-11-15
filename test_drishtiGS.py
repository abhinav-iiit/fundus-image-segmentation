import glob
import scipy.ndimage
import numpy as np
import cv2
from scipy.misc import imread, imresize

from keras.preprocessing.image import array_to_img, img_to_array
from keras.models import Model, load_model 
from keras.layers import concatenate, Dropout, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Input,  merge,  UpSampling2D, core

def get_unet(img_rows, img_cols):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    conv1 = Dropout(0.3)(conv1)
    pool1 =  MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 =  MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    pool3 =  MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv3)

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    pool4 =  MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv4)

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
   
    up6 = UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv5)
    up6 = concatenate([conv4, up6], axis=1)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)

    up7 = UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv6)
    up7 = concatenate([conv3, up7], axis=1)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv7)

    up8 = UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv7)
    up8 = concatenate([conv2, up8], axis=1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv8)

    up9 = UpSampling2D(size=(2, 2), data_format = 'channels_first')(conv8)
    up9 = concatenate([conv1, up9], axis=1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv9 = Dropout(0.3)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', data_format='channels_first')(conv9)
    model = Model(input=inputs, output=conv10)
    model.summary()
    return model

model = get_unet(128, 128)

images = (glob.glob("./input/*.png"))

model.load_weights('last_checkpoint.hdf5')

image_list = []
for i in images:
    img = imread(i)
    img = img.swapaxes(0,2)
    image_list.append(img)
     
    img1 = img.swapaxes(0,2)
    img1 = array_to_img(img1)
    #img1.show()
    
img_pred = model.predict(np.asarray(image_list), batch_size=1, verbose=1)

for i in range(51):
    img = img_pred[i]
    img = img.swapaxes(0,2)
    img = array_to_img(img) 
    #img.show()


