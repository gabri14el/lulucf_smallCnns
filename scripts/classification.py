# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:31:31 2022

@author: Gabriel
"""

import rasterio
import math
import tensorflow as tf
import numpy as np
from  scipy import signal


#constants
dim = (5, 5, 13, 1)
WINDOWS_SIZE = dim[0]
WIDOWS_HALF_SIZE = int(WINDOWS_SIZE/2)
MODEL_PATH = r'C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague\mlruns\3\fc971352ac224908b474d3d41483523c\artifacts\Baseline3DExtendedExperiment_model__20220915-0913'
nclasses = 6
lines = 10
from tqdm import tqdm 

#select






#fp = r'C:\Users\Gabriel\Downloads\Mosaic20pxbuffer.tif'
fp = r'C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague\Raster.tif'
wp = r'C:\Users\Gabriel\Downloads\model_3D_suavized.tif'
img = rasterio.open(fp)
array = img.read()

print(img.meta)
#create array to put the result
result = np.zeros((2, array.shape[1], array.shape[2]))

max = 65535

batch = array.shape[1]
'''
def normalize(X):
    return X * (1./max)

'''
gfilter = [[1,2,1], [2,4,2], [1,2,1]]
def normalize(X):
    
    '''
    new_size = 7
    new_half=int(new_size/2)
    original_half = int(X.shape[0]/2)
    
    dim = (new_size, new_size)
    
    tf.image.resize(X, [new_size, new_size])
    resized = cv2.resize(X, dim, interpolation = cv2.INTER_LINEAR)
    resized = resized[(new_half-original_half):(new_half+original_half+1), (new_half-original_half):(new_half+original_half+1)]
    '''
    X = X[:,:,:,:,0]
    nx = np.copy(X).astype('float')
    for j in range(X.shape[0]):
        for i in range(X.shape[-1]):
            nx[j, :, :, i] = signal.convolve2d(X[j,:, :, i], gfilter, mode='same')
    nx=nx/16.0
    return np.expand_dims(nx, axis=-1)/(max+0.0)

steps = math.ceil(array.shape[1]/lines)
rest = False
if array.shape[1]%lines > 0:
    rest = True
print(array.shape[1]%lines)

with tf.device('/gpu:1'):
    nmodel = tf.keras.models.load_model(MODEL_PATH, compile=False)
#strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#with strategy.scope():

'''
_input = tf.keras.layers.Input(shape=(dim[0],dim[1],dim[2],dim[3],))
x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(_input)
x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
x = tf.keras.layers.BatchNormalization()(x)

#x = tf.keras.layers.GlobalAveragePooling3D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(40, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
output = tf.keras.layers.Dense(nclasses, activation='softmax')(x)
nmodel = tf.keras.models.Model(_input, output)
'''
#nmodel.load_weights(MODEL_PATH)
#iterate above the pixels
for s in tqdm(list(range(steps))):
    #the size of the array is the quantity of lines*size of one line
    array_size = lines*array.shape[2]
    start_line = s*lines
    final_line= start_line+lines

    
    #in case of rest of division
    if rest and s == (steps-1):
        print('entrou no final')
        array_size = (array.shape[1]%lines)*array.shape[2]
        final_line = start_line+(array.shape[1]%lines)
        print(array.shape[1]%lines)
        print(array_size, start_line, final_line)
    
    #create the array to the prediction
    to_pred = np.zeros((array_size, WINDOWS_SIZE, WINDOWS_SIZE, array.shape[0], 1))

    #prepare the window
    #verify if is possible to acquire the windows first
    for i, line in enumerate(range(start_line, final_line)):
            for j in range(array.shape[2]):
                window = np.zeros((WINDOWS_SIZE, WINDOWS_SIZE, array.shape[0]))
                if not ((line < WIDOWS_HALF_SIZE) or (j < WIDOWS_HALF_SIZE) or ((line + WIDOWS_HALF_SIZE) > (array.shape[1]-1)) or ((j + WIDOWS_HALF_SIZE) > (array.shape[2]-1))):
                    #fill the matriz in the correct format
                    for k in range(WINDOWS_SIZE):
                        for l in range(WINDOWS_SIZE):
                            for m in range(array.shape[0]):
                                index1 = line+k-WIDOWS_HALF_SIZE
                                index2 = j+l-WIDOWS_HALF_SIZE
                                window [k, l, m] = array[m, index1, index2]    
                #print(array.shape[1], i, j, array.shape[2])
                to_pred[array.shape[2]*i+j,:,:,:,0] = window
    to_pred=normalize(to_pred)
    pred = nmodel(to_pred)
    #pred = nmodel.predict(to_pred, batch_size=batch, use_multiprocessing=True)
    #save the result
    percent = np.max(pred.numpy(), axis=-1)*100.
    percent = percent.astype(np.int16)
    classes = np.argmax(pred.numpy(), axis=-1) + 1
    for _i, _line in enumerate(range(start_line, final_line)):
        result[0, _line, :] = classes[_i*array.shape[2]:(_i*array.shape[2])+array.shape[2]]
        result[1, _line, :] = percent[_i*array.shape[2]:(_i*array.shape[2])+array.shape[2]]
        if rest and s == (steps-1):
            print(_i, _line)

#backup the original array
backup = np.copy(result)
#set the values that was not possible to classify to 10            
result[:, 0:WIDOWS_HALF_SIZE,:] = 10
result[:, array.shape[1]-WIDOWS_HALF_SIZE:,:] = 10
result[:, :, 0:WIDOWS_HALF_SIZE] = 10
result[:, :, array.shape[2]-WIDOWS_HALF_SIZE:] = 10


# Get a copy of the source dataset's profile. Thus our
# destination dataset will have the same dimensions,
# number of bands, data type, and georeferencing as the
# source dataset.
kwds = img.profile
kwds['count'] = 2

with rasterio.open(wp, 'w', **kwds) as dst_dataset:
    dst_dataset.write(result.astype(rasterio.uint16))

                        