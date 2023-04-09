# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yvYfTteXukO4l6_kXn1h3o077ftpeba0
"""


import os

#EXPERIMENT CONFIGURATION
os.chdir(r'C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague')
dim = (5, 5, 13)
nclasses = 6
batch_size = 256
mlflow_activate = True
experiment_name = 'SyrrisExperiment'
path = r'C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague\data_5_v5_Lines_aug'
comments = ''

#SAVE PARAMS
import datetime
import utils

time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model_save_name = experiment_name+'_model_'+'_'+time
weight_save_name = experiment_name+'_weight_'+'_'+time+'.h5'
path_model = F"models\\{experiment_name}\\{model_save_name}"
path_weight = F"models\\{experiment_name}\\{weight_save_name}"

os.makedirs(F"models\\{experiment_name}", exist_ok=True)

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow as tf 
import os
import pandas as pd
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools

#IMPORT THE DATASET
columns = ['path', 'class', 'name', 'set']
df=pd.DataFrame(columns=columns)
#path = r'C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague\data2'


for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".txt"):
             dic = {}
             dic['path'] = os.path.join(root, file)
             dic['name'] = dic['path'].split(os.path.sep)[-1]
             dic['class'] = dic['path'].split(os.path.sep)[-2]
             dic['set'] = dic['path'].split(os.path.sep)[-3]
             df=df.append(dic, ignore_index=True)

print(df.head())

df['class'] = df['class'].astype(int)
df['class'] = df['class'].values - 1

df_train = df[df['set'] == 'train']
df_val = df[df['set'] == 'validation']
df_test = df[df['set'] == 'test']

train_ids = df_train.path.values
train_classes = {}

for i, row in df_train.iterrows():
    train_classes[row['path']] = int(row['class'])
    
test_ids = df_test.path.values
test_classes = {}

for i, row in df_test.iterrows():
    test_classes[row['path']] = int(row['class'])

val_ids = df_val.path.values
val_classes = {}

for i, row in df_val.iterrows():
    val_classes[row['path']] = int(row['class'])

#PRE-PROCESSING
#max value of uint16
max = 65535

def normalize(X):
  return X * (1./max)


#TRAINING
train_params = {'ids':train_ids, 'labels':train_classes, 'dim':dim, 'n_classes':nclasses, 'batch_size':batch_size}
train_gen = utils.TxtDataGenerator(**train_params)

test_params = {'ids':test_ids, 'labels':test_classes,
               'dim':dim, 'n_classes':nclasses, 'shuffle':False, 'batch_size':1}
test_gen = utils.TxtDataGenerator(**test_params)

val_params = {'ids':val_ids, 'labels':val_classes, 'dim':dim, 'n_classes':nclasses, 'batch_size':batch_size}
val_gen = utils.TxtDataGenerator(**val_params)    
    

# example of a 3-block vgg style architecture
_input = tf.keras.layers.Input(shape=(dim[0],dim[1],dim[2],))
x = tf.keras.layers.Conv2D(128, (2, 2), activation='tanh', kernel_initializer='he_uniform', strides=1)(_input)
x = tf.keras.layers.Conv2D(128, (2, 2), activation='tanh', kernel_initializer='he_uniform', strides=1) (x)
#x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.25)(x)

x = tf.keras.layers.Conv2D(512, (2, 2), activation='tanh', kernel_initializer='he_uniform', strides=1)(x)
x = tf.keras.layers.Conv2D(512, (2, 2), activation='tanh', kernel_initializer='he_uniform', strides=1) (x)
#x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.20)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='tanh')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.20)(x)
x = tf.keras.layers.Dense(128, activation='tanh')(x)
output = tf.keras.layers.Dense(nclasses, activation='softmax')(x)
nmodel = tf.keras.models.Model(_input, output)

import tensorflow_addons as tfa

nmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', utils.f1_m, utils.precision_m, utils.recall_m])
    
#callbacks
#set early sto
estop = tf.keras.callbacks.EarlyStopping(monitor='val_f1_m', patience=20, mode='max', restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_f1_m',
                                                 factor = 0.2,
                                                 patience = 2,
                                                 verbose = 1,
                                                 min_lr = 1e-8,
                                                 mode = 'max')

model_checkpoint_weights_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path_weight,
    save_weights_only=True,
    monitor='val_f1_m',
    mode='max',
    save_best_only=True)

model_checkpoint_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=path_model,
    save_weights_only=False,
    monitor='val_f1_m',
    mode='max',
    save_best_only=True)

callbacks_list = [model_checkpoint_weights_callback,
                  model_checkpoint_model_callback,
                  estop,
                  reduce_lr]

callbacks_list = [model_checkpoint_weights_callback,
                  model_checkpoint_model_callback,
                  estop, reduce_lr]

history = nmodel.fit(train_gen, steps_per_epoch=len(train_gen),
                               epochs=50, validation_data=val_gen,
                               validation_steps=len(val_gen), shuffle=True,
                               verbose=True, callbacks=callbacks_list)
   

#TESTING
#load the best weights
nmodel.load_weights(path_weight)
#test the model
report = utils.confusion_matrix(test_gen, nmodel)
evaluate = nmodel.evaluate(test_gen, return_dict=True)

if mlflow_activate:
    mlflow.set_experiment(experiment_name)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("dim", dim)
    mlflow.log_param("ds", path.split(os.path.sep)[-1])
    mlflow.log_param("optimizer", str(nmodel.optimizer).split()[0].split(".")[-1])
    mlflow.log_param("lr", nmodel.optimizer.lr.numpy())
    mlflow.log_artifact(os.getcwd()+os.path.sep+F"models\\{experiment_name}\\{model_save_name}")
    mlflow.log_text(report, os.getcwd()+os.path.sep+F"models\\{experiment_name}\\{model_save_name}_cm.txt")
    mlflow.log_metrics(evaluate)
    mlflow.log_figure(plt.gcf(), 'cm.png')
    mlflow.log_param("loss", str(nmodel.loss))
    mlflow.log_param("comments", comments if 'comments' in vars() else '')
    mlflow.log_param("dataset", path.split(os.path.sep)[-1])
    mlflow.end_run()