# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import metrics
import itertools
import numpy as np
import tensorflow as tf
import math

MAX = 65535
#DEFINITION OF F1 SCORE
import tensorflow.keras.backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def step_decay(epoch, initial_lrate=0.01):
    flattern_factor = initial_lrate ** 2.25
    epochs_drop = 5.0
    #drop modelado como modelado no artigo
    drop = initial_lrate **(flattern_factor/epochs_drop)
    
    lrate = initial_lrate * math.pow(drop,  
            math.floor((epoch)/epochs_drop))
    return lrate

def normalize_rgb_ln(X, preprocess=None):
    a = np.log(X)/np.log(65535.0)
    a = a * 255
    if not preprocess:
        return a.astype('uint8')
    return preprocess(a.astype('uint8'))

#DEFINITION OF TEST METHODS
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def confusion_matrix(test_data_generator, model, return_fig=False):
  #test_data_generator.reset()
  predictions = model.predict(test_data_generator, steps=test_data_generator.samples)
  #print(predictions)
  # Get most likely class
  predicted_classes = np.argmax(predictions, axis=1)
  #print(predicted_classes)
  #print(len(predicted_classes))
  true_classes = list(test_data_generator.labels.values())
  class_labels = [str(x) for x in np.unique(true_classes)]
  #print(class_labels)  
  #print(len(true_classes))
  report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4)
  cm = metrics.confusion_matrix(true_classes, predicted_classes)
  print(report)
  fig = plot_confusion_matrix(cm, class_labels)
  if return_fig:
      (report, fig)
  return report





class TxtDataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, ids, labels, batch_size=32, dim=(5, 5, 13),
                 n_channels=1, n_classes=10, shuffle=True, preprocessing=None, filter_bands=None):
        '''
        

        Parameters
        ----------
        ids : list
            paths od the files that must be loaded.
        labels : dict
            dict cotaining the class of each file. It should be in the format {'path':id}.
        batch_size : int, optional
            size of the batch. The default is 32.
        dim : tuple, optional
            tuple containing the shape of the data. The default is (5, 5, 13).
        n_channels : TYPE, optional
            DESCRIPTION. The default is 1.
        n_classes : int, optional
            number of the classes. The default is 10.
        shuffle : boolean, optional
            Flag that activate the shuffle of the data furing the generation. The default is True.

        Returns
        -------
        None.

        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.preprocessing = preprocessing
        self.samples = len(ids)
        self.classes = np.unique(list(labels.values()))
        self.filter_bands = filter_bands
    
    def on_epoch_end(self):
        '''
        This method is responsible for doing the shuffle of the data if it was
        seted True.

        Returns
        -------
        None.

        '''
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, ids_temp):
        
        #create the X and y
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        
        for i, id in enumerate(ids_temp):
            array = np.load(id)
            if self.filter_bands:
                array = array[:, :, self.filter_bands]
            if self.preprocessing:
                X[i, ] = self.preprocessing(array)
            else:
                X[i, ] = array
            y[i, ] = self.labels[id]
        
        #the classes must be integer values starting in 0
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
    
    
    def __getitem__(self, index):
        
        #generate indexes of the batch, cropping the array
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        
        ids_temp = [self.ids[k] for k in indexes]
        
        X, y = self.__data_generation(ids_temp)
        
        
        return X, y

class MultiInputGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, ids, labels, batch_size=32, dim=(5, 5, 13),
                 n_channels=1, n_classes=10, shuffle=True, preprocessing=None):
        '''
        

        Parameters
        ----------
        ids : list
            paths od the files that must be loaded.
        labels : dict
            dict cotaining the class of each file. It should be in the format {'path':id}.
        batch_size : int, optional
            size of the batch. The default is 32.
        dim : tuple, optional
            tuple containing the shape of the data. The default is (5, 5, 13).
        n_channels : TYPE, optional
            DESCRIPTION. The default is 1.
        n_classes : int, optional
            number of the classes. The default is 10.
        shuffle : boolean, optional
            Flag that activate the shuffle of the data furing the generation. The default is True.

        Returns
        -------
        None.

        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.preprocessing = preprocessing
        self.samples = len(ids)
        self.classes = np.unique(list(labels.values()))
    
    def on_epoch_end(self):
        '''
        This method is responsible for doing the shuffle of the data if it was
        seted True.

        Returns
        -------
        None.

        '''
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, ids_temp):
        
        #create the X and y
        X = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, self.dim[2]))
        y = np.empty((self.batch_size), dtype=int)
        
        for i, id in enumerate(ids_temp):
            if self.preprocessing:
                x = self.preprocessing(np.load(id))    
            else:
                x = np.load(id)
            X[i, ] = x
            X2[i, ] = x[int(self.dim[0]/2), int(self.dim[0]/2), :]
            y[i, ] = self.labels[id]
        
        X_set = [X, X2]
        #the classes must be integer values starting in 0
        return X_set, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
    
    
    def __getitem__(self, index):
        
        #generate indexes of the batch, cropping the array
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        
        ids_temp = [self.ids[k] for k in indexes]
        
        X, y = self.__data_generation(ids_temp)
        
        
        return X, y
    

class SpectralGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, ids, labels, batch_size=32, dim=(5, 5, 13),
                 n_channels=1, n_classes=10, shuffle=True, preprocessing=None):
        '''
        

        Parameters
        ----------
        ids : list
            paths od the files that must be loaded.
        labels : dict
            dict cotaining the class of each file. It should be in the format {'path':id}.
        batch_size : int, optional
            size of the batch. The default is 32.
        dim : tuple, optional
            tuple containing the shape of the data. The default is (5, 5, 13).
        n_channels : TYPE, optional
            DESCRIPTION. The default is 1.
        n_classes : int, optional
            number of the classes. The default is 10.
        shuffle : boolean, optional
            Flag that activate the shuffle of the data furing the generation. The default is True.

        Returns
        -------
        None.

        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.preprocessing = preprocessing
        self.samples = len(ids)
        self.classes = np.unique(list(labels.values()))
    
    def on_epoch_end(self):
        '''
        This method is responsible for doing the shuffle of the data if it was
        seted True.

        Returns
        -------
        None.

        '''
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, ids_temp):
        
        #create the X and y
        X = np.empty((self.batch_size, self.dim[-1]))
        
        y = np.empty((self.batch_size), dtype=int)
        
        for i, id in enumerate(ids_temp):
            if self.preprocessing:
                x = self.preprocessing(np.load(id))    
            else:
                x = np.load(id)
            X[i, ] = x[int(self.dim[0]/2), int(self.dim[0]/2), :]
            y[i, ] = self.labels[id]
        
        #the classes must be integer values starting in 0
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
    
    
    def __getitem__(self, index):
        
        #generate indexes of the batch, cropping the array
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        
        ids_temp = [self.ids[k] for k in indexes]
        
        X, y = self.__data_generation(ids_temp)
        
        
        return X, y    

class AEGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, ids, labels, batch_size=32, dim=(5, 5, 13),
                 n_channels=1, n_classes=10, shuffle=True, preprocessing=None):
        '''
        

        Parameters
        ----------
        ids : list
            paths od the files that must be loaded.
        labels : dict
            dict cotaining the class of each file. It should be in the format {'path':id}.
        batch_size : int, optional
            size of the batch. The default is 32.
        dim : tuple, optional
            tuple containing the shape of the data. The default is (5, 5, 13).
        n_channels : TYPE, optional
            DESCRIPTION. The default is 1.
        n_classes : int, optional
            number of the classes. The default is 10.
        shuffle : boolean, optional
            Flag that activate the shuffle of the data furing the generation. The default is True.

        Returns
        -------
        None.

        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.preprocessing = preprocessing
        self.samples = len(ids)
        self.classes = np.unique(list(labels.values()))
    
    def on_epoch_end(self):
        '''
        This method is responsible for doing the shuffle of the data if it was
        seted True.

        Returns
        -------
        None.

        '''
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, ids_temp):
        
        #create the X and y
        X = np.empty((self.batch_size, *self.dim))
        
        for i, id in enumerate(ids_temp):
            if self.preprocessing:
                X[i, ] = self.preprocessing(np.load(id))
            else:
                X[i, ] = np.load(id)
        
        #the classes must be integer values starting in 0
        return X, X
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
    
    
    def __getitem__(self, index):
        
        #generate indexes of the batch, cropping the array
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        
        ids_temp = [self.ids[k] for k in indexes]
        
        X, y = self.__data_generation(ids_temp)
        
        
        return X, y

class ContrastiveDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, values, labels, batch_size=32, dim=(5, 5, 13), shuffle=True, preprocessing=None, expand_dims=True):
        '''
        

        Parameters
        ----------
        ids : list
            paths od the files that must be loaded.
        labels : dict
            dict cotaining the class of each file. It should be in the format {'path':id}.
        batch_size : int, optional
            size of the batch. The default is 32.
        dim : tuple, optional
            tuple containing the shape of the data. The default is (5, 5, 13).
        n_channels : TYPE, optional
            DESCRIPTION. The default is 1.
        n_classes : int, optional
            number of the classes. The default is 10.
        shuffle : boolean, optional
            Flag that activate the shuffle of the data furing the generation. The default is True.

        Returns
        -------
        None.

        '''
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.ids = ids
        self.shuffle = shuffle
        self.on_epoch_end()
        self.preprocessing = preprocessing
        self.samples = len(ids)
        self.classes = np.unique(list(labels.values()))
        self.values = values
        self.expand_dims = expand_dims
    
    def on_epoch_end(self):
        '''
        This method is responsible for doing the shuffle of the data if it was
        seted True.

        Returns
        -------
        None.

        '''
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, ids_temp):
        
        #create the X and y
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=np.float32)
        
       
        for i, id in enumerate(ids_temp):
            if self.preprocessing:
                X1[i] = self.preprocessing(np.load(self.values[id][0]))
                X2[i] = self.preprocessing(np.load(self.values[id][1]))
            else:
                X1[i] = np.load(self.values[id][0])
                X2[i] = np.load(self.values[id][1])
        if self.expand_dims:
            X1 = np.expand_dims(X1, axis=-1)
            X2 = np.expand_dims(X2, axis=-1)

        y[i, ] = self.labels[id]

        #the classes must be integer values starting in 0
        return [X1, X2], y
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
    
    
    def __getitem__(self, index):
        
        #generate indexes of the batch, cropping the array
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        
        ids_temp = [self.ids[k] for k in indexes]
        
        X, y = self.__data_generation(ids_temp)
        
        
        return X, y


def class_balanced_categorical_crossentropy(y_true,
                             y_pred,
                             samples_per_cls,
                             batch_size,
                             from_logits=False,
                             beta = 0.9999,
                             label_smoothing=0.,
                             axis=-1,
                             cce = tf.keras.losses.CategoricalCrossentropy()):
    
    
    #### code referent to get the weights for the function
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(samples_per_cls)
    
    #expand dims
    weights = tf.constant(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights, 0)
    
    #y is already in one-hot representation
    
    #create a vector with 
    weights = tf.tile(weights, [batch_size, 1]) * y_true
    weights = tf.reduce_sum(weights, axis=1)
    weights = tf.expand_dims(weights, axis=1)
    #weights = tf.tile(weights, [1, len(samples_per_cls)])
    ###

        
    return cce(y_true, y_pred, sample_weight=weights)