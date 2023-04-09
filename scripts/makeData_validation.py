# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:15:13 2022

@author: Gabriel
"""

import pandas as pd
import numpy as np

direcotory_to_save = r'C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague\data_65'
df = pd.read_csv(r'C:\Users\Gabriel\Downloads\ValidTable.txt')

#factor of the data sampling
factor = 10 
size_window = 65 * (factor)
size = int(size_window/2)

#data exploration
import numpy as np

print('dtypes: ', df.dtypes)

columns = df.columns

for c in columns[6:]:
  print('min_{0}, {1}, max_{0}, {2}'.format(c,np.min(df[c].values), np.max(df[c].values)))
  

#removing points 
df_filtred = df[df['b13_Mozaika_12'] > 0]
print(np.unique(df_filtred['Xrel']))

validation_points = df_filtred[df_filtred['Xrel'] == 0]
validation_points = validation_points [validation_points['Yrel'] == 0]

print(validation_points)

val_percent = 0.5
test_percent = 0.5

val_set = {}
test_set = {}

polygons = {}
def add_to_dict(tup, dict_polygons):
    if not int(tup[0]) in dict_polygons:
        dict_polygons[int(tup[0])] = []
    dict_polygons[int(tup[0])].append(tup[1])

[add_to_dict(x, polygons) for x in list(validation_points.groupby(['LULUCF_1','RectangleID']).groups.keys())]

print(polygons)

polygons_size = {}

for x in list(polygons.keys()):
  polygons_size[x] = len(polygons[x])

print(polygons_size)

import os
import random
import copy

polygons_copy = copy.deepcopy(polygons)

for x in list(polygons.keys()):
    #take the percentage
    test_quantity = round(test_percent*polygons_size[x])
    val_quantity = round(val_percent*polygons_size[x])
  
    test_set[x] = []
  
    
    #create the test set
    for i in range(test_quantity):
        indice = round(random.random()*(len(polygons_copy[x])-1))
        test_set[x].append(polygons_copy[x][indice])
        del polygons_copy[x][indice]
    
    #copy the remainder elements to the val set
    val_set[x] = polygons_copy[x]

[print(x, ': ', len(test_set[x])) for x in test_set.keys()]
print('-------')
[print(x, ': ', len(val_set[x])) for x in test_set.keys()]

mapping = {}

for x in list(test_set.values()):
    for y in x:
        mapping[y] = 'test'

for x in list(val_set.values()):
    for y in x:
        mapping[y] = 'validation' 

validation_points['dataset'] = validation_points['RectangleID'].map(mapping)

for i, row in validation_points.iterrows():
    
    image = np.zeros((int(size_window/factor), int(size_window/factor), 13))
    #take the coordinates of the point and the class
    x = int(row['Xrel'])
    y = int(row['Yrel'])
    name = '('+str(x)+','+str(y)+')'
    
    #take the class
    class_landcover = int(row['LULUCF_1'])
    polygon_nb = int(row['RectangleID'])
    
    subset = row['dataset']

    
    #take points of polygon
    df_aux = df_filtred[df_filtred['RectangleID'] == row['RectangleID']]
  
    #crop the interested lines
    df_aux = df_aux[df_aux['Xrel'] >= x-size]
    df_aux = df_aux[df_aux['Xrel'] <= x+size]
    df_aux = df_aux[df_aux['Yrel'] >= y-size]
    df_aux = df_aux[df_aux['Yrel'] <= y+size]
  
  
    #take the min of each coordinate
    min_x = np.min(df_aux['Xrel'])
    min_y = np.min(df_aux['Yrel'])
  
    for k, row2 in df_aux.iterrows():
      xrel = int(((row2['Xrel'])-min_x)/factor)
      yrel = int(((row2['Yrel'])-min_y)/factor)
      image[xrel, yrel] = row2.values[6:19]
    
    os.makedirs(os.path.join(direcotory_to_save, subset, str(class_landcover)), exist_ok=True)
    
    
    with open(os.path.join(direcotory_to_save, subset, str(class_landcover), str(polygon_nb)+'_'+str(int(i))+'_'+name+'.txt'), 'wb') as f:
      np.save(f, image)
      f.close()
    