# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:15:13 2022

@author: Gabriel
"""

import pandas as pd
import numpy as np

direcotory_to_save = r"C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague\data_32_lines_v2"
df = pd.read_csv(r"C:\Users\Gabriel\OneDrive - Universidade de Tras-os-Montes e Alto Douro\UTAD\Prague\csv_data\JuneLastWeek\TrainingLines.txt")

#factor of the data sampling
factor = 10 
size_window = 31 * (factor)
size = int(size_window/2)

#data exploration
import numpy as np

print('dtypes: ', df.dtypes)

columns = df.columns

for c in columns[6:]:
  print('min_{0}, {1}, max_{0}, {2}'.format(c,np.min(df[c].values), np.max(df[c].values)))
  

#removing points 
df_filtred = df[df['Training'] == 1]
print(np.unique(df_filtred['Xrel']))



import os
import random
import copy



for i, row in df_filtred.iterrows():
    
    image = np.zeros((int(size_window/factor), int(size_window/factor), 13))
    #take the coordinates of the point and the class
    x = int(row['Xrel'])
    y = int(row['Yrel'])
    name = '('+str(x)+','+str(y)+')_L'
    
    #take the class
    class_landcover = int(row['LULUCF'])
    polygon_nb = int(row['Id'])
    
    subset = 'train'

    
    #take points of polygon
    df_aux = df[df['Id'] == row['Id']]
  
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
      image[xrel, yrel] = row2.values[6:]
    
    os.makedirs(os.path.join(direcotory_to_save, subset, str(class_landcover)), exist_ok=True)
    
    #print(image[:,:,5])
    
    with open(os.path.join(direcotory_to_save, subset, str(class_landcover), str(polygon_nb)+'_'+str(int(i))+'_'+name+'.txt'), 'wb') as f:
      np.save(f, image)
      f.close()
    