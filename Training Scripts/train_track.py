#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:36:14 2020

@author: Tom
"""

from model_track import Track_unet
from data import DataImport, TrackDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle


#Model path
model_file = '../scripts/tracknet.hdf5'

#Model parameters
unet_type = 'track'
target_size = (1024,1024)
input_size = (1024,1024) + (3,)
class_weights = (1,1,1)

#Training parameters
batch_size =8
epochs = 400
steps_per_epoch =200 

#Augmentation parameters
aug_keys = dict(horizontal_flip=True, 
                vertical_flip=True, rotations_360=True,
                elastic_deformation = True)

#Import data
imports = DataImport(unet_type = unet_type, target_size = target_size,sliding_window = False)

#Initiate data generator
train_data_gen  = TrackDataGenerator(imports, batch_size = batch_size, 
                                   aug_keys = aug_keys)

#Initiate U-NET
model = Track_unet(input_size = input_size, class_weights = class_weights)

#Define loss checkpoint
model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)

#Initate training
history = model.fit_generator(train_data_gen, steps_per_epoch=steps_per_epoch,
                    epochs=epochs, callbacks=[model_checkpoint])

with open('history.pkl','wb') as f:
    pickle.dump(history,f)
