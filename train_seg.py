#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:32:45 2020

@author: Tom
"""
from model_seg import Seg_unet
from data import DataImport, SegDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


#U-Net Type
unet_type = 'seg'
#Model path
model_file = '../scripts/segnet.{epoch:02d}.hdf5'

#Model parameters
target_size = (1024,1024)
input_size = target_size + (1,)
epochs = 100

#Train parameters:
batch_size = 10
steps_per_epoch = 200

#Augmentation parameters
aug_keys = {'horizontal_flip': False,
 'vertical_flip': False,
 'rotations_360': False,
 'elastic_deformation': False,
 'histogram_voodoo': False}

#Initiate U-NET
model = Seg_unet(input_size = input_size)

#Import training data
train_import = DataImport(unet_type = unet_type, target_size = target_size)


#Train data generator needs to re-initialised
train_data_gen  = SegDataGenerator(train_import, batch_size = batch_size, 
                                   aug_keys = aug_keys)


model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=False)


model.fit_generator(train_data_gen,steps_per_epoch=steps_per_epoch,epochs=epochs,
                    callbacks=[model_checkpoint])


