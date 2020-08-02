#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:37:06 2020

@author: Tom
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam


def pixelwise_weighted_binary_crossentropy(y_true, y_pred):
    '''
    This function calculates the pixel-wise weighted, binary cross-entropy value
    between the prediction (y_pred) and the pixel-wise weight map, which is unstacked
    from y_true. 
    On the Gauss ssh server, tf.log must be written as tf.math.log
    '''
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = (y_pred >= zeros)
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(relu_logits - y_pred*seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None)
    
    # This is essentially the only part that is different from the Keras code:
    return K.mean(math_ops.multiply(weight, entropy), axis=-1)
    
def unstack_acc(y_true, y_pred):
    ''' 
    This function unstacks the ground truth mask from the y_true tensor and calculates 
    binary accuracy against a prediction
    '''
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    return keras.metrics.binary_accuracy(seg,y_pred)

def class_weighted_categorical_crossentropy(class_weights):
    '''
    This function returns a loss function. And that loss function is a 
    class-weighted categorical cross-entropy, i.e. the loss associated with 
    each class/category is weighted by weights in the class_weights list.
    '''
    def loss_function(y_true, y_pred):
        
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, -1, True)
        # manual computation of crossentropy
        epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Multiply each class by its weight:
        classes_list = tf.unstack(y_true * tf.math.log(y_pred), axis=-1)
        for i in range(len(classes_list)):
            classes_list[i] = tf.scalar_mul(class_weights[i], classes_list[i])
        
        # Return weighted sum:
        return - tf.reduce_sum(tf.stack(classes_list, axis=-1), -1)
    
    return loss_function

class CustomSaver(keras.callbacks.Callback):
    """Define custom saver to save the model at the end of every epoch, with epoch number in name"""
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 1:  # or save after some epoch, each k-th epoch etc.
            self.model.save("model_{}.hd5".format(epoch))# Generic unet declaration:
            
def Define_unet(input_size = (1024,1024,1), final_activation = 'sigmoid', output_classes = 1):
    '''
    Declares the U-Net model. Input size can be varied in new_main.py.
    Generic U-Net declaration. Only the input size, final activation layer and 
    dimensionality of the output vary between the different uses.
    '''
    
    inputs = Input(input_size,  name='true_input')	
	
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = Dropout(0.5)(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
    merge7 = Concatenate(axis = 3)([conv5,up7]) 
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis = 3)([conv4,up8])
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis = 3)([conv3,up9])
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up10 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    merge10 = Concatenate(axis = 3)([conv2,up10])
    conv10 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)

    up11 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
    merge11 = Concatenate(axis = 3)([conv1,up11])
    conv11 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11)
    conv11 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv12 = Conv2D(output_classes, 1, activation = final_activation, name = 'true_output')(conv11)
    
    model = Model(inputs = inputs, outputs = conv12)
    
    return model


# Use the following model for segmentation:
def Seg_unet(epochs, pretrained_weights = None, input_size = (1024,1024,1)):
    
    model = Define_unet(input_size = input_size, final_activation = 'sigmoid', output_classes = 1)
    model.compile(optimizer = Adam(lr = 1e-4), loss = pixelwise_weighted_binary_crossentropy, metrics = [unstack_acc])

    return model

# Use the following model for tracking and lineage reconstruction:
def Track_unet(pretrained_weights = None, input_size = (1024,1024,3),class_weights=(1,1,1)):
    
    model = Define_unet(input_size = input_size, final_activation = 'softmax', output_classes = 3)
    model.compile(optimizer=Adam(lr = 1e-4), loss=class_weighted_categorical_crossentropy(class_weights), metrics = ['categorical_accuracy'])

    return model


