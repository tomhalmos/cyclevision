#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:30:05 2020

@author: Tom
"""
from __future__ import print_function
import numpy as np 
import os
import random
import cv2
import copy
import skimage.io
import skimage
from skimage import morphology,segmentation
import scipy
from scipy import ndimage
from scipy import interpolate
import elasticdeform


def Import(filename, target_size=(1024,1024), binarise=False, order=1):
    '''
    Import() imports, equalises, crops, binarises
    and reshapes an image given a filename.
    '''
    # Import, equalalise and crop
    i = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    
    # Re-import with skimage for U2OS truth masks
    if i.size == 696*520:
        i = skimage.io.imread(filename)
    else:
        i = i[0:target_size[0], 0:target_size[1]]
        i = i/i.max() #Equalise to [0,1]
    
    # Binarise
    if binarise:
        i = Binarise(i, threshold=0.75)
        
    # Reshape to match U-Net input
    i = np.reshape(i,i.shape + (1,))
    
    return np.nan_to_num(i) # Remove any Nan elements
    
def Binarise(i,threshold=0.75):
    '''
    Binarise() binarises an image, i, about threshold 0.75
    '''
    binary = np.copy(np.zeros(i.shape))
    binary[i > threshold]  =  1
    binary[i <= threshold] =  0

    return binary

def ErrorMap(truth,prediction,p,target_size = (1024,1024)):
    '''
    ErrorMap() returns an error array for a given truth/prediction pair by 
    mixing the pixel-wise error with a uniform array. The degree of mixing is
    defined by the parameter 'p'
    '''
    truth = np.squeeze(truth)
    prediction = np.squeeze(prediction)
    uniform = np.ones(target_size)
    
    # Calculate error array
    error   = abs(truth-prediction)
    
    # Calculate error+uniform overlay 
    weight  = (p*uniform)+((1-p)*error)
    
    # Normalise and reshape
    weight  = weight / np.sum(weight)
    weight = np.reshape(weight,weight.shape + (1,))
    
    return weight

def Weightmap(i,tolerance):
    '''
    The Weightmap function generates the pixel-wise weight map of image i.
    Tolerance is a combination of window size w and threhsold n, and optimal
    somewhere in the range 0.6-0.9.
    '''
    if tolerance == 0:
        return np.ones(i.shape)
    
    # Get window size (w) and threshold (n) from the tolerance paramter
    w = int(30*tolerance)
    n = 1-tolerance/2
    w_area = 4*w**2
    
    # Define arrays for weightmap and feature scaling
    W = np.zeros(i.shape)
    X = np.zeros(i.shape)
    
    #Iterate over each pixel in i:
    for x in range(i.shape[0]):
        for y in range(i.shape[1]):
            #Flag nuclear pixels with a value of -1 for later identification
            if i[x,y] == 1:
                W[x,y] = -1
            else:
                if x > w and y > w: # Avoid indexing beyond image border
                    #Sum intensity in the local area
                    local_density = (np.sum(i[x-w : x+w,y-w : y+w]) / w_area)
                    #Assign local intensity to the pixel value
                    W[x,y] = local_density
                else:
                    W[x,y] = 0
                    
    Wmax = W.max()
    W[np.logical_and(W>0,W<n*Wmax)] = 0 # Set anything below the threshold n to 0
    W[W<0] = Wmax*0.25 # Assign small positive value to nuclear pixels
    Wmin  = W.min()
    a,b   = 1,10
    
    # Use min/max feature scaling to normalise the array
    for x in range(i.shape[0]):
        for y in range(i.shape[1]):
            pixel = W[x,y]
            #Use Min-Max feature scaling to normalise to [1,10]    
            X[x,y] = a + (((pixel-Wmin)*(b-a)) / (Wmax - Wmin))
    
    return X
   
def DataAugmentation(images_input, aug_keys):
    '''
    DataAugmentation() applies the data augmentations specified in aug_par
    to input images. It compiles a numpy array of batch_size images, as well as
    concatenating ground truth masks and weight maps into a single numpy array.  
    '''

    output = list(images_input)
    
    # Apply augmentation operations if key word in aug_par:  
    if "horizontal_flip" in aug_keys:
        if aug_keys["horizontal_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.fliplr(item)
                    
    if "vertical_flip" in aug_keys:
        if aug_keys["vertical_flip"]:
            if random.randint(0,1): #coin flip
                for index, item in enumerate(output):
                    output[index] = np.flipud(item)
                
    
    if "rotations_360" in aug_keys:
        if aug_keys["rotations_360"]:
            rot = random.randint(0,360)
            if rot>0:
                for index,item in enumerate(output):
                    output[index] = ndimage.rotate(item,rot,order=0,reshape=False)
    
    if "elastic_deformation" in aug_keys:
        if aug_keys['elastic_deformation']:
            if random.randint(0,1): #coin flip 
                   output = elasticdeform.deform_random_grid(output,
                                                      sigma=25,
                                                      points=3,
                                                      order=0,
                                                      axis=(0,1))
                   # Correct for large black patches in deformed raw image
                   img = output[0]
                   output[0] = np.where(img==0,np.mean(img),img)
               
    if "histogram_voodoo" in aug_keys:
        if aug_keys['histogram_voodoo']:
            if random.randint(0,1): #coin flip
                # Only transform the input image, not masks
                output[0] = histogram_voodoo(output[0])

            
    return output


def histogram_voodoo(image,num_control_points=3):
    '''
    This function kindly provided by Daniel Eaton from the Paulsson lab.
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination
    '''
    control_points = np.linspace(0,1,num=num_control_points+2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=0.1,high=0.9,size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)
    
    return mapping(image)


def DataImport(unet_type,target_size=(1024,1024)):
    '''
    DataImport() imports images, segmentation masks and weight maps from the training 
    dataset files.
    
    The training images/groundtruths... are all saved under the same name, but can be
    distinguished by their folders.
    '''
    
    if unet_type == 'seg':
        train_folders = ['img','truth']
    if unet_type == 'track':
        train_folders = ['seed','tracked','seg','daughter','prevseg']

    
    #Get list of training image names from the first training folder
    directory = '../data/' + unet_type + '/' + train_folders[0]
    filenames = [name for name in os.listdir(directory) if 'png' in name]
    
    # Create 'imports' to store training images/arrays under their type
    imports= {}
    for folder in train_folders:
        imports[folder] = []
        for filename in filenames:
            path = '../data/' + unet_type +'/'+ folder + '/' + filename
            imports[folder].append(Import(path,target_size = target_size))
        print(imports[folder][50].size)
    # For the segNet, create and append Weightmaps to imports
    if unet_type =='seg':
        imports['weight'] = []
        for truth in imports['truth']:
            imports['weight'].append(np.ones((truth.shape)))
            #imports['weight'].append(Weightmap(truth,0.75))
    
    # U2OS images need pairing and resizing into 1024x1024 images
    if unet_type == 'seg':
        # Get U2OS images based on size
        i_U2OS = [img for img in imports['img'] if img.size == 696*520]
        t_U2OS = [img for img in imports['truth'] if img.size == 696*520*4]
        print(len(i_U2OS))
        print(len(t_U2OS))
        # Process truth arrays into ground truth masks
        for j,truth in enumerate(t_U2OS):
            truth = truth[:,:,0]
            
            annot = skimage.morphology.label(truth)
            annot = skimage.morphology.remove_small_objects(annot, min_size=25)
            boundaries = skimage.segmentation.find_boundaries(annot)
            
            z = np.zeros(truth.shape)
            z[(annot != 0) & (boundaries ==0)] = 1

            t_U2OS[j] = truth
                    
        Di_U2OS, Dt_U2OS = [],[] #Empty lists for doubled images
        
        for i in range(0,len(i_U2OS),2):
            i_double = np.zeros((1040,1040))
            img1  = np.squeeze(i_U2OS[i])
            img2  = np.squeeze(i_U2OS[i+1])
            
            #Generate random image overlay indices
            r1 = np.random.randint(328) #1024-img.shape[1]
            r2 = np.random.randint(328) #1024-img.shape[1]
            i_double[0:520,r1:696+r1] = img1 
            i_double[520:,r2:696+r2]  = img2
            
            t_double = np.zeros((1040,1040),dtype=np.int64)
            mask1  = np.squeeze(t_U2OS[i])
            mask2  = np.squeeze(t_U2OS[i+1])
            
            t_double[0:520,r1:696+r1] = mask1 
            t_double[520:,r2:696+r2]  = mask2
            
            # Crop down to 1024,1024
            i_double = i_double[:1024,:1024]
            t_double = t_double[:1024,:1024]
            
            # Stop the two images touching and creating stange cell morphologies 
            i_double[515:525,:] = 0
            t_double[515:525,:] = 0
            
            # Remove any small objects from the mask
            t_double = skimage.morphology.label(t_double)
            t_double = skimage.morphology.remove_small_objects(t_double, min_size=50)
            t_double[t_double>0]=1
            
            # Reshape and store
            Di_U2OS.append(np.reshape(i_double,i_double.shape+(1,)))
            Dt_U2OS.append(np.reshape(t_double,t_double.shape+(1,)))
        
        # Add the doubled U2OS images to the PCNA ones
        i_PCNA = [img for img in imports['img'] if img.size == 1024*1024]
        t_PCNA = [img for img in imports['truth'] if img.size == 1024*1024]
        
        imports['img'] = i_PCNA + Di_U2OS
        imports['truth'] = t_PCNA + Dt_U2OS

    return imports


def Shuffle(imports,data_split=0.1, seed = 1):
    '''
    Shuffle() random splits the imported training data into training and validation
    training sets
    '''
    random.seed(a=seed)
    
    train_data = {}
    val_data   = {}
    
    for key in imports:
        train_data[key] = []
        val_data[key] = []
    
    set_size = len(imports[[*imports][0]]) #First key from either kind of import (seg or track)
    for i in range(set_size):
        j = np.random.binomial(1,data_split)
        if j==1:
            for key in imports:
                val_data[key].append(imports[key][i])
        elif j==0:
            for key in imports:
                train_data[key].append(imports[key][i])
    
    return train_data, val_data
    
def SegDataGenerator(train_import, batch_size=32,
                     aug_keys = {}):
    '''
    DataGenerator() compiles a numpy array of batch_size images, and concatenates 
    ground truths masks with weightmaps into a single array, and returns both
    in a tuple. 
    '''
    images = train_import['img']
    segmentation_masks = train_import['truth']
    weight_maps = train_import['weight']

    
    #Set up the generator of batch_size arrays of augmented training images
    while True:
        #Reset image arrays:
        img_arr = []
        seg_arr = []
        wei_arr = []
        for _ in range(batch_size):
            #Pick random image index:
            index = random.randrange(0,len(images))
            
            #Get from imported image lists
            img = images[index]
            seg = segmentation_masks[index]
            wei = weight_maps[index]
            
            #Use DataAugmentation() to augment the indexed images
            [img,seg,wei] = DataAugmentation([img,seg,wei], aug_keys)
            
            img_arr.append(img)
            seg_arr.append(seg)
            wei_arr.append(wei)
    
        #Convert all lists to numpy arrays, and concatenate masks and weightmaps
        img_arr = np.array(img_arr)
        seg_wei_arr = np.concatenate((seg_arr,wei_arr),axis=-1)
        yield (img_arr,seg_wei_arr)
        
def SegValidationPrep(val_import):
    '''
    ValidationPrep() prepares a validation dataset for network evaluation
    during training by concatenating masks and weight maps into a single array
    '''
    images = val_import['img']
    segmentation_masks = val_import['truth']
    weight_maps = val_import['weight']
    
    img_arr =np.array(images)
    seg_weight_arr = np.concatenate((segmentation_masks,weight_maps),axis=-1)
    return (img_arr,seg_weight_arr)

def EvaluationData(validation_data, target_size =(256,256),
                   min_area=25,aug_keys={}, seed = 1):
    '''
    EvaluationData() prepares an evaluation data set to use in trained network 
    evaluation
    '''
    images, segmentation_masks =  validation_data[0],validation_data[1]
    
    #Segmentation masks are filtered to remove small artefacts
    filtered_segmentation_masks = []    
    for mask in segmentation_masks:
        new_segmentation = np.zeros(target_size)
        #Count and label the objects in the segmentation mask
        labeled_segmentation = skimage.measure.label(np.squeeze(mask),return_num=True)[0] #[0] indexes labelled array
        #Iterate through object labels, outputting each object alone in a new mask
        for j in range(1,labeled_segmentation.max()+1):
                single_object = np.where(labeled_segmentation==j,1,0)
                #Apply minimum area threshold
                if np.sum(single_object) > min_area:
                    new_segmentation += single_object.reshape(target_size)
        filtered_segmentation_masks.append(np.squeeze(new_segmentation))
    
    #Augment the data for evaluation
    eval_images, eval_masks = [],[]
    
    for i in range(len(images)):
        for j in range(10):
            img, seg = images[i], filtered_segmentation_masks[i]
            aug_img,aug_seg = DataAugmentation([img,seg],aug_keys)
            eval_images.append(np.squeeze(aug_img))
            eval_masks.append(np.squeeze(aug_seg))
  

    return (eval_images, eval_masks)
    

def TrackDataGenerator(train_import, batch_size=32,
                     aug_keys = {}, rand_seed = 1):
    '''
    This function generates the training data for the tracking unet but with fewer inputs.
    It compiles a numpy array of batch_size x3 images;
    1) prevseg     = Segmentation of the previous image
    2) seed     = Segmentation of the cell to be tracked
    3) seg = Segmentation of the current image

    It also compiles three numpy arrays to train over, one mother output mask,
    one daughter output mask, and a third background mask.
    Images modified on the fly with the data_augmentation function.
    '''
    seeds = train_import['seed']
    segs = train_import['seg']
    tracked_cells = train_import['tracked']
    prevsegs  = train_import['prevseg']
    daughters = train_import['daughter']
    
    #Reset the pseudo-random generator:
    random.seed(a=rand_seed)

    #Set up the generator of batch_size arrays of augmented training images
    while True:
        #Reset image arrays:
        seed_arr = []
        seg_arr = []
        tracked_arr = []
        prevseg_arr  = []
        daughter_arr = []
           
        for _ in range(batch_size):
            #Pick random image index:
            index = random.randrange(0,len(seeds))
            
            #Get from imported image lists
            seed = seeds[index]
            seg = segs[index]
            tracked_cell = tracked_cells[index]
            prevseg = prevsegs[index]
            daughter = daughters[index]
                        
            #Use DataAugmentation() to augment the indexed images
            [seed,seg,tracked_cell,prevseg,daughter] = DataAugmentation(
                    [seed,seg,tracked_cell,prevseg,daughter], aug_keys)
            
            #Append to arrays
            seed_arr.append(seed)
            seg_arr.append(seg)
            tracked_arr.append(tracked_cell)
            prevseg_arr.append(prevseg)
            daughter_arr.append(daughter)  
        
        #Convert all lists to numpy arrays, and concatenate masks and weightmaps
        inputs_arr = np.concatenate((seed_arr,prevseg_arr,seg_arr),axis=-1)
        outputs_arr = np.concatenate((tracked_arr,daughter_arr),axis=-1)
        yield (inputs_arr,outputs_arr)
 
    
 

 
    
 
    
        
    
    
    