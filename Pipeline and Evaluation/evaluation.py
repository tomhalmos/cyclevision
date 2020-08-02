#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:58:31 2020

@author: Tom
"""

from model import Define_unet
from data import Binarise
import matplotlib.pyplot as plt
import numpy as np
import skimage, cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import morphology


'''
The following functions are used in the evluation of a given unet. 
See example below

validation_data = DataImport(target_size=(256,256),datatype='validation/')

aug_keys = dict(horizontal_flip=True,
                vertical_flip=True,
                rotations_90d=True)

images, groundtruths = EvaluationData(validation_data,aug_keys = aug_keys)

unet = LoadUnet('02-03-20--256--v2')
predictions = BatchPredict(images,unet)
groundtruth_maps, prediction_maps = Batch_NucleiMap(predictions,groundtruths)
precision, recall = Batch_ObjectLevelPR(groundtruth_maps, prediction_maps)
precision, recall = Batch_PixelLevelPR(predictions, groundtruths, groundtruth_maps)
'''


def LoadUnet(unet_ID,unet_type,weight_ID,
             directory = '/Users/Tom/Documents/trained_networks', 
             target_size = (1024,1024)):
    '''
    LoadUnet() compiles and adds pre-trained weights to a U-Net.
    '''
    weight_directory = directory + '/' + unet_ID
    if unet_type == 'seg':
        input_size = target_size + (1,)
    elif unet_type == 'track':
        input_size = target_size + (4,)

    unet = Define_unet(input_size=input_size)
    unet.load_weights(weight_directory + '/' + weight_ID + '.hdf5')
    
    return unet

def BatchPredict(images,unet,img_size,Filter=False,Binary=True,BinThresh=0.7):
    '''
    BatchPredict() returns predictions of a batch of images
    '''
    predictions = []
    
    for img in images:
        #Reshape image to match U-Net input size
        img = img.reshape(1,img.shape[0],img.shape[1],1)
        
        #Predict, binarise and remove added dimensions to img
        prediction = np.squeeze(unet.predict(img))
        
        #Binarise, if necessary
        if Binary == True:
            prediction = Binarise(prediction,BinThresh=BinThresh)
    
        if Filter == True:
            prediction = skimage.morphology.remove_small_objects(prediction, min_size=25)
        
        predictions.append(prediction)
            
    return predictions

def VisualisePrediction(index,predictions,groundtruths):
    '''
    VisualisePrediction() plots a prediction against a ground truth mask.
    '''
    fig, axs = plt.subplots(1,2, figsize=(10,3))
    axs[0].imshow(np.squeeze(groundtruths[index]))
    axs[0].set_title('GroundTruth')
    axs[1].imshow(np.squeeze(predictions[index]))
    axs[1].set_title('Prediction')
    plt.show()

def NucleiMap(prediction,groundtruth):
    '''
    NucleiMap() returns two mapping dictionaries which describe to which predicted nuclei
    each ground truth nuclei maps to, and visa versa in a second dictionary.
    '''
    prediction = np.squeeze(prediction)
    groundtruth = np.squeeze(groundtruth)
    
    #For each nuclei in the groundtruth and predicted images respectively,
    #extract the nuclei, and store it in its respective dictionary
    #under its skimage.measure.label label key 
    groundtruth_nuclei = {}
    predicted_nuclei  = {}
    
    #[0] index in skimage.measure.label pulls the labelled image array only
    labeled_groundtruth = skimage.measure.label(groundtruth,return_num=True)[0]
    labeled_prediction = skimage.measure.label(prediction,return_num=True)[0]
    
    #Using 'j' in the np.where assignment retains the label information in the resulting extraction
    for j in range(1,labeled_groundtruth.max()+1):
        groundtruth_nuclei[j] = np.where(labeled_groundtruth==j,j,0)
    for j in range(1,labeled_prediction.max()+1):
        predicted_nuclei[j] = np.where(labeled_prediction==j,j,0)
   
    groundtruth_map = {}
    prediction_map = {}
    #Each separate groundtruth nuclei is multiplied by the entire labeled prediction
    #THe unique integers found in the resulting product array refer to which predicted nuclei
    #overlap with the single ground truth nucleus. The groundtruth nucleus has its intensity set to 1
    #to not corrupt the labels of the prediction upon multiplication. 
    for label in groundtruth_nuclei:
        product = (groundtruth_nuclei[label]/label)*labeled_prediction
        overlaps = np.unique(product)
        #As no nulceus fills the entire image, 0 is always present in the product array 
        #thus if len(overlaps) == 1, the nucleus in question does not map to any predicted nuclei. 
        #Otherwise, the integers stored in the overlap list are assigned as the labels of each
        #predicted nuclei to which the groundtruth nuclei in question maps to
        if len(overlaps) == 1:
            groundtruth_map[label] = [0]
        else:
            groundtruth_map[label] = list(overlaps.astype(int)[1:]) #[1:] excludes the 0 label.
    
    #The same but reverse is then applied to generate the predicted nuclei mapping dictionary
    for label in predicted_nuclei:
        product = (predicted_nuclei[label]/label)*labeled_groundtruth
        overlaps = np.unique(product)
        if len(overlaps) == 1:
            prediction_map[label] = [0]
        else:
            prediction_map[label] = list(overlaps.astype(int)[1:])
            
    return groundtruth_map, prediction_map

def Batch_NucleiMap(predictions,groundtruths):
    '''
    Batch_NucleiMap() runs NucleiMap for a batch of prediction/groundtruth pairs
    '''
    groundtruth_maps = []
    prediction_maps  = []
    
    for i in range(len(predictions)):
        groundtruth_map, prediction_map = NucleiMap(predictions[i],groundtruths[i])         
        groundtruth_maps.append(groundtruth_map)
        prediction_maps.append(prediction_map)
    
    return groundtruth_maps, prediction_maps

def ObjectLevelPR(groundtruth_map,prediction_map):
    '''
    ObjectLevelPR() returns the false/true positive/negatives on a single prediction/groundtruth pair
    at the nuclear object level. S/C/A/M refer to split, create, absent and merge. 
    '''
    false_positives_S, false_positives_C = 0,0
    false_negatives_A,false_negatives_M = 0,0
    
    # *False Negative Absent Type*
    #Use the groundtruth map to look for groundtruth nuclei that do not map
    #to any predicted nuclei (i.e not segmented)
    for label in groundtruth_map:
        if groundtruth_map[label] == [0]:
            false_negatives_A += 1
            
    # *False Negative Merge Type*
    #Use the prediction_map to look for predicted nuclei that map to more than one 
    #ground truth nucleus (i.e groundtruth nuclei merged)
    # 1 groundtruth nuclei merged to y predicted nuclei equates y-1 false negatives
    for label in prediction_map:
        len_map_list = len(prediction_map[label])
        if len_map_list > 1:
            false_negatives_M += (len_map_list - 1)
    
    # *False Positive Split Type*
    #Use the groundtruth map to look for groundtruth nuclei that map to multiple predicted nuclei
    # 1 groundtruth nuclei split to y predicted nuclei equates y-1 false positives
    for label in groundtruth_map:
        len_map_list = len(groundtruth_map[label])
        if len_map_list > 1:
            false_positives_S += (len_map_list - 1)
     
    # *False Positive Create Type*
    #Use prediction map to look for predicted nuclei that do not map to any groundtruth nuclei
    for label in prediction_map:
        if prediction_map[label] == [0]:
            false_positives_C += 1
    
    #Generalise false positive/negatives
    false_negatives = false_negatives_A + false_negatives_M
    false_positives = false_positives_S + false_positives_C
    
    #True positives are then calculated by subtracting both false positive types from the number
    # of cells in the predicted image
    true_positives = len(prediction_map) - false_positives
    
    #Dictionary to store each type clearly
    object_error_dictionary = {}
    
    object_error_dictionary['true_positives']   = true_positives
    object_error_dictionary['false_positives']  = false_positives
    object_error_dictionary['false_negatives']  = false_negatives
    object_error_dictionary['false_positives_S']= false_positives_S
    object_error_dictionary['false_positives_C']= false_positives_C
    object_error_dictionary['false_negatives_A']= false_negatives_A
    object_error_dictionary['false_negatives_M']= false_negatives_M
    object_error_dictionary['total true cells'] = true_positives + false_negatives

    return object_error_dictionary
    
def Batch_ObjectLevelPR(groundtruth_maps,prediction_maps,false_types = False):
    '''
    Batch_ObjectLevelPR() returns precision and recall for a batch of prediction/groundtruth pairs
    If false_types == True, specific false negative/positive types are returned.
    '''
    true_positives = 0
    false_positives,  false_negatives  = 0,0
    false_positives_S,false_positives_C= 0,0
    false_negatives_A,false_negatives_M= 0,0
    
    for i in range(len(groundtruth_maps)):
        object_error_dictionary = ObjectLevelPR(groundtruth_maps[i],prediction_maps[i])
        
        true_positives  += object_error_dictionary['true_positives']
        false_positives += object_error_dictionary['false_positives']
        false_negatives += object_error_dictionary['false_negatives']
        
        false_positives_S += object_error_dictionary['false_positives_S']
        false_positives_C += object_error_dictionary['false_positives_C']
        false_negatives_A += object_error_dictionary['false_negatives_A']
        false_negatives_M += object_error_dictionary['false_negatives_M']
        
        total_true = object_error_dictionary['total true cells']
    
    precision = (true_positives / (true_positives + false_positives+1)*100) 
    recall    = (true_positives / (true_positives + false_negatives+1)*100) 
    
    false_types = [false_positives_S,false_positives_C,false_negatives_A,false_negatives_M,total_true]
    
    
    if false_types:
        return (precision, recall, false_types)
    else:
        return precision, recall

def PixelLevelPR(groundtruth_label,predicted_label,
                 prediction,groundtruth, groundtruth_map, 
                 target_size=(256,256)):
    '''
    PixelLevelPR() returns the number of true/false positive/negative pixels between a single 
    groundtruth  nucleus and its corresponding predicted nucleus
    '''    
    #Label the input images
    labeled_prediction  = skimage.measure.label(prediction,return_num=True)[0]
    labeled_groundtruth  = skimage.measure.label(groundtruth,return_num=True)[0]
    
    #Extract the nucleus from each image using its label
    predicted_nucleus   = np.where(labeled_prediction==predicted_label,1,0)
    groundtruth_nucleus = np.where(labeled_groundtruth==groundtruth_label,1,0)
    
    #Multiply the two nuclei to count true positive pixels
    true_positives = np.sum(predicted_nucleus*groundtruth_nucleus)
    
    #Subtract the predicted nucleus from the groundtruth to count the 
    #false positive/negative pixels.
    subtraction = groundtruth_nucleus - predicted_nucleus
    false_negatives = np.sum(subtraction==1) #Groundtruth where no prediction
    false_positives = np.sum(subtraction==-1)#PRediction where no groundtruth
    
    return true_positives, false_positives, false_negatives
    
def Batch_PixelLevelPR(predictions,groundtruths,groundtruth_maps):
    '''
    Batch_PixelLevelPR returns precision and recall calculated at the pixel level
    from a batch of images, only on true positive cells
    '''  
    true_positives, false_positives,  false_negatives  = 0,0,0
    
    for i in range(len(groundtruths)):
        groundtruth_map = groundtruth_maps[i]
        prediction = predictions[i]
        groundtruth = groundtruths[i]
        
        #A crude method is used to identify true positive cells
        labels = []
        for label in groundtruth_map:
            labels.append(groundtruth_map[label])
        labels = [item for sublist in labels for item in sublist] #flattens list of list to list
        
        for label in groundtruth_map:
            #True positive cells map to a single predicted nuclei, whose label does not appear
            #in the label lists of any other groundtruth nuclei
            if len(groundtruth_map[label]) == 1 and labels.count(groundtruth_map[label]) == 1:
                pixel_err = PixelLevelPR(label, groundtruth_map[label],prediction,
                                              groundtruth,groundtruth_map)
                true_positives  += pixel_err[0]
                false_positives += pixel_err[1]
                false_negatives += pixel_err[2]
            else:
                pass

    precision = (true_positives / (true_positives + false_positives+1)*100) 
    recall    = (true_positives / (true_positives + false_negatives+1)*100) 
    
    return precision,recall
    
def Watershed(img):
    from skimage.segmentation import watershed
    '''
    Watershed() returns the watershed segmentation on a given img.
    '''  
    #Image processing and thresholding
    imgA = (img/img.max()) * 255
    imgB = np.array(imgA,dtype=np.uint8)
    imgC = cv2.adaptiveThreshold(imgB, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, -5)
    #Watershed
    distance = ndi.distance_transform_edt(imgC)
    localMax = peak_local_max(distance, indices=False, footprint=np.ones((3,3)),labels=imgC)
    markers = ndi.label(localMax)[0]
    labels = watershed(-distance,markers,mask=imgC)
    #Erode and dilate
    labelsU8  = np.array(labels, dtype=np.uint8)
    kernel = np.ones((5,5),np.uint8)
    eroded = cv2.erode(labelsU8, kernel,iterations = 1)
    dilation = cv2.dilate(eroded,kernel,iterations = 1)
    #Final equalisation
    dilation[dilation>0] = 1
    watershed = dilation
    
    return watershed
    
def Batch_Watershed(images): 
    '''
    Batch_Watershed() returns the watersheds from a batch of images
    ''' 
    watersheds = []
    
    for img in images:
        watershed = Watershed(np.squeeze(img))
        watersheds.append(watershed)
    
    return watersheds
    
    
    
    
    
    
