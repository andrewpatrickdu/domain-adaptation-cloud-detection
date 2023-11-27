#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that evaluates a cloud detector.

python evaluate.py \
    --MODEL cloudscout-128a-S2-2018 \
    --DATASET L9-2023 \
    --NUM_BANDS 3 \
    --GPU 0 
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

from dataset import setup_dataset

from cloudscout import CloudScout
from cloudscout import CloudScout8
from resnet import ResNet18
from resnet import ResNet34
from resnet import ResNet50
from resnet import ResNet101
from resnet import ResNet152

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
# os.getcwd()

# set random seeds for reproducibility
seed = 0
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
torch.cuda.manual_seed(seed) 
np.random.seed(0) 
random.seed(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='DUA')
    parser.add_argument('--MODEL',
                        help='model folder name',
                        type=str,
                        default='')
    parser.add_argument('--NUM_BANDS',
                        help='number of bands is either 3 or 8',
                        type=int,
                        default=3)
    parser.add_argument('--DATASET',
                        help='dataset used to evaluate model (either S2-2018 or L9-2023)',
                        type=str,
                        default='')
    parser.add_argument('--ROOT',
                        help='root directory',
                        type=str,
                        default='/domain-adaptation-cloud-detection')
    parser.add_argument('--GPU',
                        help='gpu to run on',
                        type=int,
                        default=0)

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    print("args=%s" % args)
    
    # select gpu to run on
    torch.cuda.set_device(args.GPU)
    
    # folder directory
    fdr_dir = args.ROOT

    ###############################################################################
    ################## DEFINE DATASET DIRECTORY AND LOAD LABELS ###################
    ###############################################################################
    
    if args.DATASET == "S2-2018":
        img_dir = '/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
        lab_dir = '/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/labels/TF70.csv'
    elif args.DATASET == "L9-2023":
        img_dir = '/datasets/Landsat-9-Level-1/preprocessed/numpy/images'
        lab_dir = '/datasets/Landsat-9-Level-1/preprocessed/labels/TF70.csv'
    
    # load labels
    labels = pd.read_csv(fdr_dir + lab_dir)
    print('\nTOTAL TF70\n', labels['is_cloudy'].value_counts())
    
    ###############################################################################
    ################################ SPLIT DATASET ################################
    ###############################################################################
    
    labels = labels.sample(frac=1, random_state=0)
    N = min(labels['is_cloudy'].value_counts()[0],
            labels['is_cloudy'].value_counts()[1])
    cloudy = labels.loc[labels['is_cloudy'] == 1]
    not_cloudy = labels.loc[labels['is_cloudy'] == 0]
    
    training = pd.concat([cloudy[0:int(0.70*N)], not_cloudy[0:int(0.70*N)]])
    test = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]])
    
    # why shuffle again?
    training = training.sample(frac=1, random_state=0)
    test = test.sample(frac=1, random_state=0)
    print('\nTF70 TRAINING\n', training['is_cloudy'].value_counts())
    print('\nTF70 TEST\n', test['is_cloudy'].value_counts())

    ###############################################################################
    ############################# PRE-PROCESS DATASET #############################
    ###############################################################################

    # define transformations 
    train_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                         # AddGaussianNoise(0., 1.),
                                         # transforms.ToPILImage(),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         # transforms.ToTensor(),
                                         # transforms.RandomAdjustSharpness(sharpness_factor, p=0.5),
                                         # transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
                                         # transforms.Resize((356, 356)),
                                         # transforms.RandomCrop((299, 299)),
                                         ])
    
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                        ])
    
    # set up the datasets
    train_data = setup_dataset(training, fdr_dir + img_dir, train_transform)
    test_data = setup_dataset(test, fdr_dir + img_dir, test_transform)
    
    # set up the dataloaders
    train_loader = DataLoader(dataset=train_data,
                              batch_size=1,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    
    test_loader = DataLoader(dataset=test_data, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=2,
                             pin_memory=True)
    
    ###############################################################################
    ################################# LOAD MODEL ##################################
    ###############################################################################
    
    # check GPU availability 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # define model
    if 'cloudscout' in args.MODEL:
        if 'S2' in args.MODEL:
            if args.NUM_BANDS == 3:
                model = CloudScout().to(device)
                summary(model, (3,512,512)) 
                model_folder = 'cloudscout-128a-S2-2018'
            elif args.NUM_BANDS == 8:
                model = CloudScout8().to(device)
                summary(model, (8,512,512))
                model_folder = 'cloudscout-8-S2-2018'                
        if 'L9' in args.MODEL:
            if args.NUM_BANDS == 3:
                model = CloudScout().to(device)
                summary(model, (3,512,512)) 
                model_folder = 'cloudscout-125-L9-2023'
            elif args.NUM_BANDS == 8:
                model = CloudScout8().to(device)
                summary(model, (8,512,512))
                model_folder = 'cloudscout-8-L9-2023'             
    if 'resnet50' in args.MODEL:
        if 'S2' in args.MODEL:
            if args.NUM_BANDS == 3:
                model = ResNet50(num_classes=2, num_bands=3).to(device)
                summary(model, (3,512,512)) 
                model_folder = 'resnet50-128a-S2-2018'
            elif args.NUM_BANDS == 8:
                model = ResNet50(num_classes=2, num_bands=8).to(device)
                summary(model, (8,512,512))
                model_folder = 'resnet50-8-S2-2018'                
        if 'L9' in args.MODEL:
            if args.NUM_BANDS == 3:
                model = ResNet50(num_classes=2, num_bands=3).to(device)
                summary(model, (3,512,512)) 
                model_folder = 'resnet50-125-L9-2023'
            elif args.NUM_BANDS == 8:
                model = ResNet50(num_classes=2, num_bands=8).to(device)
                summary(model, (8,512,512))
                model_folder = 'resnet50-8-L9-2023'     

    filepath = f'checkpoints/source-models/{model_folder}/model70-final.ckpt'
    model.load_state_dict(torch.load(filepath, map_location='cuda:0'))

    ###############################################################################
    ############################# EVALUATE THE MODEL ##############################
    ###############################################################################
    
    print('Evaluating the model...')
    # define lists to keep track predicitons and their corresponding ground truths
    total_predictions = []
    total_labels = []
    
    model.eval()  # disables dropout if any
    with torch.no_grad():
        correct = 0
        total = 0
        # for images, labels in train_loader:
        for images, labels in test_loader:
            
            if args.DATASET == "S2-2018":
                if args.NUM_BANDS == 3:
                    images = images[:,[0,1,8],:,:].to(device, dtype=torch.float)
                elif args.NUM_BANDS == 8: 
                    images = images[:,[0,1,2,3,8,10,11,12],:,:].to(device, dtype=torch.float)   
            elif args.DATASET == "L9-2023":        
                if args.NUM_BANDS == 3:
                    images = images[:,[0,1,4],:,:].to(device, dtype=torch.float) 
                elif args.NUM_BANDS == 8: 
                    images = images.to(device, dtype=torch.float) 
            
            # load ground truths
            labels = labels.to(device)
            
            # forward propagation - input testing images into model
            logits = model(images)
            
            # calcualte testing accuracy 
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_predictions.append(predicted)
            total_labels.append(labels)
    
        print('Test accuracy: {}%'.format(100 * correct / total))
    
    
    " Calculating the confusion matrix - 1: Cloudy and 0: Not Cloudy"
    def confusion(prediction, truth):
        """ Returns the confusion matrix for the values in the `prediction` and `truth`
        tensors, i.e. the amount of positions where the values of `prediction`
        and `truth` are
        - 1 and 1 (True Positive)
        - 1 and 0 (False Positive)
        - 0 and 0 (True Negative)
        - 0 and 1 (False Negative)
        """
    
        confusion_vector = prediction / truth
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)
    
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()
    
        return true_positives, false_positives, true_negatives, false_negatives
    
    # flatten lists 
    prediction = [item for sublist in total_predictions for item in sublist]
    truth = [item for sublist in total_labels for item in sublist]
    
    # convert lists to tensor
    prediction = torch.FloatTensor(prediction)
    truth = torch.FloatTensor(truth)
    
    # calculate the confusion matrix
    confusion_matrix = confusion(prediction, truth)
    
    # false postive rate of test set
    fp = 100 * confusion_matrix[1] / prediction.shape[0]
    print('False positive rate: {}%'.format(fp))

if __name__ == '__main__':
    main()