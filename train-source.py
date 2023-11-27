#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that trains a source cloud detector.

python train-source.py \
    --MODEL_ARCH cloudscout \
    --DATASET S2-2018 \
    --NUM_BANDS 3 \
    --GPU 0 \
    --NUM_EPOCHS 300
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
import timeit
import time
import sys
import os
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
    parser.add_argument('--MODEL_ARCH',
                        help='model architecture (either cloudscout, cloudscout8, resnet50)',
                        type=str,
                        default='cloudscout')
    parser.add_argument('--NUM_BANDS',
                        help='number of bands is either 3 or 8',
                        type=int,
                        default=3)
    parser.add_argument('--DATASET',
                        help='source dataset used to train model (either S2-2018 or L9-2023)',
                        type=str,
                        default='')
    parser.add_argument('--NUM_EPOCHS',
                        help='number of epochs',
                        type=int,
                        default=300)    
    parser.add_argument('--BATCH_SIZE',
                        help='batch size',
                        type=int,
                        default=4)    
    parser.add_argument('--LEARNING_RATE',
                        help='initial learning rate',
                        type=float,
                        default=0.01)    
    parser.add_argument('--ROOT',
                        help='root directory',
                        type=str,
                        default='/home/andrew/domain-adaptation-cloud-detection')
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
    torch.cuda.set_device(0)
    
    # folder directory
    fdr_dir = args.ROOT
    
    ###############################################################################
    ################## DEFINE DATASET DIRECTORY AND LOAD LABELS ###################
    ###############################################################################
    
    if args.DATASET == "S2-2018":
        img_dir = '/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/numpy/images'
        lab_dir_30 = '/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/labels/TF30.csv'
        lab_dir_70 = '/datasets/Sentinel-2-Cloud-Mask-Catalogue/preprocessed/labels/TF70.csv'
    elif args.DATASET == "L9-2023":
        img_dir = '/datasets/Landsat-9-Level-1/preprocessed/numpy/images'
        lab_dir_30 = '/datasets/Landsat-9-Level-1/preprocessed/labels/TF30.csv'
        lab_dir_70 = '/datasets/Landsat-9-Level-1/preprocessed/labels/TF70.csv'
    
    # load labels
    labels_30 = pd.read_csv(fdr_dir + lab_dir_30)
    labels_70 = pd.read_csv(fdr_dir + lab_dir_70)
    print('\nTOTAL TF30\n', labels_30['is_cloudy'].value_counts())
    print('\nTOTAL TF70\n', labels_70['is_cloudy'].value_counts())
    
    # plot pie chart "
    # label = 'Not Cloudy', 'Cloudy'
    # plt.figure(figsize = (8,8))
    # plt.pie(labels_30.groupby('is_cloudy').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
    # plt.show()
    
    # label = 'Not Cloudy', 'Cloudy'
    # plt.figure(figsize = (8,8))
    # plt.pie(labels_70.groupby('is_cloudy').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
    # plt.show()

    ###############################################################################
    ################################ SPLIT DATASET ################################
    ###############################################################################
    
    # TF30 dataset
    labels_30 = labels_30.sample(frac=1, random_state=0)
    N = min(labels_30['is_cloudy'].value_counts()[0],
            labels_30['is_cloudy'].value_counts()[1])
    cloudy = labels_30.loc[labels_30['is_cloudy'] == 1]
    not_cloudy = labels_30.loc[labels_30['is_cloudy'] == 0]
    
    training_30 = pd.concat([cloudy[0:int(0.70*N)], not_cloudy[0:int(0.70*N)]])
    validation_30 = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]])
    # test_30 = pd.concat([cloudy[int(0.85*N):int(len(cloudy)*1.00)], not_cloudy[int(0.85*N):int(len(not_cloudy)*1.00)]])
    
    # shuffle again?
    training_30 = training_30.sample(frac=1, random_state=0)
    validation_30 = validation_30.sample(frac=1, random_state=0)
    # test_30 = test_30.sample(frac=1, random_state=0)
    print('\nTF30 TRAINING\n', training_30['is_cloudy'].value_counts())
    print('\nTF30 VALIDATION\n', validation_30['is_cloudy'].value_counts())
    # print(test_30['is_cloudy'].value_counts())

    ###############################################################################
    # TF70 dataset
    labels_70 = labels_70.sample(frac=1, random_state=0)
    N = min(labels_70['is_cloudy'].value_counts()[0],
            labels_70['is_cloudy'].value_counts()[1])
    cloudy = labels_70.loc[labels_70['is_cloudy'] == 1]
    not_cloudy = labels_70.loc[labels_70['is_cloudy'] == 0]
    
    training_70 = pd.concat([cloudy[0:int(0.70*N)], not_cloudy[0:int(0.70*N)]])
    validation_70 = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]])
    # test_70 = pd.concat([cloudy[int(0.85*N):int(len(cloudy)*1.00)], not_cloudy[int(0.85*N):int(len(not_cloudy)*1.00)]])
    
    # shuffle again?
    training_70 = training_70.sample(frac=1, random_state=0)
    validation_70 = validation_70.sample(frac=1, random_state=0)
    # test_70 = test_70.sample(frac=1, random_state=0)
    print('\nTF70 TRAINING\n', training_70['is_cloudy'].value_counts())
    print('\nTF70 VALIDATION\n', validation_70['is_cloudy'].value_counts())
    # print(test_70['is_cloudy'].value_counts())

    ###############################################################################
    ############################# PRE-PROCESS DATASET #############################
    ###############################################################################
    
    # define transformations 
    class AddGaussianNoise(object):
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean
    
        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
        def __repr__(self):
            return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
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
    
    valid_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                         # transforms.ToPILImage(),
                                         # transforms.RandomHorizontalFlip(p=0.5),
                                         # transforms.ToTensor(),
                                         ])
    
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                        ])

    # set up the datasets
    train_data_30 = setup_dataset(training_30, fdr_dir + img_dir, train_transform)
    valid_data_30 = setup_dataset(validation_30, fdr_dir + img_dir, valid_transform)
    # test_data_30 = setup_dataset(test_30, fdr_dir + img_dir, test_transform)
    
    train_data_70 = setup_dataset(training_70, fdr_dir + img_dir, train_transform)
    valid_data_70 = setup_dataset(validation_70, fdr_dir + img_dir, valid_transform)
    # test_data_70 = setup_dataset(test_70, fdr_dir + img_dir, test_transform)


    # define the hyperparameters
    num_epochs = args.NUM_EPOCHS
    batch_size = args.BATCH_SIZE
    learning_rate = args.LEARNING_RATE

    # set up the dataloaders
    train_loader_30 = DataLoader(dataset=train_data_30,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    
    valid_loader_30 = DataLoader(dataset=valid_data_30,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=True)
    
    # test_loader_30 = DataLoader(dataset=test_data_30,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          num_workers=2,
    #                          pin_memory=True)
    
    train_loader_70 = DataLoader(dataset=train_data_70,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    
    valid_loader_70 = DataLoader(dataset=valid_data_70,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=True)
    
    # test_loader_70 = DataLoader(dataset=test_data_70,
    #                          batch_size=batch_size,
    #                          shuffle=False,
    #                          num_workers=2,
    #                          pin_memory=True)

###############################################################################
################################# LOAD MODEL ##################################
###############################################################################

    # check GPU availability
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # define model
    if 'cloudscout' in args.MODEL_ARCH:
        if args.DATASET == 'S2-2018':
            if args.NUM_BANDS == 3:
                model = CloudScout().to(device)
                summary(model, (3,512,512)) 
                model_folder = 'cloudscout-128a-S2-2018'
            elif args.NUM_BANDS == 8:
                model = CloudScout8().to(device)
                summary(model, (8,512,512))
                model_folder = 'cloudscout-8-S2-2018'                
        elif args.DATASET == 'L9-2023':
            if args.NUM_BANDS == 3:
                model = CloudScout().to(device)
                summary(model, (3,512,512)) 
                model_folder = 'cloudscout-125-L9-2023'
            elif args.NUM_BANDS == 8:
                model = CloudScout8().to(device)
                summary(model, (8,512,512))
                model_folder = 'cloudscout-8-L9-2023'             
    if 'resnet50' in args.MODEL_ARCH:
        if args.DATASET == 'S2-2018':
            if args.NUM_BANDS == 3:
                model = ResNet50(num_classes=2, num_bands=3).to(device)
                summary(model, (3,512,512)) 
                model_folder = 'resnet50-128a-S2-2018'
            elif args.NUM_BANDS == 8:
                model = ResNet50(num_classes=2, num_bands=8).to(device)
                summary(model, (8,512,512))
                model_folder = 'resnet50-8-S2-2018'                
        elif args.DATASET == 'L9-2023':
            if args.NUM_BANDS == 3:
                model = ResNet50(num_classes=2, num_bands=3).to(device)
                summary(model, (3,512,512)) 
                model_folder = 'resnet50-125-L9-2023'
            elif args.NUM_BANDS == 8:
                model = ResNet50(num_classes=2, num_bands=8).to(device)
                summary(model, (8,512,512))
                model_folder = 'resnet50-8-L9-2023'  

    ###############################################################################
    ####################### TRAIN FEATURE EXTRACTION LAYER ########################
    ###############################################################################
    
    print(f'Training {model_folder} feature extraction layer...')
    
    # create folder to store training results
    if not os.path.exists(f'checkpoints/source-models/{model_folder}'):
        os.makedirs(f'checkpoints/source-models/{model_folder}')

    # define loss
    weight = torch.tensor([2., 1.]).cuda()
    criterion = nn.CrossEntropyLoss(
        weight=weight, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    
    # define optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # start timer
    start = timeit.default_timer()

    # train feature extraction layer
    for epoch in range(1, num_epochs + 1):
    
        t0 = time.time()
        scheduler_30 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=np.exp(-0.6*(epoch-1)), last_epoch=-1, verbose=False)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=0.6, last_epoch=-1, verbose=False)
    
        # keep track of training loss and accuracy
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
    
        training_step = 0
    
        model.train()
        for images, labels in train_loader_30:
            
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
    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
    
            # forward propagation - input training images into model
            logits = model(images)
    
            # calculate the batch loss
            loss = criterion(logits, labels)
    
            # backward propagation: compute gradient of the loss wrt model parameters
            loss.backward()
    
            # update the model parameters
            optimizer.step()
    
            # update training loss
            train_loss += loss.item() * images.size(0)
    
            # update training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            training_step += 1
        
        # keep track of validation loss and accuracy
        valid_loss = 0.0
        valid_correct = 0.0
        valid_total = 0.0
    
        # validate the model
        model.eval()
        for images, labels in valid_loader_30:
            
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
    
            # forward propagation - input validation images into model
            logits = model(images)
    
            # calculate the batch loss
            loss = criterion(logits, labels)
    
            # update validation loss
            valid_loss += loss.item() * images.size(0)
    
            # update validation accuracy
            _, predicted = torch.max(logits.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
    
        # calculate average losses
        train_loss = train_loss/len(train_loader_30.sampler)
        valid_loss = valid_loss/len(valid_loader_30.sampler)
    
        # calculate average accuracies
        train_acc = 100 * train_correct / train_total
        valid_acc = 100 * valid_correct / valid_total
    
        # update learning rate
        scheduler_30.step()
    
        # print training and validation statistics and learning rate
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining accuracy: {:.2f} \tValidation accuracy: {:.2f} \tLearning rate: {} \tTime(s): {:.2f}'.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc, scheduler_30.get_last_lr()[0], time.time() - t0))
    
        # Save model every 20 epochs
        # if epoch%20 == 0:
        #     filepath = f'checkpoints/source-models/{model_folder}/saves-train'
        #     if not os.path.exists(filepath):
        #         os.makedirs(f'{filepath}')
        #     torch.save(model.state_dict(), f'{filepath}/model70-epoch-{epoch}.ckpt')
    
    # stop timer
    stop = timeit.default_timer()
    
    # evaluate run time of training
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours,  mins = divmod(mins, 60)
    sys.stdout.write(
        'Total run time of training feature extraction layer: %d:%d:%d.\n' % (hours, mins, secs))
    
    # save the trained model
    torch.save(model.state_dict(), f'checkpoints/source-models/{model_folder}/model30-final.ckpt')

    ###############################################################################
    ######################### TRAIN CLASSIFICATION LAYER ##########################
    ###############################################################################
    
    print(f'Training {model_folder} classification layer...')
    
    # CHECKER - display trainable model parameters "
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         # print(name, param.data)
    #         print(name)
    
    # freeze feature extraction layer
    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # CHECKER - display trainable model parameters "
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         # print(name, param.data)
    #         print(name)
    
    # define loss
    weight = torch.tensor([2., 1.]).cuda()
    criterion = nn.CrossEntropyLoss(
        weight=weight, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    
    # define optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # start timer
    start = timeit.default_timer()
    
    # define lists to keep track of losses and accuracy 
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    # train feature extraction layer
    for epoch in range(1, num_epochs + 1):
    
        t0 = time.time()
        scheduler_70 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=np.exp(-0.6*(epoch-1)), last_epoch=-1, verbose=False)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=0.6, last_epoch=-1, verbose=False)
    
        # keep track of training loss and accuracy
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
    
        training_step = 0
    
        model.train()
        for images, labels in train_loader_70:
    
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
    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
    
            # forward propagation - input training images into model
            logits = model(images)
    
            # calculate the batch loss
            loss = criterion(logits, labels)
    
            # backward propagation: compute gradient of the loss wrt model parameters
            loss.backward()
    
            # update the model parameters
            optimizer.step()
    
            # update training loss
            train_loss += loss.item() * images.size(0)
    
            # update training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            training_step += 1
    
        # keep track of validation loss and accuracy
        valid_loss = 0.0
        valid_correct = 0.0
        valid_total = 0.0
    
        # validate the model
        model.eval()
        for images, labels in valid_loader_70:
            
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
    
            # forward propagation - input validation images into model
            logits = model(images)
    
            # calculate the batch loss
            loss = criterion(logits, labels)
    
            # update validation loss
            valid_loss += loss.item() * images.size(0)
    
            # update validation accuracy
            _, predicted = torch.max(logits.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
    
        # calculate average losses
        train_loss = train_loss/len(train_loader_70.sampler)
        valid_loss = valid_loss/len(valid_loader_70.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    
        # calculate average accuracies
        train_acc = 100 * train_correct / train_total
        valid_acc = 100 * valid_correct / valid_total
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
    
        # update learning rate
        scheduler_70.step()
    
        # print training and validation statistics and learning rate
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining accuracy: {:.2f} \tValidation accuracy: {:.2f} \tLearning rate: {} \tTime(s): {:.2f}'.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc, scheduler_70.get_last_lr()[0], time.time() - t0))
    
        # Save model every 20 epochs
        # if epoch%20 == 0:
        #     filepath = f'checkpoints/source-models/{model_folder}/saves-train'
        #     if not os.path.exists(filepath):
        #         os.makedirs(f'{filepath}')
        #     torch.save(model.state_dict(), f'{filepath}/model70-epoch-{epoch}.ckpt')
    
    # stop timer
    stop = timeit.default_timer()
    
    # evaluate run time of training
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours,  mins = divmod(mins, 60)
    sys.stdout.write(
        'Total run time of training classification layer: %d:%d:%d.\n' % (hours, mins, secs))
    
    # save the trained model
    torch.save(model.state_dict(), f'checkpoints/source-models/{model_folder}/model70-final.ckpt')
    
    ###############################################################################
    ############################# EVALUATE THE MODEL ##############################
    ###############################################################################
    
    sourceFile = open(f'checkpoints/source-models/{model_folder}/training.txt', 'w')
    
    print('Evaluating the model...')
    # define lists to keep track predicitons and their corresponding ground truths
    total_predictions = []
    total_labels = []
    
    model.eval()  # disables dropout if any
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader_70:
    
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
        print('Test accuracy: {}%'.format(100 * correct / total), file=sourceFile)
    
    
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
    print('False positive rate - test set: {}%'.format(fp))
    print('False positive rate - test set: {}%'.format(fp), file=sourceFile)
    
    " Plotting the losses vs epochs "
    # loss curves
    plt.figure(figsize=[8,6])
    plt.plot(train_losses, 'b', label='Training loss')
    plt.plot(valid_losses, 'r', label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.legend(frameon=False)
    plt.savefig(f"checkpoints/source-models/{model_folder}/loss_plot.png")
    
    # accuracy curves
    plt.figure(figsize=[8,6])
    plt.plot(train_accuracies, 'b', label='Training Accuracy')
    plt.plot(valid_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.legend(frameon=False)
    plt.savefig(f"checkpoints/source-models/{model_folder}/accuracy_plot.png")
    
    sourceFile.close()

if __name__ == '__main__':
    main()