#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that updates the CloudScout model using FISH Mask to the target domain.

python fish-mask-cloudscout.py \
    --MODEL cloudscout-128a-S2-2018 \
    --NUM_BANDS 3 \
    --DATASET L9-2023 \
    --TRAIN_EPOCH 300 \
    --TRAIN_BATCH_SIZE 2 \
    --TEST_BATCH_SIZE 2 \
    --FISH_NUM_SAMPLES 2000 \
    --FISH_KEEP_RATIO 0.25 \
    --FISH_SAMPLE_TYPE label \
    --FISH_GRAD_TYPE square \
    --GPU 0 \
    --LOG True
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

from dataset import setup_dataset

from cloudscout import CloudScout
from cloudscout import CloudScout8

from generate_masks import calculate_the_importance_label, create_mask_gradient
from generate_masks import create_mask_random

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import timeit
import time
import sys
import os
os.getcwd()

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
    parser = argparse.ArgumentParser(description='FISH Mask')
    parser.add_argument('--MODEL',
                        help='model folder name',
                        type=str,
                        default='')
    parser.add_argument('--NUM_BANDS',
                        help='number of bands',
                        type=int,
                        default=3)
    parser.add_argument('--DATASET',
                        help='dataset used to update model (either S2-2018 or L9-2023)',
                        type=str,
                        default='')
    parser.add_argument('--TRAIN_EPOCH',
                        help='number of training epoch',
                        type=int,
                        default=300)
    parser.add_argument('--TRAIN_BATCH_SIZE',
                        help='batch size of training data',
                        type=int,
                        default=6)
    parser.add_argument('--TEST_BATCH_SIZE',
                        help='batch size of test data',
                        type=int,
                        default=6)
    parser.add_argument('--FISH_NUM_SAMPLES',
                        help='number of samples used for calculating the FISH mask',
                        type=int,
                        default=256)
    parser.add_argument('--FISH_KEEP_RATIO',
                        help='percentage of parameters to update e.g. 0.005 = 0.5% and 1.00 = 100 (standard training)',
                        type=float,
                        default=0.25)
    parser.add_argument('--FISH_SAMPLE_TYPE',
                        help='whether to calculate the empirical FISH mask (label) or standard empirical FISH mask (expect)',
                        type=str,
                        default='label')
    parser.add_argument('--FISH_GRAD_TYPE',
                        help='whether to square or absolute the gradient calculation',
                        type=str,
                        default='square')
    parser.add_argument('--ROOT',
                        help='root directory',
                        type=str,
                        default='/domain-adaptation-cloud-detection')
    parser.add_argument('--GPU',
                        help='gpu to run on',
                        type=int,
                        default=0)
    parser.add_argument('--LOG',
                        help='record output in text document',
                        type=str,
                        default='True')

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print("args=%s" % args)
    #print("args.name=%s" % args.name)

    # create folder to store adaptation results
    adaptation_results = f'checkpoints/fish-{args.MODEL}'
    if not os.path.exists(adaptation_results):
        os.makedirs(adaptation_results)

    logs = f'checkpoints/fish-{args.MODEL}/log-NS{args.FISH_NUM_SAMPLES}-KR{args.FISH_KEEP_RATIO}-ST{args.FISH_SAMPLE_TYPE}-GT{args.FISH_GRAD_TYPE}'
    if not os.path.exists(logs):
        os.makedirs(logs)

    # record output log
    if args.LOG == 'True':
        orig_stdout = sys.stdout
        f = open(f'{logs}/log.txt', 'w')
        sys.stdout = f
    
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
    print("TOTAL DATASET:", args.DATASET)
    print(labels['is_cloudy'].value_counts())    
        
    ###############################################################################
    ################################ SPLIT DATASET ################################
    ###############################################################################
    
    labels = labels.sample(frac=1, random_state=0)
    N = min(labels['is_cloudy'].value_counts()[0],
            labels['is_cloudy'].value_counts()[1])
    cloudy = labels.loc[labels['is_cloudy'] == 1]
    not_cloudy = labels.loc[labels['is_cloudy'] == 0]
    
    training = pd.concat([cloudy[0:int(0.70*N)], not_cloudy[0:int(0.70*N)]])
    validation = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]])
    # test = pd.concat([cloudy[int(0.85*N):int(len(cloudy)*1.00)], not_cloudy[int(0.85*N):int(len(not_cloudy)*1.00)]])
    
    # why shuffle again?
    training = training.sample(frac=1, random_state=0)
    validation = validation.sample(frac=1, random_state=0)
    # test = test.sample(frac=1, random_state=0)
    print("DATASET SPLITS:", args.DATASET)
    print("-Training -")
    print(training['is_cloudy'].value_counts())
    print("-Validation -")
    print(validation['is_cloudy'].value_counts())
    # print("-Test -")
    # print(test['is_cloudy'].value_counts())

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
    
    # test_transform = transforms.Compose([
    #                                     transforms.ToTensor(),
    #                                     # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
    #                                     ])
    
    # set up the datasets
    train_data = setup_dataset(training, fdr_dir + img_dir, train_transform)
    valid_data = setup_dataset(validation, fdr_dir + img_dir, valid_transform)
    # test_data = setup_dataset(test, fdr_dir + img_dir, test_transform)
    
    ###############################################################################
    ################################# LOAD MODEL ##################################
    ###############################################################################
    
    # check GPU availability 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # define model
    if args.NUM_BANDS == 3:
        model = CloudScout().to(device)
        if 'S2' in args.MODEL:
            print("CloudScout - S2 (2018) - Bands 1, 2 and 8a")
            model_folder = args.MODEL 
        elif 'L9' in args.MODEL:
            print("CloudScout - L9 (2023) - Bands 1, 2 and 5")
            model_folder = args.MODEL 
        summary(model, (3,512,512)) 
            
    elif args.NUM_BANDS == 8:
        model = CloudScout8().to(device)
        if 'S2' in args.MODEL:
            print("CloudScout - S2 (2018) - 8 bands")
            model_folder = args.MODEL
        elif 'L9' in args.MODEL:
            print("CloudScout - L9 (2023) - 8 bands")         
            model_folder = args.MODEL 
        summary(model, (8,512,512))
    
    filepath = f'checkpoints/source-models/{model_folder}/model70-final.ckpt'
    model.load_state_dict(torch.load(filepath, map_location=f'cuda:{args.GPU}'))
    
    # CHECKER - view parameter values and their corresponding gradients
    with open(f"{logs}/initialised_parameters_finetune.txt", "w") as external_file:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data, file=external_file)
                # print('gradients:', param.grad)
        external_file.close()
    
    ###############################################################################
    ########################## CONSTRUCT SPARSITY MASK ############################
    ###############################################################################
    
    print('Constructing the sparsity mask')    
    gradients, mask = create_mask_gradient(
                    model=model,
                    train_dataset=train_data,
                    target=args.DATASET,
                    bands=args.NUM_BANDS,
                    num_samples=args.FISH_NUM_SAMPLES,
                    keep_ratio=args.FISH_KEEP_RATIO,  # i.e., 0.005 = 0.5% (sparse), 1.00 = 100% (standard)
                    sample_type=args.FISH_SAMPLE_TYPE,
                    grad_type=args.FISH_GRAD_TYPE,
                    save_dir=logs
    )
    # NOTE: mask - 1: update, 0: freeze

    # Copy of frozen parameters
    frozen_parameters = {}
    for name, param in model.named_parameters():
        frozen_parameters[name] = param.data * (1-mask[name])
    
    with open(f"{logs}/frozen_parameters_finetune.txt", "w") as external_file:
        print(frozen_parameters, file=external_file)
        external_file.close()
    
    ###############################################################################
    ############################## UPDATE THE MODEL ###############################
    ###############################################################################
    
    # print("Checking model parameters with requires_grad=True...")
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name, param.data)
        
    # define the hyperparameters
    num_epochs = args.TRAIN_EPOCH
    learning_rate = 0.01
    
    # set up the dataloaders
    train_loader = DataLoader(dataset=train_data, 
                              batch_size=args.TRAIN_BATCH_SIZE, 
                              shuffle=True, 
                              num_workers=2,
                              pin_memory=True)
    
    valid_loader = DataLoader(dataset=valid_data, 
                              batch_size=args.TEST_BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=2, 
                              pin_memory=True)
    
    # test_loader = DataLoader(dataset=test_data, 
    #                          batch_size=args.TEST_BATCH_SIZE, 
    #                          shuffle=False, 
    #                          num_workers=2,
    #                          pin_memory=True)
    
    # define loss
    weight = torch.tensor([2., 1.]).cuda()
    criterion = nn.CrossEntropyLoss(
        weight=weight, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    
    # define optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # define lists to keep track of losses and accuracy 
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    print(f'Updating {args.MODEL} on {args.DATASET}...')
    # start timer
    start = timeit.default_timer()
    
    for epoch in range(1, num_epochs + 1):
        
        t0 = time.time()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=np.exp(-0.6*(epoch-1)), last_epoch=-1, verbose=False)
        
        # keep track of training loss and accuracy
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        
        # train the model
        model.train()
        for images, labels in train_loader:
    
            # move data and target tensors to GPU         
            if args.NUM_BANDS == 3 and 'S2' in args.MODEL:
                images = images[:,[0,1,4],:,:].to(device, dtype=torch.float) # L9
            elif args.NUM_BANDS == 3 and 'L9' in args.MODEL:
                images = images[:,[0,1,8],:,:].to(device, dtype=torch.float) # S2
            elif args.NUM_BANDS == 8 and 'S2' in args.MODEL:
                images = images.to(device, dtype=torch.float) # L9
            elif args.NUM_BANDS == 8 and 'L9' in args.MODEL:
                images = images[:,[0,1,2,3,8,10,11,12],:,:].to(device, dtype=torch.float) # S2
            
            # ground truths
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
    
            # restore frozen parameters
            for name, param in model.named_parameters():
                # print(name, param.data)
                param.data = torch.where(mask[name] == 0, frozen_parameters[name], param.data)
            
            # update training loss
            train_loss += loss.item() * images.size(0)
            
            # update training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()      
    
        # keep track of validation loss and accuracy
        valid_loss = 0.0
        valid_correct = 0.0
        valid_total = 0.0
            
        # validate the model
        model.eval()
        for images, labels in valid_loader:
            
            if args.NUM_BANDS == 3 and 'S2' in args.MODEL:
                images = images[:,[0,1,4],:,:].to(device, dtype=torch.float) # L9
            elif args.NUM_BANDS == 3 and 'L9' in args.MODEL:
                images = images[:,[0,1,8],:,:].to(device, dtype=torch.float) # S2
            elif args.NUM_BANDS == 8 and 'S2' in args.MODEL:
                images = images.to(device, dtype=torch.float) # L9
            elif args.NUM_BANDS == 8 and 'L9' in args.MODEL:
                images = images[:,[0,1,2,3,8,10,11,12],:,:].to(device, dtype=torch.float) # S2
            
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            # update validation loss
            valid_loss += loss.item() * images.size(0)
            
            # update validation accuracy 
            _, predicted = torch.max(logits.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # calculate average accuracies
        valid_acc = 100 * valid_correct / valid_total
        train_acc = 100 * train_correct / train_total
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        
        # update learning rate
        scheduler.step()    
            
        # print training and validation statistics and learning rate
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining accuracy: {:.2f} \tValidation accuracy: {:.2f} \tLearning rate: {} \tTime(s): {:.2f}'.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc, scheduler.get_last_lr()[0], time.time() - t0))
        
        # Save model every 20 epochs
        # if epoch%20 == 0:
        #     filepath = f'{logs}/model70'
        #     if not os.path.exists(filepath):
        #         os.makedirs(filepath)
        #     torch.save(model.state_dict(), f'{filepath}/model70-epoch-{epoch}.ckpt')
            
    # stop timer
    stop = timeit.default_timer()
    
    # evaluate run time of training 
    total_time = stop - start
    mins, secs = divmod(total_time, 60)
    hours,  mins = divmod(mins, 60)
    sys.stdout.write(
        f'Total run time of updating {args.MODEL}: %d:%d:%d.\n' % (hours, mins, secs))
    
    
    with open(f"{logs}/trained_parameters_finetune.txt", "w") as external_file:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data, file=external_file)
                # print('gradients:', param.grad)
        external_file.close()
    
    ###############################################################################
    ############################# EVALUATE THE MODEL ##############################
    ###############################################################################
    
    sourceFile = open(f'{logs}/training70.txt', 'w')
    
    print('Evaluating the model...')
    # define lists to keep track predicitons and their corresponding ground truths
    total_predictions = []
    total_labels = []
    
    model.eval()  # disables dropout if any
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            
            if args.NUM_BANDS == 3 and 'S2' in args.MODEL:
                images = images[:,[0,1,4],:,:].to(device, dtype=torch.float) # L9
            elif args.NUM_BANDS == 3 and 'L9' in args.MODEL:
                images = images[:,[0,1,8],:,:].to(device, dtype=torch.float) # S2
            elif args.NUM_BANDS == 8 and 'S2' in args.MODEL:
                images = images.to(device, dtype=torch.float) # L9
            elif args.NUM_BANDS == 8 and 'L9' in args.MODEL:
                images = images[:,[0,1,2,3,8,10,11,12],:,:].to(device, dtype=torch.float) # S2
            
            labels = labels.to(device)
            
            logits = model(images)
            
            # calcualte validation accuracy 
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_predictions.append(predicted)
            total_labels.append(labels)
              
        print('Test accuracy: {}%'.format(100 * correct / total))
        print('Test accuracy:: {}%'.format(100 * correct / total), file=sourceFile)
    
    
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
    plt.savefig(f"{logs}/loss_plot70.png")
    
    # accuracy curves
    plt.figure(figsize=[8,6])
    plt.plot(train_accuracies, 'b', label='Training Accuracy')
    plt.plot(valid_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.legend(frameon=False)
    plt.savefig(f"{logs}/accuracy_plot70.png")
    
    # Save the trained model
    torch.save(model.state_dict(), f'{logs}/model70-final.ckpt')
    
    sourceFile.close()
    
    if args.LOG == 'True':
        sys.stdout = orig_stdout
        f.close()

if __name__ == '__main__':
    main()
    
