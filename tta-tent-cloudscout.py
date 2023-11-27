#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that updates the CloudScout model to the target domain using TENT.

python tta-tent-cloudscout.py \
    --MODEL cloudscout-128a-S2-2018 \
    --NUM_BANDS 3 \
    --DATASET L9-2023 \
    --ADAPTATION_EPOCH 1 \
    --ADAPTATION_BATCH_SIZE 6 \
    --ADAPTATION_SHUFFLE False \
    --ADAPTATION_NUM_SAMPLES 9999 \
    --ADAPTATION_RESET_STATS True \
    --ADAPTATION_LEARNING_RATE 0.001 \
    --GPU 0 \
    --LOG False
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.distributions import Categorical
from torchsummary import summary

from dataset import setup_dataset

from cloudscout import CloudScout
from cloudscout import CloudScout8

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
    parser = argparse.ArgumentParser(description='TENT')
    parser.add_argument('--MODEL',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--NUM_BANDS',
                        help='number of bands',
                        type=int,
                        default=3)
    parser.add_argument('--DATASET',
                        help='dataset is either Sentinel-2 (S2-2018) or Landsat-9 (L9-2023)',
                        type=str,
                        default='')
    parser.add_argument('--ADAPTATION_EPOCH',
                        help='number of epoch used for adaptation',
                        type=int,
                        default=1)
    parser.add_argument('--ADAPTATION_BATCH_SIZE',
                        help='batch size used for adaptation',
                        type=int,
                        default=1)
    parser.add_argument('--ADAPTATION_SHUFFLE',
                        help='whether to shuffle the dataset used for adaptation',
                        type=str,
                        default='False')
    parser.add_argument('--ADAPTATION_NUM_SAMPLES',
                        help='number of samples used for adaptation',
                        type=int,
                        default=9999)
    parser.add_argument('--ADAPTATION_RESET_STATS',
                        help='whether to reset running statistics of batch norm layers',
                        type=str,
                        default='True')
    parser.add_argument('--ADAPTATION_LEARNING_RATE',
                        help='learning rate for adaptation',
                        type=float,
                        default='0.001')
    parser.add_argument('--SEED_RANGE',
                        help='number of seeds to run',
                        type=int,
                        default='1')
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
    adaptation_results = f'checkpoints/tent-{args.MODEL}'
    if not os.path.exists(adaptation_results):
        os.makedirs(adaptation_results)
        
    logs = f'checkpoints/tent-{args.MODEL}/log-E{args.ADAPTATION_EPOCH}-BS{args.ADAPTATION_BATCH_SIZE}-R{args.ADAPTATION_RESET_STATS}'
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

    adaptation = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]])
    test = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]])

    # why shuffle again?
    adaptation = adaptation.sample(frac=1, random_state=0)
    test = test.sample(frac=1, random_state=0)
    print("ADAPTATION DATASET:", args.DATASET)
    print(adaptation['is_cloudy'].value_counts())
    print("TEST DATASET:", args.DATASET)
    print(test['is_cloudy'].value_counts())

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
    
    
    adaptation_transform = transforms.Compose([
                                              transforms.ToTensor(),
                                              # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                              ])
    
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.3837, 0.3630, 0.3838), (0.2696, 0.2729, 0.2553)),
                                        ])

    # set up the datasets
    adaptation_data = setup_dataset(adaptation, fdr_dir + img_dir, adaptation_transform)
    test_data = setup_dataset(test, fdr_dir + img_dir, test_transform)
    
    # define lists to save results
    test_accuracy = []
    false_positive_rates = []

    for seed in range(0, args.SEED_RANGE):
        print('seed:', seed)
        torch.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed) 
        torch.cuda.manual_seed(seed) 
        np.random.seed(seed) 
        random.seed(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # set up the dataloaders
        adaptation_loader = DataLoader(dataset=adaptation_data, 
                                       batch_size=args.ADAPTATION_BATCH_SIZE, 
                                       shuffle=args.ADAPTATION_SHUFFLE, 
                                       num_workers=2,
                                       pin_memory=True)
        
        test_loader = DataLoader(dataset=test_data, 
                                 batch_size=8, 
                                 shuffle=False, 
                                 num_workers=2,
                                 pin_memory=True)
    
        ###############################################################################
        ############################### LOAD THE MODEL ################################
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
    
        filepath = 'checkpoints/source-models/' + model_folder + '/model70-final.ckpt'
        model.load_state_dict(torch.load(filepath, map_location=f'cuda:{args.GPU}'))
    
        ###############################################################################
        # CHECKER - save initial parameter values and their corresponding gradients
        with open(f"{logs}/initial_parameters.txt", "w") as external_file:
        
            for name, param in model.named_parameters():
                print(name, param.data, file=external_file)
                # print('gradients:', param.grad)
        
            external_file.close()
    
        # CHECKER - save training parameters in batch norm layers
        with open(f"{logs}/initial_bn_training_parameters.txt", "w") as external_file:
        
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    print(m, file=external_file)
        
            external_file.close()
    
        # CHECKER - save running mean and variance of batch norm layers 
        with open(f"{logs}/initial_bn_running_statistics.txt", "w") as external_file:
            
            for layer_name, m in model.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    print(layer_name, file=external_file)
                    print('mean:', m.running_mean, file=external_file)
                    print('variance:', m.running_var, file=external_file)
        
            external_file.close()
        ###############################################################################
    
        ###############################################################################
        # FREEZE MODEL PARAMETERS IN SPECIFIC LAYERS 
        ###############################################################################
        
        # train BN layer only
        for name, param in model.named_parameters():
            if 'batch_norm' in name or 'bn' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        if args.ADAPTATION_RESET_STATS == 'True':
            " Resetting running statistics (mean and variance)"
            for layer_name, m in model.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.reset_running_stats()
        
        ###############################################################################
        # # CHECKER - display running mean and variance after resetting running statistics 
        # for layer_name, m in model.named_modules():
        #     if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        #         print(layer_name)
        #         print('mean:', m.running_mean)
        #         print('variance:', m.running_var)
        ###############################################################################
        
        # display trainable model parameters
        print('Displaying trainable model parameter')
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name)
                # print(name, param.data)
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad == False:
        #         print(name)
        
        
        # CHECKER - save resetted running mean and variance of batch norm layers 
        with open(f"{logs}/resetted_bn_running_statistics.txt", "w") as external_file:
            
            for layer_name, m in model.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    print(layer_name, file=external_file)
                    print('mean:', m.running_mean, file=external_file)
                    print('variance:', m.running_var, file=external_file)
        
            external_file.close()
        ###############################################################################
    
        ###############################################################################
        ##################### ADAPT THE MODEL ON VALIDATION SET #######################
        ###############################################################################
        
        print('Adapting the model on validation set...')
        
        # define the hyperparameters
        learning_rate = args.ADAPTATION_LEARNING_RATE
        
        # define optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
        # define adaptation parameters
        num_samples = args.ADAPTATION_NUM_SAMPLES
    
        for epoch in range(1, args.ADAPTATION_EPOCH + 1):
            
            print(f'Epoch {epoch}')
            
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=np.exp(-0.6*(epoch-1)), last_epoch=-1, verbose=False)
            
            # train the model
            model.train()
            for i, (images, labels) in enumerate(adaptation_loader):
                
                if args.ADAPTATION_RESET_STATS == 'False':
                    break
        
                if i == num_samples/adaptation_loader.batch_size:
                    break
        
                # load images
                if args.NUM_BANDS == 3 and 'S2' in args.MODEL:
                    images = images[:,[0,1,4],:,:].to(device, dtype=torch.float) # L9
                elif args.NUM_BANDS == 3 and 'L9' in args.MODEL:
                    images = images[:,[0,1,8],:,:].to(device, dtype=torch.float) # S2
                elif args.NUM_BANDS == 8 and 'S2' in args.MODEL:
                    images = images.to(device, dtype=torch.float) # L9
                elif args.NUM_BANDS == 8 and 'L9' in args.MODEL:
                    images = images[:,[0,1,2,3,8,10,11,12],:,:].to(device, dtype=torch.float) # S2
                
                # # load ground truths
                # labels = labels.to(device)
                
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                
                # forward propagation - input training images into model
                logits = model(images)        
        
                # # numpy - entropy
                # p = softmax(logits).cpu().detach().numpy()
                # logp = np.log(p)
                # entropy = np.sum(-p*logp)
                
                # # torch - entropy(probs)
                # softmax = torch.nn.Softmax(dim=1)
                # loss = Categorical(probs=softmax(logits), logits=None).entropy()
                
                # calculate the batch loss
                # loss = torch.mean(Categorical(probs=None, logits=logits).entropy())
                loss = torch.sum(Categorical(probs=None, logits=logits).entropy())
                
                # backward propagation: compute gradient of the loss wrt model parameters
                loss.backward()
                
                # update the model parameters
                optimizer.step()
                
                print('Training Step: {} \tEntropy Loss: {:.6f} \tLearning Rate: {}'.format(i, loss, scheduler.get_last_lr()[0]))
                 
        
        # save adapted model
        adapted_model = model    
        torch.save(adapted_model.state_dict(), f"{logs}/model70_adapted.pth")
        
        # CHECKER - save parameter values and their corresponding gradients
        with open(f"{logs}/final_parameters.txt", "w") as external_file:
        
            for name, param in adapted_model.named_parameters():
                print(name, param.data, file=external_file)
                # print('gradients:', param.grad)
        
            external_file.close()
        
        # CHECKER - save training parameters batch norm layers
        with open(f"{logs}/final_bn_training_parameters.txt", "w") as external_file:
        
            for m in adapted_model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    print(m, file=external_file)
        
            external_file.close()    
           
        # CHECKER - save running mean and variance of batch norm layers 
        with open(f"{logs}/final_bn_running_statistics.txt", "w") as external_file:
            
            for layer_name, m in adapted_model.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    print(layer_name, file=external_file)
                    print('mean:', m.running_mean, file=external_file)
                    print('variance:', m.running_var, file=external_file)
        
            external_file.close() 
        
        ###############################################################################
        ############################# EVALUATE THE MODEL ##############################
        ###############################################################################
        
        print('Evaluating the model on test set...')
        
        # define lists to keep track predicitons and their corresponding ground truths
        total_predictions = []
        total_labels = []
        
        adapted_model.eval()  # disables dropout if any
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                
                # load images
                if args.NUM_BANDS == 3 and 'S2' in args.MODEL:
                    images = images[:,[0,1,4],:,:].to(device, dtype=torch.float) # L9
                elif args.NUM_BANDS == 3 and 'L9' in args.MODEL:
                    images = images[:,[0,1,8],:,:].to(device, dtype=torch.float) # S2
                elif args.NUM_BANDS == 8 and 'S2' in args.MODEL:
                    images = images.to(device, dtype=torch.float) # L9
                elif args.NUM_BANDS == 8 and 'L9' in args.MODEL:
                    images = images[:,[0,1,2,3,8,10,11,12],:,:].to(device, dtype=torch.float) # S2
                
                labels = labels.to(device)
                
                logits = adapted_model(images)
                
                # calcualte validation accuracy 
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                total_predictions.append(predicted)
                total_labels.append(labels)
                  
            print('Test accuracy of model: {}%'.format(100 * correct / total))
            test_accuracy.append(round(100 * correct / total, 2))
        
        
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
        false_positive_rates.append(round(fp, 2))
    
    if args.LOG == 'True':    
        sys.stdout = orig_stdout
        f.close()

    with open(f"{logs}/results.txt", "w") as output:
        output.write(str("TENT PARAMETERS\n"))
        output.write(str(f" Batch size: {adaptation_loader.batch_size}\n Epochs: {args.ADAPTATION_EPOCH}\n\n\n"))
        output.write(str("UPDATED MODEL\n"))
        output.write(str("Test accuracies: "))
        output.write(str(f"{test_accuracy}\n"))
        output.write(str("False positive rates: "))
        output.write(str(f"{false_positive_rates}\n"))

if __name__ == '__main__':
    main()
