#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that updates the ResNet50 model to the target domain using DUA.

python tta-dua-resnet50.py \
    --MODEL resnet50-128a-S2-2018 \
    --NUM_BANDS 3 \
    --DATASET L9-2023 \
    --ADAPTATION_BATCH_SIZE 16 \
    --ADAPTATION_SHUFFLE False \
    --ADAPTATION_NUM_SAMPLES 16 \
    --ADAPTATION_AUGMENTATION False \
    --ADAPTATION_DECAY_FACTOR 0.94 \
    --ADAPTATION_MIN_MOMENTUM_CONSTANT 0.005 \
    --ADAPTATION_MOM_PRE 0.1 \
    --GPU 0 \
    --LOG False

"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

from dataset import setup_dataset

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
    parser = argparse.ArgumentParser(description='DUA')
    parser.add_argument('--MODEL',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--NUM_BANDS',
                        help='number of bands',
                        type=int,
                        default=3)
    parser.add_argument('--DATASET',
                        help='data directory',
                        type=str,
                        default='')
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
                        default=10)
    parser.add_argument('--ADAPTATION_AUGMENTATION',
                        help='perform batch augmentation',
                        type=str,
                        default=False)
    parser.add_argument('--ADAPTATION_DECAY_FACTOR',
                        help='momentum decay factor',
                        type=float,
                        default=0.94)
    parser.add_argument('--ADAPTATION_MIN_MOMENTUM_CONSTANT',
                        help='lower bound of momentum',
                        type=float,
                        default=0.05)
    parser.add_argument('--ADAPTATION_MOM_PRE',
                        help='initialised momentum ',
                        type=float,
                        default=0.1)
    parser.add_argument('--SEED_RANGE',
                        help='number of seeds to run',
                        type=int,
                        default='1')
    parser.add_argument('--ROOT',
                        help='root directory',
                        type=str,
                        default='/home/andrew/domain-adaptation-cloud-detection')
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
    adaptation_results = f'checkpoints/dua-{args.MODEL}'
    if not os.path.exists(adaptation_results):
        os.makedirs(adaptation_results)

    logs = f'checkpoints/dua-{args.MODEL}/log-{args.ADAPTATION_NUM_SAMPLES}'
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
    
    adaptation = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]]) # validation set
    test = pd.concat([cloudy[int(0.70*N):int(0.85*N)], not_cloudy[int(0.70*N):int(0.85*N)]]) # validation set
    
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
                                       batch_size=1, 
                                       shuffle=args.ADAPTATION_SHUFFLE, 
                                       num_workers=2,
                                       pin_memory=True)
        
        test_loader = DataLoader(dataset=test_data, 
                                 batch_size=8, 
                                 shuffle=False, 
                                 num_workers=2,
                                 pin_memory=True)
        
        ###############################################################################
        ############################### LOAD THE MODEL ###############################
        ###############################################################################
        
        # check GPU availability 
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        
        # define model
        if args.NUM_BANDS == 3:
            model = ResNet50(num_classes=2, num_bands=3).to(device)
            if 'S2' in args.MODEL:
                print("ResNet50 - S2 (2018) - Bands 1, 2 and 8a")
                model_folder = args.MODEL 
            elif 'L9' in args.MODEL:
                print("ResNet50 - L9 (2023) - Bands 1, 2 and 5")
                model_folder = args.MODEL 
            summary(model, (3,512,512)) 
                
        elif args.NUM_BANDS == 8:
            model = ResNet50(num_classes=2, num_bands=8).to(device)
            if 'S2' in args.MODEL:
                print("ResNet50 - S2 (2018) - 8 bands")
                model_folder = args.MODEL
            elif 'L9' in args.MODEL:
                print("ResNet50 - L9 (2023) - 8 bands")            
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
        ##################### ADAPT THE MODEL ON VALIDATION SET #######################
        ###############################################################################
        
        print(f'Adapting the model on validation set - {args.ADAPTATION_NUM_SAMPLES} number of samples used ...')
        
        # switch to evaluate mode
        model.eval()
        
        # adaptation parameters
        num_samples = args.ADAPTATION_NUM_SAMPLES
        decay_factor = args.ADAPTATION_DECAY_FACTOR
        min_momentum_constant = args.ADAPTATION_MIN_MOMENTUM_CONSTANT
        mom_pre = args.ADAPTATION_MOM_PRE
        
        # augmentations
        tr_transform_adapt = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(512, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
    
        import torchvision.transforms.functional as TF
        def tensor_rot_90(x):
            x = TF.rotate(x, 90)
            return x
        
        
        def tensor_rot_180(x):
            x = TF.rotate(x, 180)
            return x
        
        
        def tensor_rot_270(x):
            x = TF.rotate(x, 270)
            return x
    
    
        def rotate_batch_with_labels(batch, labels):
            images = []
            for img, label in zip(batch, labels):
                if label == 1:
                    img = tensor_rot_90(img)
                elif label == 2:
                    img = tensor_rot_180(img)
                elif label == 3:
                    img = tensor_rot_270(img)
                images.append(img.unsqueeze(0))
            return torch.cat(images)
    
        def rotate_batch(batch, label):
            if label == 'rand':
                labels = torch.randint(4, (len(batch),), dtype=torch.long)
            elif label == 'expand':
                labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                                    torch.zeros(len(batch), dtype=torch.long) + 1,
                                    torch.zeros(len(batch), dtype=torch.long) + 2,
                                    torch.zeros(len(batch), dtype=torch.long) + 3])
                batch = batch.repeat((4, 1, 1, 1))
            else:
                assert isinstance(label, int)
                labels = torch.zeros((len(batch),), dtype=torch.long) + label
            return rotate_batch_with_labels(batch, labels), labels
        
        def get_adaption_inputs(img, batch_size, tr_transform_adapt, device):
            inputs = [(tr_transform_adapt(img[0])) for _ in range(batch_size)]
            inputs = torch.stack(inputs)
            inputs_ssh, _ = rotate_batch(inputs, 'rand')
            inputs_ssh = inputs_ssh.to(device, non_blocking=True)
            return inputs_ssh
    
        ########################################################################## 
        
        # create folder to store batch norm information
        bn_parameters = f'{logs}/batch_norm_update/training_parameters'
        if not os.path.exists(bn_parameters):
            os.makedirs(bn_parameters)
        
        bn_statistics = f'{logs}/batch_norm_update/running_statistics'
        if not os.path.exists(bn_statistics):
            os.makedirs(bn_statistics)
        
        # CHECKER - save training parameters batch norm layers
        with open(f"{bn_parameters}/image_initial.txt", "w") as external_file:
    
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    print(m, file=external_file)
        
            external_file.close()   
    
        # CHECKER - save running mean and variance of batch norm layers 
        with open(f"{bn_statistics}/image_initial.txt", "w") as external_file:
            for layer_name, m in model.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    print(layer_name, file=external_file)
                    print('mean:', m.running_mean, file=external_file)
                    print('variance:', m.running_var, file=external_file)
        
            external_file.close()
    
        ########################################################################## 
    
        with torch.no_grad():
            for i, (images, labels) in enumerate(adaptation_loader):
                
                print("batch_idx:", i)
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
                
                if args.ADAPTATION_AUGMENTATION == 'True':
                    adapted_images = get_adaption_inputs(img=images, batch_size=args.ADAPTATION_BATCH_SIZE, tr_transform_adapt=tr_transform_adapt, device=device)
                elif args.ADAPTATION_AUGMENTATION == 'False':
                    adapted_images = images

                print(adapted_images.shape)
                
                # img = adapted_images[0, :, :, :]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()            
        
                # load ground truths
                labels = labels.to(device)
    
                # update momentum term
                model.eval()
                mom_new = (mom_pre * decay_factor)
                for m in model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.train()
                        m.momentum = mom_new + min_momentum_constant
                mom_pre = mom_new 
    
                # forward propagation - input adaptation images into model
                logits = model(adapted_images)
                
                model.eval()
    
                ########################################################################## 
    
                # CHECKER - save training parameters batch norm layers
                with open(f"{bn_parameters}/image_{i}.txt", "w") as external_file:
    
                    for m in model.modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            print(m, file=external_file)
                
                    external_file.close()
    
                # CHECKER - save running mean and variance of batch norm layers 
                with open(f"{bn_statistics}/image_{i}.txt", "w") as external_file:
                    
                    for layer_name, m in model.named_modules():
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                            print(layer_name, file=external_file)
                            print('mean:', m.running_mean, file=external_file)
                            print('variance:', m.running_var, file=external_file)
                
                    external_file.close()
    
                ##########################################################################         
    
        adapted_model = model 
        
        # save adapted model
        torch.save(adapted_model.state_dict(), f"{logs}/model70_adapted.pth")       
     
        ###############################################################################
        # CHECKER - save parameter values and their corresponding gradients
        with open(f"{logs}/final_parameters.txt", "w") as external_file:
        
            for name, param in adapted_model.named_parameters():
                if param.requires_grad:
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
        
        ###############################################################################
        ####################### EVALUATE THE MODEL ON TEST SET ########################
        ###############################################################################
        print('Evaluating the model on test set...')
        
        # define lists to keep track predicitons and their corresponding ground truths
        total_predictions = []
        total_labels = []
        
        adapted_model.eval()
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
                
                # load ground truths
                labels = labels.to(device)
                
                # forward propagation - input testing images into model
                logits = adapted_model(images)
                
                # calcualte test accuracy 
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

if __name__ == '__main__':
    main()
