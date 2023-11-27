#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CloudScout(nn.Module):
    def __init__(self):
        super(CloudScout, self).__init__()
       
        # convolution
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
       
        # batch normalisation
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
       
        # max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # global max pooling 
        self.pool4 = nn.MaxPool2d(kernel_size=64, stride=None, padding=0)
       
        # fully connected
        self.fc1 = nn.Linear(512, 512) # (256*64*64, 128)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
       
        # conv1 
        x = self.conv1(x)        
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
       
        # conv2
        x = self.conv2(x)        
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # conv3
        x = self.conv3(x)        
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # conv4
        x = self.conv4(x)        
        x = self.batch_norm4(x)
        x = F.relu(x)
        
        # global max pooling
        x = self.pool4(x)
       
        # flatten
        x = x.reshape(x.size(0), -1)
        
        # fc1
        x = self.fc1(x)
        x = F.relu(x)  
        
        # fc2
        x = self.fc2(x)

        return x

# model = CloudScout()  
# summary(model, (3,512,512)) 

# import torch
# out = model(torch.randn(1,3,512,512).to('cuda'))
# print(out.size())

""" 
Memory footprint of CloudScout network 4.93MB.   
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 128, 512, 512]       9,728
├─BatchNorm2d: 1-2                       [-1, 128, 512, 512]       256
├─MaxPool2d: 1-3                         [-1, 128, 256, 256]       --
├─Conv2d: 1-4                            [-1, 256, 256, 256]       295,168
├─BatchNorm2d: 1-5                       [-1, 256, 256, 256]       512
├─MaxPool2d: 1-6                         [-1, 256, 128, 128]       --
├─Conv2d: 1-7                            [-1, 256, 128, 128]       590,080
├─BatchNorm2d: 1-8                       [-1, 256, 128, 128]       512
├─MaxPool2d: 1-9                         [-1, 256, 64, 64]         --
├─Conv2d: 1-10                           [-1, 512, 64, 64]         131,584
├─BatchNorm2d: 1-11                      [-1, 512, 64, 64]         1,024
├─MaxPool2d: 1-12                        [-1, 512, 1, 1]           --
├─Linear: 1-13                           [-1, 512]                 262,656
├─Linear: 1-14                           [-1, 2]                   1,026
==========================================================================================
Total params: 1,292,546
Trainable params: 1,292,546
Non-trainable params: 0
Total mult-adds (G): 32.04
==========================================================================================
Input size (MB): 3.00
Forward/backward pass size (MB): 864.00
Params size (MB): 4.93
Estimated Total Size (MB): 871.93
==========================================================================================
"""

class CloudScout8(nn.Module):
    def __init__(self):
        super(CloudScout8, self).__init__()
       
        # convolution
        self.conv1 = nn.Conv2d(8, 128, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
       
        # batch normalisation
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
       
        # max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # global max pooling 
        self.pool4 = nn.MaxPool2d(kernel_size=64, stride=None, padding=0)
       
        # fully connected
        self.fc1 = nn.Linear(512, 512) # (256*64*64, 128)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
       
        # conv1 
        x = self.conv1(x)        
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
       
        # conv2
        x = self.conv2(x)        
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # conv3
        x = self.conv3(x)        
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # conv4
        x = self.conv4(x)        
        x = self.batch_norm4(x)
        x = F.relu(x)
        
        # global max pooling
        x = self.pool4(x)
       
        # flatten
        x = x.reshape(x.size(0), -1)
        
        # fc1
        x = self.fc1(x)
        x = F.relu(x)  
        
        # fc2
        x = self.fc2(x)

        return x
    
# model = CloudScout8()  
# summary(model, (8,512,512)) 
'''
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Conv2d: 1-1                            [-1, 128, 512, 512]       25,728
├─BatchNorm2d: 1-2                       [-1, 128, 512, 512]       256
├─MaxPool2d: 1-3                         [-1, 128, 256, 256]       --
├─Conv2d: 1-4                            [-1, 256, 256, 256]       295,168
├─BatchNorm2d: 1-5                       [-1, 256, 256, 256]       512
├─MaxPool2d: 1-6                         [-1, 256, 128, 128]       --
├─Conv2d: 1-7                            [-1, 256, 128, 128]       590,080
├─BatchNorm2d: 1-8                       [-1, 256, 128, 128]       512
├─MaxPool2d: 1-9                         [-1, 256, 64, 64]         --
├─Conv2d: 1-10                           [-1, 512, 64, 64]         131,584
├─BatchNorm2d: 1-11                      [-1, 512, 64, 64]         1,024
├─MaxPool2d: 1-12                        [-1, 512, 1, 1]           --
├─Linear: 1-13                           [-1, 512]                 262,656
├─Linear: 1-14                           [-1, 2]                   1,026
==========================================================================================
Total params: 1,308,546
Trainable params: 1,308,546
Non-trainable params: 0
Total mult-adds (G): 36.24
==========================================================================================
Input size (MB): 8.00
Forward/backward pass size (MB): 864.00
Params size (MB): 4.99
Estimated Total Size (MB): 877.00
==========================================================================================
'''

