#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.image as img
from torch.utils.data import Dataset

class setup_dataset(Dataset):
    def __init__(self, data, path , transform = None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_name,label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        
        image = np.load(img_path)
        # image = img.imread(img_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
