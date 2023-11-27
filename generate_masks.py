#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions used to create the FISH Mask.
"""

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_the_importance_label(model, data_loader, target, bands, num_samples, cuda_device, grad_type):
    """
    Calculates the empirical Fisher information matrix i.e. gradient of the loss 
    w.r.t. to all model parameters.
    
    Arguments:
        model -- network to train or fine-tune
        data_loader -- training samples, of shape (number of samples, number of channels, height size, width size) 
        num_samples -- number of training samples to compute parameters importance
        cuda_device -- gpu to run on
        grad_type -- (square or absolute) 
        
    Return:
        gradients_dict -- dictionary of gradients for each layer in our model
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square
        
    # define loss 
    criterion = nn.CrossEntropyLoss()
    
    idx = 0
    for inputs, labels in data_loader:
        
        if idx >= num_samples:
            break

        # move data from CPU to GPU
        if bands == 3 and 'L9' in target:
            inputs = inputs[:,[0,1,4],:,:].to(cuda_device, dtype=torch.float) # L9
        elif bands == 3 and 'S2' in target:
            inputs = inputs[:,[0,1,8],:,:].to(cuda_device, dtype=torch.float) # S2
        elif bands == 8 and 'L9' in target:
            inputs = inputs.to(cuda_device, dtype=torch.float) # L9
        elif bands == 8 and 'S2' in target:
            inputs = inputs[:,[0,1,2,3,8,10,11,12],:,:].to(cuda_device, dtype=torch.float) # S2        
        
        # move labels from CPU to GPU
        labels = labels.to(cuda_device)

        # forward pass w/o softmax 
        logits = model(inputs)

        # calculate loss
        loss = criterion(logits, labels)

        # calculate gradient of loss wrt model parameters (param)
        loss.backward()

        # accumulate gradients
        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data
        
        # zero out gradients
        model.zero_grad()
        
        idx += 1

    return gradients_dict


def create_mask_gradient(model, train_dataset, target, bands, num_samples, keep_ratio, sample_type, grad_type, save_dir):
    """
    Creates a FISH mask that indicates which model parameters are important to 
    use for training a model. 
            
    Arguments:
        model -- network to train or fine-tune
        train_dataset -- training samples, of shape (number of samples, number of channels, height size, width size)
        num_samples -- number of training samples to compute parameters importance
        keep_ratio -- trainable parameters to total parameters [0,1] 
        sample_type -- method to select trainable parameters ("label" or "expect")
        grad_type -- ("square" or "absolute") 
        save_dir -- folder to save FISH mask
    
    Return:
        mask -- dictionary of binary matrices
        
    """    

    # move model to GPU (if possible)
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(cuda_device)

    # set up dataloader
    data_loader = DataLoader(train_dataset,
                             batch_size=1,
                             collate_fn=None,
                             shuffle=True
    )

    # select method to calculate gradients
    if sample_type == "label":
        importance_method = calculate_the_importance_label
    else:
        raise NotImplementedError

    # calculate gradients
    gradients = importance_method(model, data_loader, target, bands, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items(): 
        # don't count classifier layer, they should be all trainable
        if 'fc' in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))
        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num >= 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=cuda_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    # Save mask 
    with open(f"{save_dir}/mask.txt", "w") as external_file:        
        for name, param in mask_dict.items():
                print(name, param.data, file=external_file)
                # print('gradients:', param.grad)
        external_file.close()

    model.to(original_device)
    
    # Print the parameter sizes of each layer
    conv_size = 0
    bn_size = 0
    fc_size = 0
    ds_size = 0
    all_params_size = 0
    for name, param in mask_dict.items():
        if "conv" in name:
            conv_size += (param == 1).sum().item()
        elif "bn" in name or "batch_norm" in name:
            bn_size += (param == 1).sum().item()
        elif "fc" in name:
            fc_size += (param == 1).sum().item()
        elif "downsample" in name:
            if "0.weight" in name:
                conv_size += (param == 1).sum().item()
            elif "1.weight" in name:
                bn_size += (param == 1).sum().item()
            elif "1.bias" in name:
                bn_size += (param == 1).sum().item()           
        all_params_size += torch.prod(torch.tensor(param.shape)).item()
    
    print("number of trainable parameters in each layer:")
    print(f"conv param: {conv_size}, bn param: {bn_size}, fc param: {fc_size}, total param: {all_params_size}")    
    
    # Print the parameters 
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "fc" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()
        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(f"total number of trainable parameters (conv + bn): {pretrain_weight_size}, fc parameters: {classifier_size}, total parameters: {all_params_size}")
    print(f"percentage of trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100}%")

    return gradients, mask_dict


def create_mask_random(model, train_dataset, num_samples, keep_ratio, save_dir):
    """
    Creates a binary mask that indicates which model parameters to freeze and 
    to use for training a model.
    
    Args:
        model:
        train_dataset:
        num_samples:
        keep_ratio:
    """
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    gradients = {}
    for name, param in model.named_parameters():
        gradients[name] = torch.rand(param.shape).to(original_device)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        
        # don't count classifier layer, they should be all trainable
        if "fc" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num >= 0

    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=original_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    # Save mask
    with open(f"{save_dir}/random_mask.txt", "w") as external_file:
        
        for name, param in mask_dict.items():
                print(name, param.data, file=external_file)
                # print('gradients:', param.grad)
    
        external_file.close()
    

    model.to(original_device)

    # Print the parameter sizes of each layer
    conv_size = 0
    bn_size = 0
    fc_size = 0
    ds_size = 0
    all_params_size = 0
    for name, param in mask_dict.items():
        if "conv" in name:
            conv_size += (param == 1).sum().item()
        elif "bn" in name or "batch_norm" in name:
            bn_size += (param == 1).sum().item()
        elif "fc" in name:
            fc_size += (param == 1).sum().item()
        elif "downsample" in name:
            if "0.weight" in name:
                conv_size += (param == 1).sum().item()
            elif "1.weight" in name:
                bn_size += (param == 1).sum().item()
            elif "1.bias" in name:
                bn_size += (param == 1).sum().item()           
            #ds_size += (param == 1).sum().item()
    
        all_params_size += torch.prod(torch.tensor(param.shape)).item()
    
    print("number of trainable parameters in each layer:")
    print(f"conv param: {conv_size}, bn param: {bn_size}, fc param: {fc_size}, total param: {all_params_size}")    
    
    # Print the parameters 
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "fc" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(f"total number of trainable parameters (conv + bn): {pretrain_weight_size}, fc parameters: {classifier_size}, total parameters: {all_params_size}")
    print(f"percentage of trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100}%")

    return mask_dict


def create_mask_bias(model, train_dataset, data_collator, num_samples, keep_ratio):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    mask_dict = {}

    for name, param in model.named_parameters():
        if "classifier" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        elif "bias" in name:
            mask_dict[name] = torch.ones_like(param, device=original_device)
        else:
            mask_dict[name] = torch.zeros_like(param, device=original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    bias_params_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            bias_params_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    print(bias_params_size, classifier_size, all_params_size)

    print(f"trainable parameters: {(bias_params_size + classifier_size) / all_params_size * 100} %")

    model.to(original_device)

    return mask_dict

