################
#
# Deep Flow Prediction for Cavity Flow - Dataset Module
#
# Handle loading, normalization and preprocessing of training data
#
# Author: Tesbo
#
################

import os, sys, random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

def makeDirs(directoryList):
    """Create directories"""
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)

class TurbDataset(Dataset):
    """Cavity flow dataset class"""
    
    # Mode constants
    TRAIN = 0  # Training mode
    TEST = 1   # Testing mode
    
    def __init__(self, prop=None, mode=TRAIN, dataDir="../data/train/", dataDirTest="../data/test/", shuffle=0, normMode=0):
        """
        Initialize dataset
        Parameters:
            prop - Data mixing ratio (None means use all data)
            mode - TRAIN or TEST mode
            dataDir - Training data directory
            dataDirTest - Testing data directory
            shuffle - Whether to shuffle data
            normMode - Normalization mode (0=standard normalization)
        """
        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error: Dataset mode must be TRAIN or TEST")
            sys.exit()
        
        print(f"\nInitializing dataset, mode: {'TRAIN' if mode == self.TRAIN else 'TEST'}")
        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest
        
        # Search for training data files
        self.sims = []
        for filename in os.listdir(dataDir):
            if filename.endswith(".npz"):
                self.sims.append(dataDir + filename)
        
        self.sims.sort()
        
        if len(self.sims) == 0:
            print(f"Error: No training data found in {dataDir}!")
            sys.exit()
        
        print(f"Found {len(self.sims)} training data files")
        
        # Load test data in test mode
        if mode == self.TEST:
            self.simsTest = []
            for filename in os.listdir(dataDirTest):
                if filename.endswith(".npz"):
                    self.simsTest.append(dataDirTest + filename)
            self.simsTest.sort()
            
            if len(self.simsTest) == 0:
                print(f"Warning: No test data found in {dataDirTest}")
                self.simsTest = self.sims  # Fallback to training data
            
            print(f"Found {len(self.simsTest)} test data files")
        
        # Shuffle data
        if shuffle:
            random.shuffle(self.sims)
            if mode == self.TEST:
                random.shuffle(self.simsTest)
        
        # Data split (use 80/20 split in training mode)
        if mode == self.TRAIN:
            splitIdx = int(len(self.sims) * 0.8)
            self.simsSplit = self.sims[:splitIdx]
            print(f"Using first 80% of data for training: {len(self.simsSplit)} samples")
        else:
            # Test mode uses all test data
            self.simsSplit = self.simsTest
            print(f"Using all test data: {len(self.simsSplit)} samples")
        
        # Compute normalization parameters
        self.compute_normalization()
    
    def compute_normalization(self):
        """
        Compute data normalization parameters (mean and standard deviation)
        Use statistics from training set
        """
        print("\nComputing normalization parameters...")
        
        # Collect data from multiple samples for statistics
        num_samples = min(len(self.sims), 100)  # Use at most 100 samples
        all_inputs = []
        all_targets = []
        
        for i in range(num_samples):
            try:
                npfile = np.load(self.sims[i])
                d = npfile['a']
                
                # Inputs: [0,1,2] lid velocity and Reynolds number
                # Outputs: [3,4,5] pressure and velocity fields
                inputs = d[0:3]
                targets = d[3:6]
                
                all_inputs.append(inputs)
                all_targets.append(targets)
            except:
                print(f"Warning: Unable to load file {self.sims[i]}")
                continue
        
        if len(all_inputs) == 0:
            print("Error: Unable to load any data for normalization computation")
            sys.exit()
        
        # Compute input statistics
        inputs_array = np.array(all_inputs)
        self.inputMean = inputs_array.mean(axis=(0, 2, 3), keepdims=True)
        self.inputStd = inputs_array.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        
        # Compute output statistics
        targets_array = np.array(all_targets)
        self.targetMean = targets_array.mean(axis=(0, 2, 3), keepdims=True)
        self.targetStd = targets_array.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        
        print(f"Input mean: {self.inputMean.squeeze()}")
        print(f"Input std: {self.inputStd.squeeze()}")
        print(f"Output mean: {self.targetMean.squeeze()}")
        print(f"Output std: {self.targetStd.squeeze()}")
    
    def __len__(self):
        """Return dataset size"""
        return len(self.simsSplit)
    
    def __getitem__(self, idx):
        """
        Get a single data sample
        Returns: (inputs, targets) tuple
        """
        try:
            npfile = np.load(self.simsSplit[idx])
            d = npfile['a']
            
            # Split inputs and outputs
            inputs = d[0:3]
            targets = d[3:6]
            
            # Normalize - keep (3, 1, 1) shape for proper broadcasting
            inputs = (inputs - self.inputMean[0, :, :, :]) / self.inputStd[0, :, :, :]
            targets = (targets - self.targetMean[0, :, :, :]) / self.targetStd[0, :, :, :]
            
            # Convert to PyTorch tensors
            inputs = torch.from_numpy(inputs).float()
            targets = torch.from_numpy(targets).float()
            
            return inputs, targets
        
        except Exception as e:
            print(f"Error: Failed to load data {self.simsSplit[idx]}: {e}")
            # Return zero data as fallback
            return torch.zeros(3, 128, 128), torch.zeros(3, 128, 128)
    
    def denormalize(self, data, isInput=True):
        """
        Denormalize data
        Parameters:
            data - Normalized data (C, H, W)
            isInput - True for input data, False for output data
        """
        if isInput:
            mean = self.inputMean[0, :, :, :]  # (3, 1, 1)
            std = self.inputStd[0, :, :, :]    # (3, 1, 1)
        else:
            mean = self.targetMean[0, :, :, :] # (3, 1, 1)
            std = self.targetStd[0, :, :, :]   # (3, 1, 1)
        
        # Convert torch tensor to numpy if needed
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        # Denormalize - mean and std are already (3,1,1) shape, can broadcast directly
        result = data * std + mean
        
        return result
