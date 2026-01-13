################
#
# Deep Flow Prediction for Cavity Flow - Testing/Prediction Script
#
# Use trained model for prediction and validation
#
# Author: Tesbo
#
################

import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DfpNet import TurbNetG
import dataset
import utils

######## Testing parameter settings ########

# Batch size (must match training or be larger)
batch_size = 32
# Channel exponent (must match training)
expo = 6
# Model path
model_path = "../data/models/model_final.pth"
# Output directory
output_dir = "../data/test_results/"

##########################

# Get model path from command line arguments
if len(sys.argv) > 1:
    model_path = sys.argv[1]
    print(f"Using model: {model_path}")

if not os.path.exists(model_path):
    print(f"Error: Model file does not exist: {model_path}")
    sys.exit(1)

print("="*60)
print("Cavity Flow Deep Learning Prediction")
print("="*60)
print(f"Model path: {model_path}")
print(f"Batch size: {batch_size}")
print("="*60)

##########################

# Create output directory
utils.makeDirs([output_dir])

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Load test dataset
print("\nLoading test data...")
dataTest = dataset.TurbDataset(
    None,
    mode=dataset.TurbDataset.TEST,
    shuffle=0,
    dataDir="../data/train/",
    dataDirTest="../data/test/"
)
testLoader = DataLoader(dataTest, batch_size=batch_size, shuffle=False, drop_last=False)
print(f"Test batches: {len(testLoader)}")

# Create and load model
print("\nLoading model...")
netG = TurbNetG(channelExponent=expo, dropout=0.)

# Handle models saved with DataParallel
state_dict = torch.load(model_path, map_location=device)
# If model was saved with DataParallel, need to remove 'module.' prefix
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
    else:
        new_state_dict[k] = v

netG.load_state_dict(new_state_dict)
netG.to(device)
netG.eval()
print("Model loaded successfully")

print(f"Network parameters: {sum(p.numel() for p in netG.parameters() if p.requires_grad):,}")

##########################
# Testing loop
##########################

print("\n" + "="*60)
print("Starting prediction")
print("="*60 + "\n")

criterionL1 = nn.L1Loss()
criterionL1.to(device)

total_loss = 0.0
sample_count = 0

with torch.no_grad():
    for i, testdata in enumerate(testLoader):
        inputs, targets = testdata
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Predict
        outputs = netG(inputs)
        
        # Calculate loss
        lossL1 = criterionL1(outputs, targets)
        total_loss += lossL1.item()
        
        print(f"Batch {i+1}/{len(testLoader)}, L1 Loss: {lossL1.item():.6f}")
        
        # Save result images (first 10 batches)
        if i < 10:
            for j in range(min(batch_size, outputs.size(0))):
                # Denormalize
                output_denorm = dataTest.denormalize(outputs[j].cpu().numpy(), isInput=False)
                target_denorm = dataTest.denormalize(targets[j].cpu().numpy(), isInput=False)
                
                # Save comparison images
                filename = output_dir + f"sample_{sample_count:04d}"
                utils.imageOut(filename, output_denorm, target_denorm, saveTargets=True)
                
                sample_count += 1
                
                if sample_count % 10 == 0:
                    print(f"  Saved {sample_count} prediction result images")

# Calculate average loss
avg_loss = total_loss / len(testLoader)
print(f"\nAverage L1 Loss: {avg_loss:.6f}")

# Save loss statistics
with open(output_dir + "test_results.txt", 'w') as f:
    f.write(f"Model: {model_path}\n")
    f.write(f"Test samples: {len(testLoader) * batch_size}\n")
    f.write(f"Average L1 Loss: {avg_loss:.6f}\n")

print(f"\nPrediction results saved at: {output_dir}")
print("\n" + "="*60)
print("Prediction completed!")
print("="*60)
