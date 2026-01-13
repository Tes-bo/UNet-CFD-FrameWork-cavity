################
#
# Deep Flow Prediction for Cavity Flow - Training Script
#
# Train deep learning model for cavity flow field prediction using GPU
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
import torch.optim as optim

from DfpNet import TurbNetG, weights_init
import dataset
import utils

######## Training parameter settings ########

# Training iterations
iterations = 10000
# Batch size (V100 has 32GB memory, can use larger batch)
batch_size = 32  # Increased from 10 to 32 to fully utilize GPU
# Learning rate
lrG = 0.0006
# Whether to use learning rate decay
decayLr = True
# Channel exponent (controls network size) - V100 can support larger networks
expo = 6  # Increased from 5 to 6 to enhance model capacity
# Dataset configuration
prop = None  # Use all training data
# Whether to save training loss
saveL1 = True

# Early Stopping parameters
use_early_stopping = True  # Whether to use early stopping
patience = 50  # How many epochs to tolerate without validation loss improvement
min_delta = 1e-6  # Minimum improvement amount (considered no improvement if below this)

##########################

# Output prefix
prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print(f"Output prefix: {prefix}")

# Network configuration
dropout = 0.
doLoad = ""  # Pretrained model path

print("="*60)
print("Cavity Flow Deep Learning Training")
print("="*60)
print(f"Learning rate: {lrG}")
print(f"Learning rate decay: {decayLr}")
print(f"Iterations: {iterations}")
print(f"Batch size: {batch_size}")
print(f"Dropout: {dropout}")
print(f"Early Stopping: {use_early_stopping}")
if use_early_stopping:
    print(f"  - Patience: {patience} epochs")
    print(f"  - Min Delta: {min_delta}")
print("="*60)

##########################

# Random seed setup
seed = random.randint(0, 2**32 - 1)
print(f"Random seed: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Create dataset
print("\nLoading training data...")
data = dataset.TurbDataset(prop, shuffle=1, dataDir="../data/train/")
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print(f"Training batches: {len(trainLoader)}")

# Create validation dataset
print("\nLoading validation data...")
dataValidation = dataset.TurbDataset(
    prop, 
    mode=dataset.TurbDataset.TEST, 
    shuffle=0,
    dataDir="../data/train/", 
    dataDirTest="../data/validation/"
)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=False)
print(f"Validation batches: {len(valiLoader)}")

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Detect available GPU count
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPU(s):")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# Create network
netG = TurbNetG(channelExponent=expo, dropout=dropout)
print(f"\nNetwork parameters: {sum(p.numel() for p in netG.parameters() if p.requires_grad):,}")

# Load pretrained model (if specified)
if len(doLoad) > 0:
    print(f"Loading pretrained model: {doLoad}")
    netG.load_state_dict(torch.load(doLoad))
    print("Model loaded successfully")
else:
    print("Randomly initializing network weights")
    netG.apply(weights_init)

netG.to(device)

# Use DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    print(f"\nUsing DataParallel for training on {torch.cuda.device_count()} GPUs")
    netG = nn.DataParallel(netG)
else:
    print("\nUsing single GPU training")

# Define loss function and optimizer
criterionL1 = nn.L1Loss()
criterionL1.to(device)
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

# Calculate total epochs
epochs = int(iterations / len(trainLoader) + 0.5)
print(f"\nTotal training epochs: {epochs}")

# Learning rate decay scheduler
if decayLr:
    lrDecay = 0.1
    decayInterval = int(epochs / 3)
    scheduler = optim.lr_scheduler.StepLR(optimizerG, step_size=decayInterval, gamma=lrDecay)
    print(f"Learning rate decay: decay by {lrDecay} every {decayInterval} epochs")

##########################
# Training main loop
##########################

print("\n" + "="*60)
print("Starting training")
print("="*60 + "\n")

# Create output directories
utils.makeDirs(["../data", "../data/models", "../data/train_log"])

# Training history
train_losses = []
val_losses = []

# Early Stopping state
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 60)
    
    netG.train()
    epoch_loss = 0.0
    
    # Training loop
    for i, traindata in enumerate(trainLoader):
        inputs, targets = traindata
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizerG.zero_grad()
        outputs = netG(inputs)
        
        # Calculate loss
        lossL1 = criterionL1(outputs, targets)
        lossL1.backward()
        
        # Update parameters
        optimizerG.step()
        
        epoch_loss += lossL1.item()
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  Batch [{i+1}/{len(trainLoader)}], Loss: {lossL1.item():.6f}")
    
    # Average training loss
    avg_train_loss = epoch_loss / len(trainLoader)
    train_losses.append(avg_train_loss)
    print(f"\nTraining loss: {avg_train_loss:.6f}")
    
    # Validation
    netG.eval()
    val_loss = 0.0
    
    if len(valiLoader) > 0:
        with torch.no_grad():
            for i, valdata in enumerate(valiLoader):
                inputs, targets = valdata
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = netG(inputs)
                lossL1 = criterionL1(outputs, targets)
                val_loss += lossL1.item()
        
        avg_val_loss = val_loss / len(valiLoader)
        val_losses.append(avg_val_loss)
        print(f"Validation loss: {avg_val_loss:.6f}")
        
        # Early Stopping check
        if use_early_stopping:
            if avg_val_loss < best_val_loss - min_delta:
                # Validation loss improved
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save best model
                best_model_path = f"../data/models/{prefix}model_best.pth"
                torch.save(netG.state_dict(), best_model_path)
                print(f"  → Validation loss improved, saving best model")
            else:
                # Validation loss did not improve
                epochs_no_improve += 1
                print(f"  → Validation loss did not improve ({epochs_no_improve}/{patience})")
                
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered! Validation loss has not improved for {patience} epochs")
                    print(f"Best validation loss: {best_val_loss:.6f}")
                    early_stop = True
    else:
        print("Warning: Validation dataset is empty, skipping validation")
        val_losses.append(0.0)
    
    # Learning rate decay
    if decayLr:
        scheduler.step()
        current_lr = optimizerG.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
    
    # Save model (every 10 epochs)
    if (epoch + 1) % 10 == 0:
        model_path = f"../data/models/{prefix}model_epoch{epoch+1}.pth"
        torch.save(netG.state_dict(), model_path)
        print(f"Saving model: {model_path}")
    
    # Check if early stopping
    if early_stop:
        break

# Save final model
final_model_path = f"../data/models/{prefix}model_final.pth"
torch.save(netG.state_dict(), final_model_path)
print(f"\nFinal model saved: {final_model_path}")

# If early stopping was used, report best model
if use_early_stopping and best_val_loss < float('inf'):
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model: ../data/models/{prefix}model_best.pth")

# Save training history
if saveL1:
    history_file = f"../data/train_log/{prefix}training_history.txt"
    with open(history_file, 'w') as f:
        f.write("Epoch,TrainLoss,ValLoss\n")
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{train_loss},{val_loss}\n")
    print(f"Training history saved: {history_file}")

print("\n" + "="*60)
print("Training completed!")
print("="*60)
