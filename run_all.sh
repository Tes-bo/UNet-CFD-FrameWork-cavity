#!/bin/bash
################
#
# Quick Start Script - Run the complete deep learning pipeline
#
# Author: Tesbo
#
################

echo "========================================"
echo "Cavity Flow Deep Learning Complete Pipeline"
echo "========================================"

# Check OpenFOAM environment
if [ -z "$WM_PROJECT" ]; then
    echo "Error: OpenFOAM environment not loaded!"
    echo "Please run: source ~/OpenFOAM/OpenFOAM-v2412/etc/bashrc"
    exit 1
fi

echo "✓ OpenFOAM environment loaded"

# Step 1: Generate data (optional, can skip if data already exists)
read -p "Generate new data? (y/n) [n]: " generate_data
generate_data=${generate_data:-n}

if [ "$generate_data" = "y" ] || [ "$generate_data" = "Y" ]; then
    echo ""
    echo "Step 1: Generating training data"
    echo "----------------------------------------"
    cd data
    python3 dataGen.py
    if [ $? -ne 0 ]; then
        echo "Error: Data generation failed!"
        exit 1
    fi
    cd ..
    echo "✓ Data generation completed"
else
    echo "Skipping data generation step"
fi

# Check if training data exists
if [ ! "$(ls -A data/train/*.npz 2>/dev/null)" ]; then
    echo "Error: Training data not found! Please generate data first."
    exit 1
fi

# Step 2: Train model
echo ""
echo "Step 2: Training deep learning model"
echo "----------------------------------------"
read -p "Start training? (y/n) [y]: " start_train
start_train=${start_train:-y}

if [ "$start_train" = "y" ] || [ "$start_train" = "Y" ]; then
    cd train
    python3 runTrain.py cavity_exp_
    if [ $? -ne 0 ]; then
        echo "Error: Training failed!"
        exit 1
    fi
    cd ..
    echo "✓ Training completed"
else
    echo "Skipping training step"
fi

# Step 3: Test/Predict
echo ""
echo "Step 3: Making predictions using trained model"
echo "----------------------------------------"

# Check if trained model exists
if [ ! -f "data/models/cavity_exp_model_final.pth" ] && [ ! -f "data/models/model_final.pth" ]; then
    echo "Error: Trained model not found!"
    exit 1
fi

read -p "Run prediction test? (y/n) [y]: " start_test
start_test=${start_test:-y}

if [ "$start_test" = "y" ] || [ "$start_test" = "Y" ]; then
    cd train
    
    # Find the latest model
    if [ -f "../data/models/cavity_exp_model_final.pth" ]; then
        python3 runTest.py ../data/models/cavity_exp_model_final.pth
    else
        python3 runTest.py ../data/models/model_final.pth
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Prediction failed!"
        exit 1
    fi
    cd ..
    echo "✓ Prediction completed"
else
    echo "Skipping prediction step"
fi

echo ""
echo "========================================"
echo "All steps completed!"
echo "========================================"
echo "Results location:"
echo "  - Training data: data/train/, data/validation/, data/test/"
echo "  - Trained models: data/models/"
echo "  - Prediction results: data/test_results/"
echo "  - Visualization images: data/data_pictures/"
echo "========================================"
