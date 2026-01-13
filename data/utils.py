################
#
# Deep Flow Prediction for Cavity Flow - Utility Functions Module
#
# Provide data processing, image saving and other auxiliary functions
#
# Author: Tesbo
#
################

import os
import numpy as np
from PIL import Image
from matplotlib import cm

def makeDirs(directoryList):
    """
    Create all directories in the directory list (if they don't exist)
    Parameters: directoryList - List of directory paths
    """
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def imageOut(filename, outputs_param, targets_param, saveTargets=False):
    """
    Save comparison images of prediction results and target values
    Parameters:
        filename - Output filename prefix
        outputs_param - Network prediction output in (3, H, W) format
        targets_param - Ground truth target values in (3, H, W) format
        saveTargets - Whether to save target value images
    """
    outputs = np.copy(outputs_param)
    targets = np.copy(targets_param)

    for i in range(3):
        # Normalize to [0,1] range
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        outputs[i] -= min_value
        targets[i] -= min_value
        max_value -= min_value
        outputs[i] /= max_value
        targets[i] /= max_value

        # Set file suffix based on channel type
        suffix = ""
        if i==0:
            suffix = "_pressure"
        elif i==1:
            suffix = "_velX"
        else:
            suffix = "_velY"

        # Apply same transformation as saveAsImage (transpose and flip vertically)
        outputs_viz = np.flipud(outputs[i].transpose())
        targets_viz = np.flipud(targets[i].transpose())

        # Save prediction results
        im = Image.fromarray(cm.magma(outputs_viz, bytes=True))
        im = im.resize((512,512))
        im.save(filename + suffix + "_pred.png")

        # Optional: save target values
        if saveTargets:
            im = Image.fromarray(cm.magma(targets_viz, bytes=True))
            im = im.resize((512,512))
            im.save(filename + suffix + "_target.png")

def saveAsImage(filename, field_param):
    """
    Save flow field data as a color image
    Parameters:
        filename - Output image filename
        field_param - Flow field data array
    """
    field = np.copy(field_param)
    field = np.flipud(field.transpose())

    # Normalize to [0,1]
    min_value = np.min(field)
    max_value = np.max(field)
    field -= min_value
    max_value -= min_value
    if max_value > 0:
        field /= max_value

    # Save using magma colormap
    im = Image.fromarray(cm.magma(field, bytes=True))
    im = im.resize((512, 512))
    im.save(filename)
