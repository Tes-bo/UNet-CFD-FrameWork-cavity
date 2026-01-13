################
#
# Deep Flow Prediction for Cavity Flow - Data Generation Script
#
# Generate cavity flow training data through OpenFOAM CFD simulations
# Vary lid velocity and Reynolds number to generate different flow field samples
#
# Author: Tesbo
#
################

import os, math, sys, random
import numpy as np
import utils

# OpenFOAM-v2412 environment setup
OPENFOAM_BASHRC = "~/OpenFOAM/OpenFOAM-v2412/etc/bashrc"

def run_with_openfoam_env(command):
    """Execute command with OpenFOAM environment"""
    full_command = f"bash -c 'source {OPENFOAM_BASHRC} && {command}'"
    return os.system(full_command)

# Data generation configuration parameters
samples = 200  # Number of datasets to generate

# Lid velocity range (m/s)
lid_velocity_min = 0.1
lid_velocity_max = 2.0

# Reynolds number range (controlled by adjusting viscosity)
reynolds_min = 10
reynolds_max = 1000

# Cavity size (fixed)
cavity_length = 0.1  # meters

# Dataset split ratios
train_ratio = 0.7        # Training set: 70%
validation_ratio = 0.15  # Validation set: 15%
test_ratio = 0.15        # Test set: 15%

# Directory path settings
train_dir = "./train/"
validation_dir = "./validation/"
test_dir = "./test/"
openfoam_case_dir = "../cavity/"  # OpenFOAM case directory

# Random seed
seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Random Seed: {}".format(seed))

def get_dataset_dir(sample_index, total_samples):
    """
    Determine which dataset directory the sample should go to based on sample index
    """
    train_split = int(total_samples * train_ratio)
    validation_split = int(total_samples * (train_ratio + validation_ratio))
    
    if sample_index < train_split:
        return train_dir
    elif sample_index < validation_split:
        return validation_dir
    else:
        return test_dir

def prepare_case(lid_velocity, kinematic_viscosity):
    """
    Prepare OpenFOAM case
    Parameters:
        lid_velocity - lid velocity (m/s)
        kinematic_viscosity - kinematic viscosity (m^2/s)
    Returns: 0 for success, -1 for failure
    """
    # Modify lid velocity boundary condition
    try:
        with open("0/U", "r") as f:
            content = f.read()
        
        # Replace lid velocity value
        import re
        content = re.sub(
            r'movingWall\s*\{\s*type\s+fixedValue;\s*value\s+uniform\s+\([^)]+\);',
            f'movingWall\n    {{\n        type            fixedValue;\n        value           uniform ({lid_velocity} 0 0);',
            content
        )
        
        with open("0/U", "w") as f:
            f.write(content)
    except Exception as e:
        print(f"\tError: Unable to modify velocity boundary condition: {e}")
        return -1
    
    # Modify kinematic viscosity
    try:
        transport_props = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

nu              {kinematic_viscosity};

// ************************************************************************* //
"""
        with open("constant/transportProperties", "w") as f:
            f.write(transport_props)
    except Exception as e:
        print(f"\tError: Unable to modify transport properties: {e}")
        return -1
    
    return 0

def runSim():
    """
    Run CFD simulation
    Returns: 0 for success, -1 for failure
    """
    # Clean old results
    result = run_with_openfoam_env("foamListTimes -rm")
    
    # Generate mesh
    result = run_with_openfoam_env("blockMesh > blockMesh.log 2>&1")
    if result != 0:
        print("\tError: Mesh generation failed! Check blockMesh.log")
        return -1
    
    # Run icoFoam solver
    result = run_with_openfoam_env("icoFoam > icoFoam.log 2>&1")
    if result != 0:
        print("\tError: Simulation failed! Check icoFoam.log")
        return -1
    
    return 0

def sample_fields():
    """
    Sample flow fields on regular grid
    Returns: 0 for success, -1 for failure
    """
    # Create sampling configuration file
    sample_dict = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2412                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      sampleDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

type sets;
libs (sampling);

setFormat raw;

interpolationScheme cellPoint;

fields (p U);

sets
(
    internalCloud
    {
        type        cloud;
        axis        xyz;
        points      
        (
"""
    
    # Generate 128x128 sampling point grid
    res = 128
    L = 0.1  # Cavity size
    points = []
    for j in range(res):
        for i in range(res):
            x = (i + 0.5) / res * L
            y = (j + 0.5) / res * L
            z = 0.0005  # z-coordinate for 2D mesh (middle position, mesh z range is 0-0.001)
            points.append(f"            ({x} {y} {z})\n")
    
    sample_dict += "".join(points)
    sample_dict += """        );
    }
);

// ************************************************************************* //
"""
    
    try:
        with open("system/sampleDict", "w") as f:
            f.write(sample_dict)
    except Exception as e:
        print(f"\tError: Unable to create sampling configuration: {e}")
        return -1
    
    # Run postProcess sampling
    result = run_with_openfoam_env("postProcess -func sampleDict -latestTime > sample.log 2>&1")
    if result != 0:
        print("\tError: Field sampling failed! Check sample.log")
        return -1
    
    return 0

def outputProcessing(basename, lid_velocity, reynolds, dataDir, openfoam_dir, res=128, imageIndex=0):
    """
    Post-processing: Convert CFD results to deep learning training data
    
    Output data format (6 channels):
    [0] Lid velocity X component (input)
    [1] Lid velocity Y component (input, always 0)
    [2] Reynolds number normalized value (input)
    [3] Pressure field (output)
    [4] Velocity X component (output)
    [5] Velocity Y component (output)
    """
    npOutput = np.zeros((6, res, res))
    
    # Find latest timestep sampling data (in OpenFOAM case directory)
    import glob
    sample_pattern = os.path.join(openfoam_dir, "postProcessing/sampleDict/*/internalCloud_p_U.xy")
    sample_files = glob.glob(sample_pattern)
    if not sample_files:
        print(f"\tError: Cannot find sampling data file")
        print(f"\tSearch path: {sample_pattern}")
        # Try to list postProcessing directory contents for debugging
        post_dir = os.path.join(openfoam_dir, "postProcessing")
        if os.path.exists(post_dir):
            print(f"\t{post_dir} directory contents:")
            for root, dirs, files in os.walk(post_dir):
                for file in files:
                    print(f"\t  {os.path.join(root, file)}")
        return -1
    
    pfile = sorted(sample_files)[-1]  # Use the latest
    print(f"\tUsing sampling file: {os.path.basename(pfile)}")
    
    try:
        # Read data: x y z p Ux Uy Uz
        data = np.loadtxt(pfile)
    except Exception as e:
        print(f"\tError: Unable to read sampling data: {e}")
        return -1
    
    # Fill data
    curIndex = 0
    for j in range(res):
        for i in range(res):
            if curIndex < len(data):
                # Input channels: lid velocity and Reynolds number
                npOutput[0][i][j] = lid_velocity / lid_velocity_max  # Normalized
                npOutput[1][i][j] = 0  # Y velocity always 0
                npOutput[2][i][j] = np.log10(reynolds) / np.log10(reynolds_max)  # Log normalized
                
                # Output channels: pressure and velocity
                npOutput[3][i][j] = data[curIndex][3]  # Pressure
                npOutput[4][i][j] = data[curIndex][4]  # Ux
                npOutput[5][i][j] = data[curIndex][5]  # Uy
                curIndex += 1
    
    # Save visualization images
    try:
        utils.saveAsImage(f'data_pictures/pressure_{imageIndex:04d}.png', npOutput[3])
        utils.saveAsImage(f'data_pictures/velX_{imageIndex:04d}.png', npOutput[4])
        utils.saveAsImage(f'data_pictures/velY_{imageIndex:04d}.png', npOutput[5])
        utils.saveAsImage(f'data_pictures/inputVel_{imageIndex:04d}.png', npOutput[0])
        utils.saveAsImage(f'data_pictures/inputRe_{imageIndex:04d}.png', npOutput[2])
    except Exception as e:
        print(f"\tWarning: Failed to save images: {e}")
    
    # Save data
    fileName = dataDir + f"{basename}_Re{int(reynolds)}_V{int(lid_velocity*1000)}"
    print(f"\tSaving data to: {fileName}.npz")
    
    try:
        np.savez_compressed(fileName, a=npOutput)
    except Exception as e:
        print(f"\tError: Failed to save data: {e}")
        return -1
    
    return 0

# Create necessary directories
utils.makeDirs(["./data_pictures", "./train", "./validation", "./test"])

# Print configuration information
print("\n" + "="*60)
print("Cavity Flow Deep Learning Data Generation")
print("="*60)
print(f"Total samples: {samples}")
print(f"Lid velocity range: {lid_velocity_min} - {lid_velocity_max} m/s")
print(f"Reynolds number range: {reynolds_min} - {reynolds_max}")
print(f"Cavity size: {cavity_length} m")
print("\nDataset split:")
print(f"  Training set: {train_ratio*100:.0f}% ({int(samples * train_ratio)} samples)")
print(f"  Validation set: {validation_ratio*100:.0f}% ({int(samples * validation_ratio)} samples)")
print(f"  Test set: {test_ratio*100:.0f}% ({samples - int(samples * train_ratio) - int(samples * validation_ratio)} samples)")
print("="*60 + "\n")

# Main loop
for n in range(samples):
    current_dataset_dir = get_dataset_dir(n, samples)
    dataset_type = "Training" if current_dataset_dir == train_dir else (
        "Validation" if current_dataset_dir == validation_dir else "Test"
    )
    
    print(f"\nSample {n+1}/{samples} ({dataset_type}):")
    
    # Generate random parameters
    lid_velocity = np.random.uniform(lid_velocity_min, lid_velocity_max)
    reynolds = np.random.uniform(reynolds_min, reynolds_max)
    
    # Calculate kinematic viscosity: Re = U * L / nu
    kinematic_viscosity = lid_velocity * cavity_length / reynolds
    
    print(f"\tLid velocity: {lid_velocity:.4f} m/s")
    print(f"\tReynolds number: {reynolds:.1f}")
    print(f"\tKinematic viscosity: {kinematic_viscosity:.6e} m^2/s")
    
    # Save current working directory
    original_dir = os.getcwd()
    
    # Switch to OpenFOAM case directory
    os.chdir(openfoam_case_dir)
    
    # Prepare case
    if prepare_case(lid_velocity, kinematic_viscosity) != 0:
        print("\tCase preparation failed, program terminated!")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Run simulation
    if runSim() != 0:
        print("\tCFD simulation failed, program terminated!")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Sample flow field
    if sample_fields() != 0:
        print("\tFlow field sampling failed, program terminated!")
        os.chdir(original_dir)
        sys.exit(1)
    
    # Return to original directory (data/ directory) for post-processing
    os.chdir(original_dir)
    if outputProcessing(f"cavity", lid_velocity, reynolds, current_dataset_dir, openfoam_case_dir, imageIndex=n) != 0:
        print("\tPost-processing failed, program terminated!")
        sys.exit(1)
    
    print(f"\t✓ Completed (saved to {dataset_type})")

print("\n" + "="*60)
print("Data generation completed!")
print("="*60)
