# Assignment 2: Array Transformation Analysis

## Overview
This document analyzes the array transformations performed on remote sensing imagery data to prepare it for machine learning algorithms. The transformations convert 3D image arrays and 2D label arrays into the format required by scikit-learn.

## Image Array Transformations

### 1. Original Image Array (3D)
- **Shape**: (3, 3, 3) - (bands, rows, columns)
- **Dimensions**: 
  - Dimension 0 (3): Number of spectral bands (Red, Green, Blue)
  - Dimension 1 (3): Number of rows (height in pixels)
  - Dimension 2 (3): Number of columns (width in pixels)
- **Data Organization**: Bands are organized first, then spatial dimensions
- **Example**: Each pixel has 3 spectral values (RGB), and we have a 3x3 pixel image

### 2. Reshape Operation
- **From**: (3, 3, 3) → **To**: (3, 9)
- **Purpose**: Flattens the spatial dimensions while keeping bands separate
- **Result**: Each row represents one spectral band, each column represents one pixel
- **Interpretation**: 
  - Row 0: All Red band values
  - Row 1: All Green band values  
  - Row 2: All Blue band values
  - Columns: 9 pixels total (3x3 grid flattened)

### 3. Transpose Operation
- **From**: (3, 9) → **To**: (9, 3)
- **Purpose**: Switches rows and columns to prepare for machine learning
- **Result**: Each row represents one pixel, each column represents one spectral band
- **Interpretation**:
  - Each row: One pixel with its 3 spectral values
  - Column 0: Red values for all pixels
  - Column 1: Green values for all pixels
  - Column 2: Blue values for all pixels

### 4. Vertical Stacking (vstack)
- **Purpose**: Combines multiple images into one training dataset
- **From**: Two (9, 3) arrays → **To**: One (18, 3) array
- **Result**: All pixels from both images in one array
- **Machine Learning Format**: (n_samples, n_features)
  - n_samples: 18 pixels total
  - n_features: 3 spectral bands per pixel

## Label Array Transformations

### 1. Original Label Array (2D)
- **Shape**: (3, 3) - (rows, columns)
- **Dimensions**:
  - Dimension 0 (3): Number of rows
  - Dimension 1 (3): Number of columns
- **Data Organization**: Spatial arrangement of class labels
- **Values**: Class identifiers (e.g., 3587, 3661, 3116, 3187, 3051, 3094)

### 2. Flatten Operation
- **From**: (3, 3) → **To**: (9,)
- **Purpose**: Converts 2D spatial array to 1D vector
- **Result**: All pixel labels in a single row
- **Interpretation**: Each element corresponds to one pixel's class label

### 3. Horizontal Stacking (hstack)
- **Purpose**: Combines multiple label arrays into one training dataset
- **From**: Two (9,) arrays → **To**: One (18,) array
- **Result**: All pixel labels from both images in one vector
- **Machine Learning Format**: (n_samples,)
  - n_samples: 18 labels total
  - Each label corresponds to one pixel in the feature array

## Key Insights

### Array Shape Progression
1. **Image**: (3, 3, 3) → (3, 9) → (9, 3) → (18, 3)
2. **Labels**: (3, 3) → (9,) → (18,)

### Machine Learning Requirements
- **Features (X)**: Shape (n_samples, n_features) = (18, 3)
- **Labels (y)**: Shape (n_samples,) = (18,)
- **Correspondence**: Each row in X corresponds to one element in y

### Spatial Information Loss
- The transformations flatten the spatial relationships between pixels
- Each pixel is treated as an independent sample
- Spatial context is lost but spectral information is preserved

### Data Integrity
- The transformations maintain the correspondence between pixels and their labels
- No data is lost during the reshaping process
- The order of pixels is preserved through the transformations

## Conclusion
These array transformations are essential for preparing remote sensing data for machine learning. They convert spatially-organized image data into the tabular format required by scikit-learn, where each row represents one pixel and each column represents one spectral band. The transformations ensure that the correspondence between image pixels and their labels is maintained throughout the process.
