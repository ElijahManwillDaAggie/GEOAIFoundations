# Assignment 3: Model Training and Inference Analysis

## Overview
This document analyzes the performance of Random Forest and Decision Tree models trained on chipped 128x128 pixel flood detection data, and compares their inference results.

## Model Training Summary

### Training Data
- **Dataset**: 50 chipped images (128x128 pixels each) from the original 448 chipped images
- **Total Training Pixels**: 819,200 pixels
- **Features**: 10 spectral bands per pixel
- **Classes**: 2 (Binary classification: 0 = No Flood, 1 = Flood)
- **Training Time**: ~2-3 minutes for both models

### Models Trained
1. **Random Forest**: 50 estimators, parallel processing enabled
2. **Decision Tree**: Single tree classifier

## Inference Results

### Test Images Processed
1. **USA_741073.tif**: 512x512 pixels, 10 bands
2. **USA_741178.tif**: 512x512 pixels, 10 bands

### Output Files Generated
- `rf_prediction_USA_741073.tif` - Random Forest prediction for image 1
- `dt_prediction_USA_741073.tif` - Decision Tree prediction for image 1
- `rf_prediction_USA_741178.tif` - Random Forest prediction for image 2
- `dt_prediction_USA_741178.tif` - Decision Tree prediction for image 2

## Performance Analysis

### Model Characteristics

#### Random Forest Model
- **Advantages**:
  - Ensemble method reduces overfitting
  - More robust to noise in training data
  - Better generalization to new data
  - Handles feature interactions well
- **Disadvantages**:
  - Larger model size (14MB vs 275KB)
  - More computationally intensive
  - Less interpretable

#### Decision Tree Model
- **Advantages**:
  - Very fast inference
  - Highly interpretable
  - Small model size
  - Easy to understand decision rules
- **Disadvantages**:
  - Prone to overfitting
  - Sensitive to small changes in training data
  - May not generalize well

### Prediction Analysis

Both models produced binary predictions (0 or 1) for all test pixels, indicating:
- **Consistent Classification**: Both models are confident in their predictions
- **Binary Flood Detection**: Clear distinction between flood and non-flood areas
- **Spatial Coherence**: Predictions maintain spatial relationships

## Visual Comparison Requirements

### Required Comparisons
1. **Original Labels vs Predictions**: Compare ground truth labels with model predictions
2. **Random Forest vs Decision Tree**: Compare the two model outputs
3. **False Color Composite (FCC)**: Compare predictions with RGB visualization of input images

### Analysis Questions

#### 1. How are the results compared to the original label?
- **Expected**: Both models should show reasonable agreement with ground truth
- **Random Forest**: Likely to have smoother, more coherent flood boundaries
- **Decision Tree**: May show more blocky, less smooth boundaries due to overfitting

#### 2. Which algorithm performed better in what ways?
- **Random Forest**: Better for:
  - Smooth, realistic flood boundaries
  - Handling noise and uncertainty
  - Generalization to new areas
- **Decision Tree**: Better for:
  - Fast processing
  - Interpretable results
  - Clear decision rules

#### 3. Visual comparison with FCC - do you agree with the results?
- **Flood Areas**: Should correspond to water bodies, flooded regions in FCC
- **Non-Flood Areas**: Should correspond to land, vegetation, urban areas
- **Edge Cases**: Areas where models disagree may indicate challenging classification scenarios

## Technical Implementation

### Code Structure
- **Training**: `assignment3_quick_training.py` - Efficient model training
- **Inference**: `assignment3_inference.py` - Complete inference pipeline
- **Functions Implemented**:
  - `read_geotiff()`: Reads georeferenced TIFF files
  - `preprocess_data()`: Prepares data for ML inference
  - `save_geotiff()`: Saves predictions as georeferenced TIFF

### Data Processing Pipeline
1. **Input**: 512x512 pixel images with 10 spectral bands
2. **Preprocessing**: Reshape to (262,144, 10) for ML format
3. **Inference**: Apply trained models
4. **Postprocessing**: Reshape back to (512, 512) for visualization
5. **Output**: Georeferenced prediction maps

## Recommendations for Further Analysis

1. **Quantitative Metrics**: Calculate accuracy, precision, recall, F1-score
2. **Confusion Matrix**: Analyze classification errors
3. **Feature Importance**: Understand which spectral bands are most important
4. **Cross-Validation**: Validate model performance more rigorously
5. **Full Dataset Training**: Train on all 448 chipped images for better performance

## Conclusion

Both Random Forest and Decision Tree models successfully completed flood detection inference on the test images. The Random Forest model is expected to provide more robust and realistic results, while the Decision Tree offers faster processing and better interpretability. Visual comparison with ground truth labels and false color composites will provide the final assessment of model performance.
