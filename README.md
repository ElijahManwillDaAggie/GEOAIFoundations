# Lab 5: Accuracy Assessment for Flood Detection Models

This repository contains the complete solution for Lab 5 accuracy assessment assignment. The assignment involves testing Decision Tree (DT) and Random Forest (RF) models trained in previous labs on a new region (Mekong) and quantifying their performance using accuracy assessment metrics.

## 📁 Files Overview

### Main Scripts
- `lab5_complete_pipeline.py` - **Main script** that runs the complete pipeline
- `step1_preprocessing_inference.py` - Preprocessing and inference
- `step2_custom_accuracy.py` - Custom accuracy assessment functions
- `step3_sklearn_comparison.py` - Scikit-learn comparison
- `lab5_accuracy_assessment.py` - Comprehensive single-file solution

### Data Structure
```
Lab_05/
├── S1F11_Mekong/
│   ├── Label/           # Ground truth labels (30 .tif files)
│   └── S2/             # Sentinel-2 images (30 .tif files)
├── inference_results/  # Generated prediction maps
└── *.py               # Python scripts
```

## 🚀 Quick Start

### 1. Activate Environment
```bash
conda activate lab5_accuracy
```

### 2. Run Complete Pipeline
```bash
python lab5_complete_pipeline.py
```

### 3. Or Run Individual Steps
```bash
# Step 1: Preprocessing and Inference
python step1_preprocessing_inference.py

# Step 2: Custom Accuracy Assessment
python step2_custom_accuracy.py

# Step 3: Scikit-learn Comparison
python step3_sklearn_comparison.py
```

## 📋 Assignment Requirements

### Step 1: Preprocessing and Inference
- ✅ Replace -1 values in labels with 0 (cloudy pixels)
- ✅ Create chips for model input
- ✅ Run inference with DT and RF models
- ✅ Save prediction maps as GeoTIFF
- ✅ Create zip file for Canvas submission

### Step 2: Custom Accuracy Assessment
- ✅ Implement custom OA, precision, recall, F1 functions
- ✅ Calculate metrics for all 30 images
- ✅ Save results in required format
- ✅ Calculate average statistics

### Step 3: Scikit-learn Comparison
- ✅ Use scikit-learn accuracy assessment functions
- ✅ Compare with custom functions
- ✅ Determine which model performs better
- ✅ Generate comparison report

## 🔧 Configuration

### Model Paths
Update the model paths in the scripts:
```python
# Replace with your actual model paths
model_dt_path = "path/to/your/dt_model.pkl"
model_rf_path = "path/to/your/rf_model.pkl"
```

### Data Paths
The scripts are configured for the default structure:
```python
data_dir = "/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong"
```

## 📊 Output Files

### Generated Files
- `inference_results.zip` - **Submit to Canvas** (Step 1)
- `custom_accuracy_results.txt` - **Submit to Canvas** (Step 2)
- `sklearn_accuracy_results.txt` - **Submit to Canvas** (Step 3)
- `lab5_accuracy_assessment_report.txt` - Summary report

### Output Format
```
Image_Name_Model OA Precision Recall F1
Mekong_1111068_DT 0.8500 0.8200 0.7800 0.8000
Mekong_1111068_RF 0.8700 0.8400 0.8000 0.8200
...
Average_stat_DT 0.8600 0.8300 0.7900 0.8100
Average_stat_RF 0.8800 0.8500 0.8100 0.8300
```

## 🧪 Testing

### Test Environment
```bash
conda activate lab5_accuracy
python -c "
import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('✅ Environment ready!')
"
```

### Test Custom Functions
The scripts include built-in tests to verify:
- ✅ Custom accuracy assessment functions work correctly
- ✅ Agreement between custom and scikit-learn functions
- ✅ Proper handling of edge cases (zero division, etc.)

## 📝 Key Functions

### Custom Accuracy Assessment
```python
def custom_accuracy_assessment(ground_truth, prediction):
    # Calculate confusion matrix elements
    tp = np.sum((gt_valid == 1) & (pred_valid == 1))  # True Positives
    tn = np.sum((gt_valid == 0) & (pred_valid == 0))  # True Negatives
    fp = np.sum((gt_valid == 0) & (pred_valid == 1))  # False Positives
    fn = np.sum((gt_valid == 1) & (pred_valid == 0))  # False Negatives
    
    # Calculate metrics
    oa = (tp + tn) / total_pixels
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
```

### Scikit-learn Comparison
```python
def scikit_accuracy_assessment(ground_truth, prediction):
    oa = accuracy_score(gt_valid, pred_valid)
    precision = precision_score(gt_valid, pred_valid, average='binary', zero_division=0)
    recall = recall_score(gt_valid, pred_valid, average='binary', zero_division=0)
    f1 = f1_score(gt_valid, pred_valid, average='binary', zero_division=0)
```

## 🐛 Troubleshooting

### Common Issues

1. **Missing S2 Images**: Ensure Sentinel-2 images are downloaded from iCloud
2. **Model Paths**: Update model paths to your actual trained models
3. **Memory Issues**: Reduce chip size or process images individually
4. **File Permissions**: Ensure write permissions for output directories

### Error Messages
- `Warning: S2 file not found` - Download S2 images from iCloud
- `Error processing image` - Check file formats and paths
- `Model not found` - Update model paths in scripts

## 📚 Assignment Questions

### Questions to Answer
1. **Do custom and scikit-learn results agree?** ✅ Yes, they should match exactly
2. **Which model performs better?** Compare average metrics across all images
3. **Are results consistent across images?** Check standard deviation of metrics
4. **What are the main sources of error?** Analyze confusion matrix elements

### Expected Results
- Custom and scikit-learn functions should agree perfectly
- One model should generally outperform the other
- Results may vary across different images
- Overall accuracy should be reasonable for flood detection

## 🎯 Submission Checklist

### Canvas Submission
- [ ] `inference_results.zip` (Step 1)
- [ ] `custom_accuracy_results.txt` (Step 2)
- [ ] `sklearn_accuracy_results.txt` (Step 3)
- [ ] Document answering comparison questions

### GitHub Submission
- [ ] Push all code to GitHub repository
- [ ] Include README with instructions
- [ ] Document any modifications made

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify your conda environment is activated
3. Ensure all data files are present
4. Check file paths and permissions

---

**Good luck with your Lab 5 assignment! 🚀**
