"""
Accuracy Assessment for Impervious Surface Classification
Weber River Watershed - Sentinel-2 Images

This script:
1. Loads ground truth labels and model predictions
2. Calculates accuracy metrics (OA, precision, recall, F1)
3. Creates confusion matrices
4. Generates accuracy assessment reports
5. Exports results for R statistical analysis
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 60)
print("Impervious Surface Classification - Accuracy Assessment")
print("=" * 60)

# Configuration
PREDICTIONS_FOLDER = 'predictions'
OUTPUT_FOLDER = 'accuracy_assessment'
RESULTS_CSV = os.path.join(OUTPUT_FOLDER, 'accuracy_results.csv')
RESULTS_SUMMARY = os.path.join(OUTPUT_FOLDER, 'accuracy_summary.txt')

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def read_geotiff(tiff_file_path):
    """
    Read a georeferenced TIFF file using GDAL
    
    Returns:
        data: numpy array
        transform: geotransform
        crs: coordinate reference system
    """
    dataset = gdal.Open(tiff_file_path)
    
    if dataset is None:
        raise ValueError(f"Could not open file: {tiff_file_path}")
    
    # Read first band (assuming single band for predictions/labels)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    
    transform = dataset.GetGeoTransform()
    crs = dataset.GetProjection()
    
    dataset = None
    
    return data, transform, crs

def calculate_metrics(ground_truth, prediction, model_name="Model"):
    """
    Calculate accuracy metrics from ground truth and predictions
    
    Args:
        ground_truth: numpy array with ground truth labels
        prediction: numpy array with predicted labels
        model_name: name of model for display
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Flatten arrays
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()
    
    # Remove any invalid pixels (if any)
    valid_mask = np.isfinite(gt_flat) & np.isfinite(pred_flat)
    gt_valid = gt_flat[valid_mask].astype(int)
    pred_valid = pred_flat[valid_mask].astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(gt_valid, pred_valid)
    
    # Calculate metrics
    accuracy = accuracy_score(gt_valid, pred_valid)
    precision = precision_score(gt_valid, pred_valid, average='binary', zero_division=0)
    recall = recall_score(gt_valid, pred_valid, average='binary', zero_division=0)
    f1 = f1_score(gt_valid, pred_valid, average='binary', zero_division=0)
    
    # Extract confusion matrix elements
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle case where only one class is present
        tn = fp = fn = tp = 0
        if len(np.unique(gt_valid)) == 1:
            if np.unique(gt_valid)[0] == 0:
                tn = cm[0, 0] if cm.shape[0] > 0 else 0
            else:
                tp = cm[0, 0] if cm.shape[0] > 0 else 0
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total_pixels': len(gt_valid)
    }
    
    return metrics

def print_metrics(metrics, model_name, image_name):
    """Print metrics in a formatted way"""
    print(f"\n  {model_name} - {image_name}")
    print(f"    Accuracy:    {metrics['accuracy']:.4f}")
    print(f"    Precision:   {metrics['precision']:.4f}")
    print(f"    Recall:      {metrics['recall']:.4f}")
    print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"    Specificity: {metrics['specificity']:.4f}")
    print(f"    F1-Score:    {metrics['f1_score']:.4f}")
    print(f"    Total Pixels: {metrics['total_pixels']:,}")
    print(f"    Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                Non-Imp  Imperv")
    print(f"      Actual Non-Imp  {metrics['tn']:6d}  {metrics['fp']:6d}")
    print(f"              Imperv   {metrics['fn']:6d}  {metrics['tp']:6d}")

def assess_predictions(ground_truth_path, prediction_path, model_name, image_name):
    """
    Assess accuracy of predictions against ground truth
    
    Args:
        ground_truth_path: path to ground truth GeoTIFF
        prediction_path: path to prediction GeoTIFF
        model_name: name of model
        image_name: name of image
    
    Returns:
        dict: metrics dictionary
    """
    # Load ground truth and predictions
    gt_data, _, _ = read_geotiff(ground_truth_path)
    pred_data, _, _ = read_geotiff(prediction_path)
    
    # Check if dimensions match
    if gt_data.shape != pred_data.shape:
        print(f"  Warning: Shape mismatch for {image_name}")
        print(f"    Ground truth: {gt_data.shape}")
        print(f"    Prediction: {pred_data.shape}")
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(gt_data, pred_data, model_name)
    print_metrics(metrics, model_name, image_name)
    
    # Add metadata
    metrics['model_name'] = model_name
    metrics['image_name'] = image_name
    
    return metrics

def create_visualizations(all_metrics, output_folder):
    """Create visualization plots for accuracy assessment"""
    
    # Extract data for plotting
    models = []
    images = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for metric in all_metrics:
        models.append(metric['model_name'])
        images.append(metric['image_name'])
        accuracies.append(metric['accuracy'])
        precisions.append(metric['precision'])
        recalls.append(metric['recall'])
        f1_scores.append(metric['f1_score'])
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': models,
        'Image': images,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1_Score': f1_scores
    })
    
    # Plot 1: Comparison of metrics by model
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy comparison
    df_pivot = df.pivot(index='Image', columns='Model', values='Accuracy')
    df_pivot.plot(kind='bar', ax=axes[0, 0], rot=45)
    axes[0, 0].set_title('Accuracy by Model and Image')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(title='Model')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision comparison
    df_pivot = df.pivot(index='Image', columns='Model', values='Precision')
    df_pivot.plot(kind='bar', ax=axes[0, 1], rot=45)
    axes[0, 1].set_title('Precision by Model and Image')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend(title='Model')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Recall comparison
    df_pivot = df.pivot(index='Image', columns='Model', values='Recall')
    df_pivot.plot(kind='bar', ax=axes[1, 0], rot=45)
    axes[1, 0].set_title('Recall by Model and Image')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend(title='Model')
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1-Score comparison
    df_pivot = df.pivot(index='Image', columns='Model', values='F1_Score')
    df_pivot.plot(kind='bar', ax=axes[1, 1], rot=45)
    axes[1, 1].set_title('F1-Score by Model and Image')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend(title='Model')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\n  Saved: accuracy_comparison.png")
    
    # Plot 2: Confusion matrices
    n_metrics = len(all_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, metric in enumerate(all_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        cm = metric['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{metric['model_name']} - {metric['image_name']}")
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: confusion_matrices.png")

def main():
    """
    Main function for accuracy assessment
    
    Note: This script assumes you have ground truth labels.
    You can create ground truth by:
    1. Manually labeling sample areas in your images
    2. Using reference data (e.g., from other sources)
    3. Creating validation points using create_training_data.py
    """
    print("\nNote: This script requires ground truth labels for accuracy assessment.")
    print("If you don't have ground truth labels yet, you can:")
    print("  1. Create validation points using create_training_data.py")
    print("  2. Manually create ground truth maps")
    print("  3. Use reference data from other sources")
    print()
    
    # Check if predictions folder exists
    if not os.path.exists(PREDICTIONS_FOLDER):
        print(f"Error: Predictions folder not found: {PREDICTIONS_FOLDER}")
        print("Please run apply_models.py first to generate predictions.")
        return
    
    # Get list of prediction files
    prediction_files = sorted([f for f in os.listdir(PREDICTIONS_FOLDER) 
                              if f.endswith('.tif')])
    
    if not prediction_files:
        print(f"Error: No prediction files found in {PREDICTIONS_FOLDER}")
        print("Please run apply_models.py first to generate predictions.")
        return
    
    print(f"Found {len(prediction_files)} prediction files")
    
    # For demonstration, we'll show the structure
    # In practice, you need to provide ground truth labels
    print("\n" + "=" * 60)
    print("Accuracy Assessment Structure")
    print("=" * 60)
    print("\nTo use this script, you need:")
    print("  1. Ground truth labels (GeoTIFF files with same dimensions as predictions)")
    print("  2. Prediction files (already in predictions/ folder)")
    print("\nExample usage:")
    print("  assess_predictions(")
    print("      ground_truth_path='ground_truth/2024_labels.tif',")
    print("      prediction_path='predictions/rf_WeberRiver_S2_2024_July_prediction.tif',")
    print("      model_name='Random Forest',")
    print("      image_name='2024'")
    print("  )")
    
    # If you have ground truth, uncomment and modify the following:
    """
    all_metrics = []
    
    # Example: Assess each prediction
    for pred_file in prediction_files:
        # Extract year and model from filename
        # Format: rf_WeberRiver_S2_2024_July_prediction.tif
        parts = pred_file.replace('.tif', '').split('_')
        year = parts[3]  # Extract year
        model_abbrev = parts[0]  # 'rf' or 'dt'
        model_name = 'Random Forest' if model_abbrev == 'rf' else 'Decision Tree'
        
        # Construct ground truth path (you need to provide this)
        gt_path = f'ground_truth/{year}_labels.tif'
        pred_path = os.path.join(PREDICTIONS_FOLDER, pred_file)
        
        if os.path.exists(gt_path):
            metrics = assess_predictions(gt_path, pred_path, model_name, year)
            if metrics:
                all_metrics.append(metrics)
    
    # Save results to CSV
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        results_df.to_csv(RESULTS_CSV, index=False)
        print(f"\nResults saved to: {RESULTS_CSV}")
        
        # Create visualizations
        create_visualizations(all_metrics, OUTPUT_FOLDER)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(results_df.groupby('model_name')[['accuracy', 'precision', 'recall', 'f1_score']].mean())
    """
    
    print("\n" + "=" * 60)
    print("Accuracy Assessment Script Ready")
    print("=" * 60)
    print("\nThis script is ready to use once you have ground truth labels.")
    print("Modify the main() function to point to your ground truth files.")

if __name__ == "__main__":
    main()

