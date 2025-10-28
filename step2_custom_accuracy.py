#!/usr/bin/env python3
"""
Lab 5 - Step 2: Custom Accuracy Assessment Functions
This script implements custom accuracy assessment functions to calculate
OA, precision, recall, and F1 for flood detection models.
"""

import numpy as np
import rasterio
from pathlib import Path
import os

def custom_accuracy_assessment(ground_truth, prediction):
    """
    Custom accuracy assessment functions
    
    Args:
        ground_truth: Ground truth array (2D)
        prediction: Prediction array (2D)
    
    Returns:
        dict: Dictionary containing OA, precision, recall, F1
    """
    # Flatten arrays
    gt_flat = ground_truth.flatten()
    pred_flat = prediction.flatten()
    
    # Remove any remaining invalid pixels (if any)
    valid_mask = (gt_flat != -1) & (pred_flat != -1)
    gt_valid = gt_flat[valid_mask]
    pred_valid = pred_flat[valid_mask]
    
    # Calculate confusion matrix elements
    tp = np.sum((gt_valid == 1) & (pred_valid == 1))  # True Positives
    tn = np.sum((gt_valid == 0) & (pred_valid == 0))  # True Negatives
    fp = np.sum((gt_valid == 0) & (pred_valid == 1))  # False Positives
    fn = np.sum((gt_valid == 1) & (pred_valid == 0))  # False Negatives
    
    # Calculate metrics
    total_pixels = len(gt_valid)
    oa = (tp + tn) / total_pixels if total_pixels > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'OA': oa,
        'precision': precision,
        'recall': recall,
        'F1': f1,
        'confusion_matrix': {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    }

def load_prediction_and_ground_truth(img_name, data_dir, inference_dir="inference_results"):
    """
    Load prediction and ground truth for a specific image
    
    Args:
        img_name: Name of the image
        data_dir: Path to data directory
        inference_dir: Path to inference results directory
    
    Returns:
        tuple: (ground_truth, dt_prediction, rf_prediction)
    """
    data_path = Path(data_dir)
    label_dir = data_path / "Label"
    inference_path = Path(inference_dir)
    
    # Load ground truth
    label_path = label_dir / f"{img_name}.tif"
    with rasterio.open(label_path) as src:
        ground_truth = src.read(1)
        # Replace -1 values with 0 (preprocessing)
        ground_truth[ground_truth == -1] = 0
    
    # Load predictions
    dt_path = inference_path / f"{img_name}_DT_prediction.tif"
    rf_path = inference_path / f"{img_name}_RF_prediction.tif"
    
    dt_prediction = None
    rf_prediction = None
    
    if dt_path.exists():
        with rasterio.open(dt_path) as src:
            dt_prediction = src.read(1)
    
    if rf_path.exists():
        with rasterio.open(rf_path) as src:
            rf_prediction = src.read(1)
    
    return ground_truth, dt_prediction, rf_prediction

def calculate_metrics_for_all_images(data_dir, inference_dir="inference_results"):
    """
    Calculate accuracy metrics for all images
    
    Args:
        data_dir: Path to data directory
        inference_dir: Path to inference results directory
    
    Returns:
        dict: Results for all images and models
    """
    data_path = Path(data_dir)
    label_dir = data_path / "Label"
    
    # Get all image names
    label_files = list(label_dir.glob("*.tif"))
    image_names = [f.stem for f in label_files]
    
    results = {}
    
    print(f"Calculating accuracy metrics for {len(image_names)} images...")
    
    for img_name in image_names:
        print(f"Processing {img_name}...")
        
        try:
            ground_truth, dt_prediction, rf_prediction = load_prediction_and_ground_truth(
                img_name, data_dir, inference_dir
            )
            
            img_results = {}
            
            # Calculate metrics for DT model
            if dt_prediction is not None:
                dt_metrics = custom_accuracy_assessment(ground_truth, dt_prediction)
                img_results['DT'] = dt_metrics
            else:
                print(f"Warning: DT prediction not found for {img_name}")
            
            # Calculate metrics for RF model
            if rf_prediction is not None:
                rf_metrics = custom_accuracy_assessment(ground_truth, rf_prediction)
                img_results['RF'] = rf_metrics
            else:
                print(f"Warning: RF prediction not found for {img_name}")
            
            results[img_name] = img_results
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    return results

def save_results_to_txt(results, output_file="custom_accuracy_results.txt"):
    """
    Save results to text file in the required format
    
    Args:
        results: Dictionary containing results for all images
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        # Calculate averages
        dt_oa_values = []
        dt_precision_values = []
        dt_recall_values = []
        dt_f1_values = []
        
        rf_oa_values = []
        rf_precision_values = []
        rf_recall_values = []
        rf_f1_values = []
        
        # Write results for each image
        for img_name, img_results in results.items():
            for model_name, metrics in img_results.items():
                f.write(f"{img_name}_{model_name} {metrics['OA']:.4f} "
                       f"{metrics['precision']:.4f} {metrics['recall']:.4f} "
                       f"{metrics['F1']:.4f}\n")
                
                # Collect values for averaging
                if model_name == 'DT':
                    dt_oa_values.append(metrics['OA'])
                    dt_precision_values.append(metrics['precision'])
                    dt_recall_values.append(metrics['recall'])
                    dt_f1_values.append(metrics['F1'])
                elif model_name == 'RF':
                    rf_oa_values.append(metrics['OA'])
                    rf_precision_values.append(metrics['precision'])
                    rf_recall_values.append(metrics['recall'])
                    rf_f1_values.append(metrics['F1'])
        
        # Write averages for DT
        if dt_oa_values:
            avg_dt_oa = np.mean(dt_oa_values)
            avg_dt_precision = np.mean(dt_precision_values)
            avg_dt_recall = np.mean(dt_recall_values)
            avg_dt_f1 = np.mean(dt_f1_values)
            
            f.write(f"Average_stat_DT {avg_dt_oa:.4f} {avg_dt_precision:.4f} "
                   f"{avg_dt_recall:.4f} {avg_dt_f1:.4f}\n")
        
        # Write averages for RF
        if rf_oa_values:
            avg_rf_oa = np.mean(rf_oa_values)
            avg_rf_precision = np.mean(rf_precision_values)
            avg_rf_recall = np.mean(rf_recall_values)
            avg_rf_f1 = np.mean(rf_f1_values)
            
            f.write(f"Average_stat_RF {avg_rf_oa:.4f} {avg_rf_precision:.4f} "
                   f"{avg_rf_recall:.4f} {avg_rf_f1:.4f}\n")
    
    print(f"Results saved to {output_file}")

def print_summary_statistics(results):
    """
    Print summary statistics
    
    Args:
        results: Dictionary containing results for all images
    """
    print("\n" + "="*60)
    print("CUSTOM ACCURACY ASSESSMENT SUMMARY")
    print("="*60)
    
    # Calculate averages for each model
    for model_name in ['DT', 'RF']:
        oa_values = []
        precision_values = []
        recall_values = []
        f1_values = []
        
        for img_name, img_results in results.items():
            if model_name in img_results:
                metrics = img_results[model_name]
                oa_values.append(metrics['OA'])
                precision_values.append(metrics['precision'])
                recall_values.append(metrics['recall'])
                f1_values.append(metrics['F1'])
        
        if oa_values:
            print(f"\n{model_name} Model:")
            print(f"  Overall Accuracy: {np.mean(oa_values):.4f} ± {np.std(oa_values):.4f}")
            print(f"  Precision:        {np.mean(precision_values):.4f} ± {np.std(precision_values):.4f}")
            print(f"  Recall:           {np.mean(recall_values):.4f} ± {np.std(recall_values):.4f}")
            print(f"  F1 Score:         {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}")
            print(f"  Number of images: {len(oa_values)}")

def main():
    """
    Main function for Step 2: Custom Accuracy Assessment
    """
    # Data directory
    data_dir = "/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong"
    inference_dir = "inference_results"
    
    print("Step 2: Custom Accuracy Assessment Functions")
    print("=" * 50)
    
    # Calculate metrics for all images
    results = calculate_metrics_for_all_images(data_dir, inference_dir)
    
    # Save results to text file
    save_results_to_txt(results)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    print(f"\nStep 2 completed!")
    print(f"Custom accuracy assessment results saved to custom_accuracy_results.txt")
    
    return results

if __name__ == "__main__":
    results = main()
