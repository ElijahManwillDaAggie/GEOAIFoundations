#!/usr/bin/env python3
"""
Lab 5 - Step 3: Scikit-learn Accuracy Assessment Comparison
This script compares custom accuracy assessment functions with scikit-learn
built-in functions to verify consistency.
"""

import numpy as np
import rasterio
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def scikit_accuracy_assessment(ground_truth, prediction):
    """
    Accuracy assessment using scikit-learn
    
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
    
    # Calculate metrics using scikit-learn
    oa = accuracy_score(gt_valid, pred_valid)
    precision = precision_score(gt_valid, pred_valid, average='binary', zero_division=0)
    recall = recall_score(gt_valid, pred_valid, average='binary', zero_division=0)
    f1 = f1_score(gt_valid, pred_valid, average='binary', zero_division=0)
    
    return {
        'OA': oa,
        'precision': precision,
        'recall': recall,
        'F1': f1
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

def calculate_sklearn_metrics_for_all_images(data_dir, inference_dir="inference_results"):
    """
    Calculate accuracy metrics using scikit-learn for all images
    
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
    
    print(f"Calculating scikit-learn accuracy metrics for {len(image_names)} images...")
    
    for img_name in image_names:
        print(f"Processing {img_name}...")
        
        try:
            ground_truth, dt_prediction, rf_prediction = load_prediction_and_ground_truth(
                img_name, data_dir, inference_dir
            )
            
            img_results = {}
            
            # Calculate metrics for DT model
            if dt_prediction is not None:
                dt_metrics = scikit_accuracy_assessment(ground_truth, dt_prediction)
                img_results['DT'] = dt_metrics
            else:
                print(f"Warning: DT prediction not found for {img_name}")
            
            # Calculate metrics for RF model
            if rf_prediction is not None:
                rf_metrics = scikit_accuracy_assessment(ground_truth, rf_prediction)
                img_results['RF'] = rf_metrics
            else:
                print(f"Warning: RF prediction not found for {img_name}")
            
            results[img_name] = img_results
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    return results

def save_sklearn_results_to_txt(results, output_file="sklearn_accuracy_results.txt"):
    """
    Save scikit-learn results to text file in the required format
    
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
    
    print(f"Scikit-learn results saved to {output_file}")

def compare_custom_vs_sklearn(custom_results, sklearn_results):
    """
    Compare custom and scikit-learn results
    
    Args:
        custom_results: Results from custom functions
        sklearn_results: Results from scikit-learn functions
    
    Returns:
        dict: Comparison results
    """
    comparison = {}
    
    print("\n" + "="*60)
    print("COMPARISON: CUSTOM vs SCIKIT-LEARN")
    print("="*60)
    
    for img_name in custom_results.keys():
        if img_name in sklearn_results:
            img_comparison = {}
            
            for model_name in ['DT', 'RF']:
                if model_name in custom_results[img_name] and model_name in sklearn_results[img_name]:
                    custom_metrics = custom_results[img_name][model_name]
                    sklearn_metrics = sklearn_results[img_name][model_name]
                    
                    model_comparison = {}
                    for metric in ['OA', 'precision', 'recall', 'F1']:
                        custom_val = custom_metrics[metric]
                        sklearn_val = sklearn_metrics[metric]
                        diff = abs(custom_val - sklearn_val)
                        
                        model_comparison[metric] = {
                            'custom': custom_val,
                            'sklearn': sklearn_val,
                            'difference': diff,
                            'agreement': diff < 1e-10  # Very small tolerance for floating point
                        }
                    
                    img_comparison[model_name] = model_comparison
            
            comparison[img_name] = img_comparison
    
    return comparison

def print_comparison_summary(comparison):
    """
    Print summary of comparison between custom and scikit-learn results
    
    Args:
        comparison: Comparison results dictionary
    """
    print("\nCOMPARISON SUMMARY:")
    print("-" * 40)
    
    total_comparisons = 0
    agreements = 0
    
    for img_name, img_comparison in comparison.items():
        for model_name, model_comparison in img_comparison.items():
            for metric, metric_comparison in model_comparison.items():
                total_comparisons += 1
                if metric_comparison['agreement']:
                    agreements += 1
                else:
                    print(f"Disagreement in {img_name}_{model_name} {metric}: "
                          f"Custom={metric_comparison['custom']:.6f}, "
                          f"Sklearn={metric_comparison['sklearn']:.6f}, "
                          f"Diff={metric_comparison['difference']:.6f}")
    
    agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0
    print(f"\nOverall Agreement Rate: {agreement_rate:.2%} ({agreements}/{total_comparisons})")
    
    if agreement_rate == 1.0:
        print("✅ Perfect agreement between custom and scikit-learn functions!")
    elif agreement_rate > 0.99:
        print("✅ Very good agreement with minor floating-point differences")
    else:
        print("⚠️  Some disagreements found - check implementation")

def determine_better_model(custom_results, sklearn_results):
    """
    Determine which model performs better based on accuracy metrics
    
    Args:
        custom_results: Results from custom functions
        sklearn_results: Results from scikit-learn functions
    
    Returns:
        dict: Model comparison results
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Calculate average metrics for each model
    dt_metrics = {'OA': [], 'precision': [], 'recall': [], 'F1': []}
    rf_metrics = {'OA': [], 'precision': [], 'recall': [], 'F1': []}
    
    for img_name, img_results in custom_results.items():
        if img_name in sklearn_results:
            for model_name in ['DT', 'RF']:
                if model_name in img_results and model_name in sklearn_results[img_name]:
                    metrics = img_results[model_name]
                    
                    if model_name == 'DT':
                        for metric in ['OA', 'precision', 'recall', 'F1']:
                            dt_metrics[metric].append(metrics[metric])
                    elif model_name == 'RF':
                        for metric in ['OA', 'precision', 'recall', 'F1']:
                            rf_metrics[metric].append(metrics[metric])
    
    # Calculate averages
    dt_avg = {metric: np.mean(values) for metric, values in dt_metrics.items()}
    rf_avg = {metric: np.mean(values) for metric, values in rf_metrics.items()}
    
    print(f"\nDecision Tree (DT) Average Performance:")
    for metric, value in dt_avg.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nRandom Forest (RF) Average Performance:")
    for metric, value in rf_avg.items():
        print(f"  {metric}: {value:.4f}")
    
    # Determine better model
    better_model = {}
    for metric in ['OA', 'precision', 'recall', 'F1']:
        if dt_avg[metric] > rf_avg[metric]:
            better_model[metric] = 'DT'
        elif rf_avg[metric] > dt_avg[metric]:
            better_model[metric] = 'RF'
        else:
            better_model[metric] = 'Tie'
    
    print(f"\nBetter Model by Metric:")
    for metric, model in better_model.items():
        print(f"  {metric}: {model}")
    
    # Overall assessment
    dt_wins = sum(1 for model in better_model.values() if model == 'DT')
    rf_wins = sum(1 for model in better_model.values() if model == 'RF')
    
    if dt_wins > rf_wins:
        overall_better = "Decision Tree (DT)"
    elif rf_wins > dt_wins:
        overall_better = "Random Forest (RF)"
    else:
        overall_better = "Tie"
    
    print(f"\nOverall Better Model: {overall_better}")
    
    return {
        'dt_avg': dt_avg,
        'rf_avg': rf_avg,
        'better_by_metric': better_model,
        'overall_better': overall_better
    }

def main():
    """
    Main function for Step 3: Scikit-learn Comparison
    """
    # Data directory
    data_dir = "/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong"
    inference_dir = "inference_results"
    
    print("Step 3: Scikit-learn Accuracy Assessment Comparison")
    print("=" * 50)
    
    # Calculate metrics using scikit-learn
    sklearn_results = calculate_sklearn_metrics_for_all_images(data_dir, inference_dir)
    
    # Save results to text file
    save_sklearn_results_to_txt(sklearn_results)
    
    # Load custom results for comparison
    try:
        # Import custom results (assuming they were saved in step 2)
        from step2_custom_accuracy import calculate_metrics_for_all_images as calculate_custom_metrics
        custom_results = calculate_custom_metrics(data_dir, inference_dir)
        
        # Compare results
        comparison = compare_custom_vs_sklearn(custom_results, sklearn_results)
        print_comparison_summary(comparison)
        
        # Determine better model
        model_comparison = determine_better_model(custom_results, sklearn_results)
        
    except ImportError:
        print("Warning: Could not import custom results for comparison")
        print("Run step2_custom_accuracy.py first to generate custom results")
    
    print(f"\nStep 3 completed!")
    print(f"Scikit-learn accuracy assessment results saved to sklearn_accuracy_results.txt")
    
    return sklearn_results

if __name__ == "__main__":
    results = main()
