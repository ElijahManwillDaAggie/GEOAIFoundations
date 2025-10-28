#!/usr/bin/env python3
"""
Lab 5: Complete Accuracy Assessment Pipeline
This script runs the complete accuracy assessment pipeline for Lab 5,
including preprocessing, inference, custom accuracy assessment, and
scikit-learn comparison.
"""

import os
import sys
import numpy as np
import rasterio
from pathlib import Path
import joblib
from datetime import datetime

# Import our custom modules
from step1_preprocessing_inference import (
    preprocess_label, load_s2_image, create_chips, 
    run_inference_on_chips, reconstruct_prediction_map,
    process_single_image, create_dummy_models
)
from step2_custom_accuracy import (
    custom_accuracy_assessment, load_prediction_and_ground_truth,
    calculate_metrics_for_all_images as calculate_custom_metrics,
    save_results_to_txt as save_custom_results
)
from step3_sklearn_comparison import (
    scikit_accuracy_assessment, calculate_sklearn_metrics_for_all_images,
    save_sklearn_results_to_txt, compare_custom_vs_sklearn,
    print_comparison_summary, determine_better_model
)

class Lab5Pipeline:
    def __init__(self, data_dir, model_dt_path=None, model_rf_path=None):
        """
        Initialize the Lab 5 pipeline
        
        Args:
            data_dir: Path to the S1F11_Mekong directory
            model_dt_path: Path to Decision Tree model (optional)
            model_rf_path: Path to Random Forest model (optional)
        """
        self.data_dir = Path(data_dir)
        self.model_dt_path = model_dt_path
        self.model_rf_path = model_rf_path
        
        # Results storage
        self.inference_results = {}
        self.custom_results = {}
        self.sklearn_results = {}
        
    def step1_preprocessing_and_inference(self):
        """
        Step 1: Preprocess dataset and run inference
        """
        print("="*60)
        print("STEP 1: PREPROCESSING AND INFERENCE")
        print("="*60)
        
        # Load or create models
        if self.model_dt_path and self.model_rf_path:
            print("Loading trained models...")
            model_dt = joblib.load(self.model_dt_path)
            model_rf = joblib.load(self.model_rf_path)
        else:
            print("Creating dummy models for demonstration...")
            print("Note: Replace with your actual trained model paths")
            model_dt, model_rf = create_dummy_models()
        
        # Get all image names
        label_dir = self.data_dir / "Label"
        label_files = list(label_dir.glob("*.tif"))
        image_names = [f.stem for f in label_files]
        
        print(f"Found {len(image_names)} images to process")
        
        # Process each image
        for img_name in image_names:
            print(f"Processing {img_name}...")
            try:
                results = process_single_image(img_name, model_dt, model_rf, str(self.data_dir))
                if results:
                    self.inference_results[img_name] = results
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
        
        # Save inference results
        self.save_inference_results()
        
        print(f"\nStep 1 completed! Processed {len(self.inference_results)} images")
        return self.inference_results
    
    def step2_custom_accuracy_assessment(self):
        """
        Step 2: Custom accuracy assessment functions
        """
        print("\n" + "="*60)
        print("STEP 2: CUSTOM ACCURACY ASSESSMENT")
        print("="*60)
        
        # Calculate custom metrics
        self.custom_results = calculate_custom_metrics(str(self.data_dir), "inference_results")
        
        # Save results
        save_custom_results(self.custom_results, "custom_accuracy_results.txt")
        
        print(f"\nStep 2 completed! Custom accuracy assessment saved")
        return self.custom_results
    
    def step3_sklearn_comparison(self):
        """
        Step 3: Scikit-learn comparison
        """
        print("\n" + "="*60)
        print("STEP 3: SCIKIT-LEARN COMPARISON")
        print("="*60)
        
        # Calculate scikit-learn metrics
        self.sklearn_results = calculate_sklearn_metrics_for_all_images(str(self.data_dir), "inference_results")
        
        # Save results
        save_sklearn_results_to_txt(self.sklearn_results, "sklearn_accuracy_results.txt")
        
        # Compare with custom results
        if self.custom_results:
            comparison = compare_custom_vs_sklearn(self.custom_results, self.sklearn_results)
            print_comparison_summary(comparison)
            
            # Determine better model
            model_comparison = determine_better_model(self.custom_results, self.sklearn_results)
        else:
            print("Warning: No custom results available for comparison")
        
        print(f"\nStep 3 completed! Scikit-learn comparison saved")
        return self.sklearn_results
    
    def save_inference_results(self):
        """
        Save inference results as GeoTIFF files and create zip
        """
        import zipfile
        
        # Create inference results directory
        os.makedirs("inference_results", exist_ok=True)
        
        # Save individual prediction files
        for img_name, img_results in self.inference_results.items():
            for model_name, model_results in img_results.items():
                pred_map = model_results['prediction_map']
                
                # Get original image info for georeferencing
                label_path = self.data_dir / "Label" / f"{img_name}.tif"
                
                with rasterio.open(label_path) as src:
                    profile = src.profile
                    profile.update(dtype=pred_map.dtype, count=1)
                
                # Save prediction
                output_path = Path("inference_results") / f"{img_name}_{model_name}_prediction.tif"
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(pred_map, 1)
        
        # Create zip file
        with zipfile.ZipFile("inference_results.zip", 'w') as zipf:
            for img_name, img_results in self.inference_results.items():
                for model_name, model_results in img_results.items():
                    pred_map = model_results['prediction_map']
                    
                    # Get original image info for georeferencing
                    label_path = self.data_dir / "Label" / f"{img_name}.tif"
                    
                    with rasterio.open(label_path) as src:
                        profile = src.profile
                        profile.update(dtype=pred_map.dtype, count=1)
                    
                    # Save prediction to temporary file
                    temp_path = f"temp_{img_name}_{model_name}.tif"
                    
                    with rasterio.open(temp_path, 'w', **profile) as dst:
                        dst.write(pred_map, 1)
                    
                    # Add to zip
                    zipf.write(temp_path, f"{img_name}_{model_name}_prediction.tif")
                    
                    # Clean up temporary file
                    os.remove(temp_path)
        
        print("Inference results saved to inference_results.zip")
    
    def create_summary_report(self):
        """
        Create a summary report of the accuracy assessment
        """
        report_path = "lab5_accuracy_assessment_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("LAB 5: ACCURACY ASSESSMENT REPORT\n")
            f.write("="*50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Number of images processed: {len(self.inference_results)}\n\n")
            
            # Custom results summary
            if self.custom_results:
                f.write("CUSTOM ACCURACY ASSESSMENT RESULTS:\n")
                f.write("-" * 40 + "\n")
                
                for model_name in ['DT', 'RF']:
                    oa_values = []
                    precision_values = []
                    recall_values = []
                    f1_values = []
                    
                    for img_name, img_results in self.custom_results.items():
                        if model_name in img_results:
                            metrics = img_results[model_name]
                            oa_values.append(metrics['OA'])
                            precision_values.append(metrics['precision'])
                            recall_values.append(metrics['recall'])
                            f1_values.append(metrics['F1'])
                    
                    if oa_values:
                        f.write(f"\n{model_name} Model (Custom):\n")
                        f.write(f"  Overall Accuracy: {np.mean(oa_values):.4f} ± {np.std(oa_values):.4f}\n")
                        f.write(f"  Precision:        {np.mean(precision_values):.4f} ± {np.std(precision_values):.4f}\n")
                        f.write(f"  Recall:           {np.mean(recall_values):.4f} ± {np.std(recall_values):.4f}\n")
                        f.write(f"  F1 Score:         {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}\n")
            
            # Scikit-learn results summary
            if self.sklearn_results:
                f.write("\n\nSCIKIT-LEARN ACCURACY ASSESSMENT RESULTS:\n")
                f.write("-" * 40 + "\n")
                
                for model_name in ['DT', 'RF']:
                    oa_values = []
                    precision_values = []
                    recall_values = []
                    f1_values = []
                    
                    for img_name, img_results in self.sklearn_results.items():
                        if model_name in img_results:
                            metrics = img_results[model_name]
                            oa_values.append(metrics['OA'])
                            precision_values.append(metrics['precision'])
                            recall_values.append(metrics['recall'])
                            f1_values.append(metrics['F1'])
                    
                    if oa_values:
                        f.write(f"\n{model_name} Model (Scikit-learn):\n")
                        f.write(f"  Overall Accuracy: {np.mean(oa_values):.4f} ± {np.std(oa_values):.4f}\n")
                        f.write(f"  Precision:        {np.mean(precision_values):.4f} ± {np.std(precision_values):.4f}\n")
                        f.write(f"  Recall:           {np.mean(recall_values):.4f} ± {np.std(recall_values):.4f}\n")
                        f.write(f"  F1 Score:         {np.mean(f1_values):.4f} ± {np.std(f1_values):.4f}\n")
            
            f.write("\n\nFILES GENERATED:\n")
            f.write("-" * 20 + "\n")
            f.write("- inference_results.zip (Inference results for Canvas submission)\n")
            f.write("- custom_accuracy_results.txt (Custom accuracy assessment)\n")
            f.write("- sklearn_accuracy_results.txt (Scikit-learn accuracy assessment)\n")
            f.write("- lab5_accuracy_assessment_report.txt (This report)\n")
        
        print(f"Summary report saved to {report_path}")
    
    def run_complete_pipeline(self):
        """
        Run the complete Lab 5 pipeline
        """
        print("LAB 5: ACCURACY ASSESSMENT PIPELINE")
        print("="*60)
        print("This pipeline will:")
        print("1. Preprocess Sen1Flood11 dataset and run inference")
        print("2. Calculate custom accuracy assessment metrics")
        print("3. Compare with scikit-learn accuracy assessment")
        print("4. Generate summary report")
        print("="*60)
        
        try:
            # Step 1: Preprocessing and Inference
            self.step1_preprocessing_and_inference()
            
            # Step 2: Custom Accuracy Assessment
            self.step2_custom_accuracy_assessment()
            
            # Step 3: Scikit-learn Comparison
            self.step3_sklearn_comparison()
            
            # Create summary report
            self.create_summary_report()
            
            print("\n" + "="*60)
            print("LAB 5 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Generated files:")
            print("- inference_results.zip (Submit to Canvas)")
            print("- custom_accuracy_results.txt (Submit to Canvas)")
            print("- sklearn_accuracy_results.txt (Submit to Canvas)")
            print("- lab5_accuracy_assessment_report.txt (Reference)")
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise

def main():
    """
    Main function to run the complete Lab 5 pipeline
    """
    # Data directory
    data_dir = "/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong"
    
    # Model paths (replace with your actual model paths)
    model_dt_path = None  # "path/to/your/dt_model.pkl"
    model_rf_path = None  # "path/to/your/rf_model.pkl"
    
    # Initialize pipeline
    pipeline = Lab5Pipeline(data_dir, model_dt_path, model_rf_path)
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()
