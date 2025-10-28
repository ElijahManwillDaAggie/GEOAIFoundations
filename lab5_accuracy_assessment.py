#!/usr/bin/env python3
"""
Lab 5: Accuracy Assessment for Flood Detection Models
This script performs accuracy assessment on DT and RF models trained in previous lab
for flood detection in a new region (Mekong).
"""

import os
import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
import zipfile
from datetime import datetime

class AccuracyAssessment:
    def __init__(self, data_dir, model_dir=None):
        """
        Initialize the accuracy assessment class
        
        Args:
            data_dir: Path to the S1F11_Mekong directory
            model_dir: Path to directory containing trained models (DT and RF)
        """
        self.data_dir = Path(data_dir)
        self.label_dir = self.data_dir / "Label"
        self.s2_dir = self.data_dir / "S2"
        self.model_dir = Path(model_dir) if model_dir else None
        
        # Results storage
        self.results = {}
        
    def preprocess_labels(self, label_path, output_path=None):
        """
        Preprocess label data by replacing -1 values with 0
        
        Args:
            label_path: Path to the label .tif file
            output_path: Path to save preprocessed label (optional)
        
        Returns:
            numpy array: Preprocessed label data
        """
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
            
            # Replace -1 values with 0
            label_data[label_data == -1] = 0
            
            if output_path:
                # Save preprocessed label
                with rasterio.open(output_path, 'w', 
                                 driver='GTiff',
                                 height=src.height,
                                 width=src.width,
                                 count=1,
                                 dtype=label_data.dtype,
                                 crs=src.crs,
                                 transform=src.transform) as dst:
                    dst.write(label_data, 1)
            
            return label_data
    
    def load_s2_image(self, s2_path):
        """
        Load Sentinel-2 image data
        
        Args:
            s2_path: Path to the S2 .tif file
        
        Returns:
            numpy array: S2 image data
        """
        with rasterio.open(s2_path) as src:
            # Read all bands
            s2_data = src.read()
            return s2_data
    
    def prepare_model_input(self, s2_data, chip_size=256):
        """
        Prepare input data for model inference by creating chips
        
        Args:
            s2_data: S2 image data (bands, height, width)
            chip_size: Size of each chip
        
        Returns:
            list: List of chips ready for model input
        """
        bands, height, width = s2_data.shape
        chips = []
        chip_coords = []
        
        # Create overlapping chips
        stride = chip_size // 2  # 50% overlap
        
        for y in range(0, height - chip_size + 1, stride):
            for x in range(0, width - chip_size + 1, stride):
                chip = s2_data[:, y:y+chip_size, x:x+chip_size]
                chips.append(chip)
                chip_coords.append((y, x))
        
        return chips, chip_coords
    
    def run_inference(self, model, chips):
        """
        Run inference on chips using the provided model
        
        Args:
            model: Trained model (DT or RF)
            chips: List of chips
        
        Returns:
            list: Predictions for each chip
        """
        predictions = []
        
        for chip in chips:
            # Reshape chip for model input
            chip_flat = chip.reshape(chip.shape[0], -1).T  # (pixels, bands)
            
            # Predict
            pred = model.predict(chip_flat)
            predictions.append(pred)
        
        return predictions
    
    def reconstruct_prediction_map(self, predictions, chip_coords, original_shape, chip_size=256):
        """
        Reconstruct full prediction map from chip predictions
        
        Args:
            predictions: List of chip predictions
            chip_coords: List of chip coordinates
            original_shape: Original image shape (height, width)
            chip_size: Size of each chip
        
        Returns:
            numpy array: Full prediction map
        """
        height, width = original_shape
        pred_map = np.zeros((height, width), dtype=np.int32)
        count_map = np.zeros((height, width), dtype=np.int32)
        
        for pred, (y, x) in zip(predictions, chip_coords):
            # Handle overlapping regions by averaging
            pred_map[y:y+chip_size, x:x+chip_size] += pred.reshape(chip_size, chip_size)
            count_map[y:y+chip_size, x:x+chip_size] += 1
        
        # Average overlapping regions
        pred_map = np.round(pred_map / np.maximum(count_map, 1)).astype(np.int32)
        
        return pred_map
    
    def custom_accuracy_assessment(self, ground_truth, prediction):
        """
        Custom accuracy assessment functions
        
        Args:
            ground_truth: Ground truth array
            prediction: Prediction array
        
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
            'F1': f1
        }
    
    def scikit_accuracy_assessment(self, ground_truth, prediction):
        """
        Accuracy assessment using scikit-learn
        
        Args:
            ground_truth: Ground truth array
            prediction: Prediction array
        
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
    
    def process_single_image(self, img_name, model_dt, model_rf):
        """
        Process a single image through both models
        
        Args:
            img_name: Name of the image (without extension)
            model_dt: Decision Tree model
            model_rf: Random Forest model
        
        Returns:
            dict: Results for both models
        """
        print(f"Processing {img_name}...")
        
        # File paths
        label_path = self.label_dir / f"{img_name}.tif"
        s2_path = self.s2_dir / f"{img_name}.tif"
        
        # Check if files exist
        if not label_path.exists():
            print(f"Warning: Label file {label_path} not found")
            return None
        
        if not s2_path.exists():
            print(f"Warning: S2 file {s2_path} not found")
            return None
        
        # Load and preprocess data
        ground_truth = self.preprocess_labels(label_path)
        s2_data = self.load_s2_image(s2_path)
        
        # Prepare chips for model input
        chips, chip_coords = self.prepare_model_input(s2_data)
        
        results = {}
        
        # Process with both models
        for model_name, model in [('DT', model_dt), ('RF', model_rf)]:
            # Run inference
            predictions = self.run_inference(model, chips)
            
            # Reconstruct prediction map
            pred_map = self.reconstruct_prediction_map(
                predictions, chip_coords, ground_truth.shape
            )
            
            # Calculate accuracy metrics
            custom_metrics = self.custom_accuracy_assessment(ground_truth, pred_map)
            sklearn_metrics = self.scikit_accuracy_assessment(ground_truth, pred_map)
            
            results[model_name] = {
                'custom': custom_metrics,
                'sklearn': sklearn_metrics,
                'prediction_map': pred_map
            }
        
        return results
    
    def process_all_images(self, model_dt_path, model_rf_path):
        """
        Process all images in the dataset
        
        Args:
            model_dt_path: Path to Decision Tree model
            model_rf_path: Path to Random Forest model
        """
        # Load models
        model_dt = joblib.load(model_dt_path)
        model_rf = joblib.load(model_rf_path)
        
        # Get all image names
        label_files = list(self.label_dir.glob("*.tif"))
        image_names = [f.stem for f in label_files]
        
        print(f"Found {len(image_names)} images to process")
        
        # Process each image
        for img_name in image_names:
            try:
                results = self.process_single_image(img_name, model_dt, model_rf)
                if results:
                    self.results[img_name] = results
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
    
    def save_results_to_txt(self, output_file, method='custom'):
        """
        Save results to text file
        
        Args:
            output_file: Path to output file
            method: 'custom' or 'sklearn'
        """
        with open(output_file, 'w') as f:
            # Write header
            f.write("Image_Name OA Precision Recall F1\n")
            
            # Calculate averages
            oa_values = []
            precision_values = []
            recall_values = []
            f1_values = []
            
            # Write results for each image
            for img_name, img_results in self.results.items():
                for model_name, model_results in img_results.items():
                    metrics = model_results[method]
                    
                    f.write(f"{img_name}_{model_name} {metrics['OA']:.4f} "
                           f"{metrics['precision']:.4f} {metrics['recall']:.4f} "
                           f"{metrics['F1']:.4f}\n")
                    
                    oa_values.append(metrics['OA'])
                    precision_values.append(metrics['precision'])
                    recall_values.append(metrics['recall'])
                    f1_values.append(metrics['F1'])
            
            # Write averages
            avg_oa = np.mean(oa_values)
            avg_precision = np.mean(precision_values)
            avg_recall = np.mean(recall_values)
            avg_f1 = np.mean(f1_values)
            
            f.write(f"Average_stat {avg_oa:.4f} {avg_precision:.4f} "
                   f"{avg_recall:.4f} {avg_f1:.4f}\n")
    
    def create_inference_zip(self, output_zip_path):
        """
        Create zip file with inference results
        
        Args:
            output_zip_path: Path to output zip file
        """
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for img_name, img_results in self.results.items():
                for model_name, model_results in img_results.items():
                    # Save prediction map as GeoTIFF
                    pred_map = model_results['prediction_map']
                    
                    # Get original image info for georeferencing
                    label_path = self.label_dir / f"{img_name}.tif"
                    with rasterio.open(label_path) as src:
                        profile = src.profile
                        profile.update(dtype=pred_map.dtype, count=1)
                    
                    # Save prediction
                    pred_filename = f"{img_name}_{model_name}_prediction.tif"
                    pred_path = f"inference_results/{pred_filename}"
                    
                    with rasterio.open(pred_path, 'w', **profile) as dst:
                        dst.write(pred_map, 1)
                    
                    zipf.write(pred_path, pred_filename)
                    os.remove(pred_path)  # Clean up temporary file
        
        print(f"Inference results saved to {output_zip_path}")


def main():
    """
    Main function to run the accuracy assessment
    """
    # Initialize paths
    data_dir = "/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong"
    
    # You'll need to provide paths to your trained models
    # model_dt_path = "path/to/your/dt_model.pkl"
    # model_rf_path = "path/to/your/rf_model.pkl"
    
    # For demonstration, we'll create dummy models
    # In practice, you should load your actual trained models
    print("Note: You need to provide paths to your trained DT and RF models")
    print("Replace the dummy model creation with your actual model loading")
    
    # Create dummy models for demonstration
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Create dummy models (replace with your actual models)
    model_dt = DecisionTreeClassifier(random_state=42)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Create dummy training data to fit models (replace with your actual training)
    X_dummy = np.random.rand(1000, 10)  # 10 features
    y_dummy = np.random.randint(0, 2, 1000)  # Binary classification
    
    model_dt.fit(X_dummy, y_dummy)
    model_rf.fit(X_dummy, y_dummy)
    
    # Initialize accuracy assessment
    assessor = AccuracyAssessment(data_dir)
    
    # Process all images
    assessor.process_all_images(model_dt, model_rf)
    
    # Save results
    assessor.save_results_to_txt("custom_accuracy_results.txt", method='custom')
    assessor.save_results_to_txt("sklearn_accuracy_results.txt", method='sklearn')
    
    # Create inference results zip
    assessor.create_inference_zip("inference_results.zip")
    
    print("Accuracy assessment completed!")
    print("Results saved to:")
    print("- custom_accuracy_results.txt")
    print("- sklearn_accuracy_results.txt")
    print("- inference_results.zip")


if __name__ == "__main__":
    main()
