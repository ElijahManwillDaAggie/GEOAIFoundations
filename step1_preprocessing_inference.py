#!/usr/bin/env python3
"""
Lab 5 - Step 1: Preprocessing and Inference
This script handles preprocessing of Sen1Flood11 dataset and runs inference
with DT and RF models on the Mekong region data.
"""

import os
import numpy as np
import rasterio
import joblib
from pathlib import Path
import zipfile
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def preprocess_label(label_path, output_path=None):
    """
    Preprocess label by replacing -1 values with 0
    
    Args:
        label_path: Path to label .tif file
        output_path: Optional output path for preprocessed label
    
    Returns:
        numpy array: Preprocessed label data
    """
    with rasterio.open(label_path) as src:
        label_data = src.read(1)
        
        # Replace -1 (invalid/cloudy) values with 0
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

def load_s2_image(s2_path):
    """
    Load Sentinel-2 image data
    
    Args:
        s2_path: Path to S2 .tif file
    
    Returns:
        numpy array: S2 image data
    """
    with rasterio.open(s2_path) as src:
        s2_data = src.read()
        return s2_data

def create_chips(s2_data, chip_size=256, stride=128):
    """
    Create chips from S2 data for model input
    
    Args:
        s2_data: S2 image data (bands, height, width)
        chip_size: Size of each chip
        stride: Stride for creating chips
    
    Returns:
        tuple: (chips, chip_coords)
    """
    bands, height, width = s2_data.shape
    chips = []
    chip_coords = []
    
    for y in range(0, height - chip_size + 1, stride):
        for x in range(0, width - chip_size + 1, stride):
            chip = s2_data[:, y:y+chip_size, x:x+chip_size]
            chips.append(chip)
            chip_coords.append((y, x))
    
    return chips, chip_coords

def run_inference_on_chips(model, chips):
    """
    Run inference on chips using the provided model
    
    Args:
        model: Trained model
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

def reconstruct_prediction_map(predictions, chip_coords, original_shape, chip_size=256):
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

def process_single_image(img_name, model_dt, model_rf, data_dir):
    """
    Process a single image through both models
    
    Args:
        img_name: Name of the image (without extension)
        model_dt: Decision Tree model
        model_rf: Random Forest model
        data_dir: Path to data directory
    
    Returns:
        dict: Results for both models
    """
    print(f"Processing {img_name}...")
    
    data_path = Path(data_dir)
    label_dir = data_path / "Label"
    s2_dir = data_path / "S2"
    
    # File paths
    label_path = label_dir / f"{img_name}.tif"
    s2_path = s2_dir / f"{img_name}.tif"
    
    # Check if files exist
    if not label_path.exists():
        print(f"Warning: Label file {label_path} not found")
        return None
    
    if not s2_path.exists():
        print(f"Warning: S2 file {s2_path} not found")
        return None
    
    # Load and preprocess data
    ground_truth = preprocess_label(label_path)
    s2_data = load_s2_image(s2_path)
    
    # Create chips
    chips, chip_coords = create_chips(s2_data)
    
    results = {}
    
    # Process with both models
    for model_name, model in [('DT', model_dt), ('RF', model_rf)]:
        # Run inference
        predictions = run_inference_on_chips(model, chips)
        
        # Reconstruct prediction map
        pred_map = reconstruct_prediction_map(
            predictions, chip_coords, ground_truth.shape
        )
        
        results[model_name] = {
            'prediction_map': pred_map,
            'ground_truth': ground_truth
        }
    
    return results

def create_dummy_models():
    """
    Create dummy models for demonstration
    In practice, you should load your actual trained models
    """
    # Create dummy training data
    X_dummy = np.random.rand(1000, 10)  # 10 features
    y_dummy = np.random.randint(0, 2, 1000)  # Binary classification
    
    # Create and fit models
    model_dt = DecisionTreeClassifier(random_state=42)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model_dt.fit(X_dummy, y_dummy)
    model_rf.fit(X_dummy, y_dummy)
    
    return model_dt, model_rf

def save_inference_results(results, output_dir="inference_results"):
    """
    Save inference results as GeoTIFF files
    
    Args:
        results: Dictionary containing results for all images
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name, img_results in results.items():
        for model_name, model_results in img_results.items():
            pred_map = model_results['prediction_map']
            
            # Get original image info for georeferencing
            label_path = Path("/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong/Label") / f"{img_name}.tif"
            
            with rasterio.open(label_path) as src:
                profile = src.profile
                profile.update(dtype=pred_map.dtype, count=1)
            
            # Save prediction
            output_path = Path(output_dir) / f"{img_name}_{model_name}_prediction.tif"
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(pred_map, 1)
            
            print(f"Saved: {output_path}")

def create_inference_zip(results, zip_path="inference_results.zip"):
    """
    Create zip file with inference results
    
    Args:
        results: Dictionary containing results for all images
        zip_path: Path to output zip file
    """
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for img_name, img_results in results.items():
            for model_name, model_results in img_results.items():
                pred_map = model_results['prediction_map']
                
                # Get original image info for georeferencing
                label_path = Path("/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong/Label") / f"{img_name}.tif"
                
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
    
    print(f"Inference results saved to {zip_path}")

def main():
    """
    Main function for Step 1: Preprocessing and Inference
    """
    # Data directory
    data_dir = "/Users/elijahmanwill/Documents/geoairemotesensing/Lab_05/S1F11_Mekong"
    
    print("Step 1: Preprocessing and Inference")
    print("=" * 50)
    
    # Load models (replace with your actual model paths)
    print("Loading models...")
    print("Note: Replace with your actual trained model paths")
    print("Example: model_dt = joblib.load('path/to/dt_model.pkl')")
    
    # For demonstration, create dummy models
    model_dt, model_rf = create_dummy_models()
    
    # Get all image names
    label_dir = Path(data_dir) / "Label"
    label_files = list(label_dir.glob("*.tif"))
    image_names = [f.stem for f in label_files]
    
    print(f"Found {len(image_names)} images to process")
    
    # Process each image
    all_results = {}
    for img_name in image_names:
        try:
            results = process_single_image(img_name, model_dt, model_rf, data_dir)
            if results:
                all_results[img_name] = results
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Save results
    print("\nSaving inference results...")
    save_inference_results(all_results)
    create_inference_zip(all_results)
    
    print(f"\nStep 1 completed!")
    print(f"Processed {len(all_results)} images")
    print("Inference results saved to inference_results.zip")
    
    return all_results

if __name__ == "__main__":
    results = main()
