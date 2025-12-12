"""
Apply Trained Models to Full Watershed Images
Weber River Watershed - Impervious Surface Classification

This script:
1. Loads trained Random Forest and Decision Tree models
2. Applies models to all yearly Sentinel-2 images
3. Creates impervious surface classification maps
4. Saves results as GeoTIFF files
"""

import os
import numpy as np
from osgeo import gdal, osr
import joblib
from pathlib import Path

print("=" * 60)
print("Impervious Surface Classification - Model Inference")
print("=" * 60)

# Configuration
TIFF_FOLDER = 'tiff_images'
MODEL_FOLDER = 'models'
OUTPUT_FOLDER = 'predictions'

RF_MODEL_PATH = os.path.join(MODEL_FOLDER, 'random_forest_model.pkl')
DT_MODEL_PATH = os.path.join(MODEL_FOLDER, 'decision_tree_model.pkl')

# Feature names (must match training)
FEATURE_NAMES = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 
                 'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'IBI']

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def read_geotiff(tiff_file_path):
    """
    Read and process a georeferenced TIFF file using GDAL
    
    Returns:
        image: numpy array (bands, height, width)
        transform: geotransform
        crs: coordinate reference system
    """
    dataset = gdal.Open(tiff_file_path)
    
    if dataset is None:
        raise ValueError(f"Could not open file: {tiff_file_path}")
    
    # Read all bands
    n_bands = dataset.RasterCount
    height = dataset.RasterYSize
    width = dataset.RasterXSize
    
    image = np.zeros((n_bands, height, width), dtype=np.float32)
    for i in range(n_bands):
        band = dataset.GetRasterBand(i + 1)
        image[i] = band.ReadAsArray()
    
    # Get geotransform and projection
    transform = dataset.GetGeoTransform()
    crs = dataset.GetProjection()
    
    # Close dataset
    dataset = None
    
    return image, transform, crs

def preprocess_data(image):
    """
    Preprocess image data for machine learning
    Reshape from (bands, height, width) to (pixels, bands)
    
    Args:
        image: numpy array (bands, height, width)
    
    Returns:
        data: numpy array (pixels, bands)
        original_shape: tuple (height, width)
    """
    n_bands, n_rows, n_cols = image.shape
    original_shape = (n_rows, n_cols)
    
    # Reshape to (bands, pixels) then transpose to (pixels, bands)
    image_array = image.reshape(n_bands, -1).T
    
    return image_array, original_shape

def save_geotiff(output_file_path, predictions, transform, crs, width, height):
    """
    Save predictions as a georeferenced TIFF file using GDAL
    
    Args:
        output_file_path: path to output file
        predictions: numpy array (height, width) with class predictions
        transform: geotransform
        crs: coordinate reference system
        width: image width
        height: image height
    """
    # Reshape predictions to (height, width) if needed
    if predictions.ndim == 1:
        predictions = predictions.reshape(height, width)
    
    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        output_file_path, 
        width, 
        height, 
        1, 
        gdal.GDT_Byte  # Use Byte for binary classification (0 or 1)
    )
    
    # Set geotransform and projection
    out_dataset.SetGeoTransform(transform)
    out_dataset.SetProjection(crs)
    
    # Write the data
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(predictions.astype(np.uint8))
    
    # Set NoData value (optional)
    out_band.SetNoDataValue(255)
    
    # Set description
    out_band.SetDescription('Impervious Surface Classification (0=Non-Impervious, 1=Impervious)')
    
    # Flush cache
    out_band.FlushCache()
    
    # Close dataset
    out_dataset = None
    
    print(f"   Saved: {output_file_path}")

def run_inference(input_tiff, model_file, output_tiff, model_name):
    """
    Complete inference pipeline for a single image
    
    Args:
        input_tiff: path to input GeoTIFF
        model_file: path to trained model
        output_tiff: path to output GeoTIFF
        model_name: name of model (for display)
    """
    print(f"\n  Processing with {model_name}...")
    print(f"    Input: {os.path.basename(input_tiff)}")
    
    # Step 1: Read the input TIFF file
    print("    Step 1: Reading input image...")
    image, transform, crs = read_geotiff(input_tiff)
    print(f"      Image shape: {image.shape} (bands, height, width)")
    
    # Step 2: Preprocess the image for inference
    print("    Step 2: Preprocessing image for ML...")
    data, original_shape = preprocess_data(image)
    print(f"      Preprocessed data shape: {data.shape} (pixels, bands)")
    
    # Step 3: Load the trained model
    print("    Step 3: Loading trained model...")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = joblib.load(model_file)
    print("      Model loaded successfully")
    
    # Step 4: Run inference
    print("    Step 4: Running inference...")
    # Process in chunks if image is very large to avoid memory issues
    chunk_size = 100000  # Process 100k pixels at a time
    
    if len(data) > chunk_size:
        print(f"      Processing in chunks of {chunk_size} pixels...")
        predictions = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            chunk_pred = model.predict(chunk)
            predictions.append(chunk_pred)
        predictions = np.hstack(predictions)
    else:
        predictions = model.predict(data)
    
    print(f"      Predictions shape: {predictions.shape}")
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"      Unique prediction values: {dict(zip(unique, counts))}")
    
    # Calculate percentages
    total_pixels = len(predictions)
    for val, count in zip(unique, counts):
        label = "Impervious" if val == 1 else "Non-Impervious"
        pct = 100 * count / total_pixels
        print(f"        {label}: {count:,} pixels ({pct:.2f}%)")
    
    # Step 5: Save the prediction results
    print("    Step 5: Saving results...")
    height, width = original_shape
    save_geotiff(output_tiff, predictions, transform, crs, width, height)
    
    return predictions

def main():
    """Main function to process all images"""
    # Check if models exist
    if not os.path.exists(RF_MODEL_PATH):
        print(f"Error: Random Forest model not found: {RF_MODEL_PATH}")
        print("Please run train_models.py first to train the models.")
        return
    
    if not os.path.exists(DT_MODEL_PATH):
        print(f"Error: Decision Tree model not found: {DT_MODEL_PATH}")
        print("Please run train_models.py first to train the models.")
        return
    
    # Get list of input images
    tiff_files = sorted([f for f in os.listdir(TIFF_FOLDER) 
                        if f.endswith('.tif')])
    
    if not tiff_files:
        print(f"Error: No GeoTIFF files found in {TIFF_FOLDER}")
        return
    
    print(f"\nFound {len(tiff_files)} images to process:")
    for f in tiff_files:
        print(f"  - {f}")
    
    # Process each image with both models
    print("\n" + "=" * 60)
    print("Running Inference on All Images")
    print("=" * 60)
    
    for tiff_file in tiff_files:
        input_path = os.path.join(TIFF_FOLDER, tiff_file)
        base_name = os.path.splitext(tiff_file)[0]
        
        print(f"\n{'='*60}")
        print(f"Processing: {tiff_file}")
        print(f"{'='*60}")
        
        # Random Forest inference
        rf_output = os.path.join(OUTPUT_FOLDER, f"rf_{base_name}_prediction.tif")
        try:
            rf_predictions = run_inference(input_path, RF_MODEL_PATH, rf_output, "Random Forest")
        except Exception as e:
            print(f"  Error with Random Forest: {e}")
            continue
        
        # Decision Tree inference
        dt_output = os.path.join(OUTPUT_FOLDER, f"dt_{base_name}_prediction.tif")
        try:
            dt_predictions = run_inference(input_path, DT_MODEL_PATH, dt_output, "Decision Tree")
        except Exception as e:
            print(f"  Error with Decision Tree: {e}")
            continue
        
        print(f"\n  ✓ Completed: {tiff_file}")
        print(f"    RF output: rf_{base_name}_prediction.tif")
        print(f"    DT output: dt_{base_name}_prediction.tif")
    
    print("\n" + "=" * 60)
    print("All Inference Completed!")
    print("=" * 60)
    print(f"\nPrediction maps saved to: {OUTPUT_FOLDER}/")
    print("\nOutput files:")
    for tiff_file in tiff_files:
        base_name = os.path.splitext(tiff_file)[0]
        print(f"  - rf_{base_name}_prediction.tif (Random Forest)")
        print(f"  - dt_{base_name}_prediction.tif (Decision Tree)")

if __name__ == "__main__":
    main()

