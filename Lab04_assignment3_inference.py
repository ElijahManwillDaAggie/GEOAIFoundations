import os
import numpy as np
from osgeo import gdal, osr
import joblib  # For loading the model

def read_geotiff(tiff_file_path):
    """
    Function to read and process a georeferenced TIFF file using GDAL
    """
    # Open the dataset
    dataset = gdal.Open(tiff_file_path)
    
    if dataset is None:
        raise ValueError(f"Could not open file: {tiff_file_path}")
    
    # Read the image data
    image = dataset.ReadAsArray()
    
    # Get geotransform and projection
    transform = dataset.GetGeoTransform()
    crs = dataset.GetProjection()
    
    # Close the dataset
    dataset = None
    
    return image, transform, crs

def preprocess_data(image):
    """
    Function to reshape and transpose the image for machine learning
    """
    # Reshape the image array to (pixels, bands)
    n_bands, n_rows, n_cols = image.shape
    image_array = image.reshape(n_bands, -1)  # reshape array
    image_array = image_array.T  # Transpose the array
    
    return image_array

def save_geotiff(output_file_path, predictions, transform, crs, width, height):
    """
    Function to reshape and save the predictions back to a georeferenced TIFF file using GDAL
    """
    # Reshape predictions to (height, width)
    predictions = predictions.reshape(height, width)
    
    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_file_path, width, height, 1, gdal.GDT_Float32)
    
    # Set geotransform and projection
    out_dataset.SetGeoTransform(transform)
    out_dataset.SetProjection(crs)
    
    # Write the data
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(predictions)
    out_band.FlushCache()
    
    # Close the dataset
    out_dataset = None

def run_inference(input_tiff, model_file, output_tiff):
    """
    Complete inference pipeline
    """
    print(f"Running inference on: {input_tiff}")
    print(f"Using model: {model_file}")
    print(f"Output will be saved to: {output_tiff}")
    
    # Step 1: Read the input TIFF file using GDAL
    print("Step 1: Reading input image...")
    image, transform, crs = read_geotiff(input_tiff)
    print(f"Image shape: {image.shape}")
    
    # Step 2: Preprocess the image for inference
    print("Step 2: Preprocessing image for ML...")
    data = preprocess_data(image)
    print(f"Preprocessed data shape: {data.shape}")
    
    # Step 3: Load the trained model
    print("Step 3: Loading trained model...")
    model = joblib.load(model_file)
    print("Model loaded successfully")
    
    # Step 4: Run inference using the model
    print("Step 4: Running inference...")
    predictions = model.predict(data)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique prediction values: {np.unique(predictions)}")
    
    # Step 5: Save the prediction results as a georeferenced TIFF using GDAL
    print("Step 5: Saving results...")
    height, width = image.shape[1], image.shape[2]  # Original height and width
    save_geotiff(output_tiff, predictions, transform, crs, width, height)
    print(f"Results saved to: {output_tiff}")
    
    return predictions

if __name__ == "__main__":
    # Test images for inference
    test_images = [
        "assignment3_Infer_Data/image/USA_741073.tif",
        "assignment3_Infer_Data/image/USA_741178.tif"
    ]
    
    # Model files (these will be created after training)
    rf_model_file = "random_forest_model.pkl"
    dt_model_file = "decision_tree_model.pkl"
    
    print("=== Assignment 3: Model Inference ===")
    print("=" * 50)
    
    # Check if models exist
    if not os.path.exists(rf_model_file):
        print(f"Error: Model file {rf_model_file} not found. Please run training first.")
        exit(1)
    
    if not os.path.exists(dt_model_file):
        print(f"Error: Model file {dt_model_file} not found. Please run training first.")
        exit(1)
    
    # Run inference for each test image with both models
    for test_image in test_images:
        if not os.path.exists(test_image):
            print(f"Warning: Test image {test_image} not found. Skipping...")
            continue
            
        base_name = os.path.splitext(os.path.basename(test_image))[0]
        
        print(f"\n{'='*60}")
        print(f"Processing: {test_image}")
        print(f"{'='*60}")
        
        # Random Forest inference
        print("\n--- Random Forest Model ---")
        rf_output = f"rf_prediction_{base_name}.tif"
        try:
            rf_predictions = run_inference(test_image, rf_model_file, rf_output)
        except Exception as e:
            print(f"Error with Random Forest: {e}")
            continue
        
        # Decision Tree inference
        print("\n--- Decision Tree Model ---")
        dt_output = f"dt_prediction_{base_name}.tif"
        try:
            dt_predictions = run_inference(test_image, dt_model_file, dt_output)
        except Exception as e:
            print(f"Error with Decision Tree: {e}")
            continue
        
        print(f"\nInference completed for {base_name}")
        print(f"Random Forest output: {rf_output}")
        print(f"Decision Tree output: {dt_output}")
    
    print("\n" + "="*60)
    print("All inference completed!")
    print("="*60)
