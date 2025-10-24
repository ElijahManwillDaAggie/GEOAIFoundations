# Assignment 1: Automated SAVI and NDSI Calculation
# This script processes two Landsat-5 images and calculates SAVI and NDSI spectral indices

from osgeo import gdal
import numpy as np
import os
import glob

def read_geotiff_band(file_path, band_number=1):
    """Read a specific band from a GeoTIFF file and return array and metadata."""
    dataset = gdal.Open(file_path)
    if dataset is None:
        raise FileNotFoundError(f"File {file_path} not found or unable to open.")
    
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    no_data_value = dataset.GetRasterBand(band_number).GetNoDataValue()
    
    # Read the band as numpy array
    band_array = dataset.GetRasterBand(band_number).ReadAsArray().astype(np.float32)
    
    dataset = None  # Close the dataset
    return band_array, x_size, y_size, projection, geotransform, no_data_value

def write_geotiff(output_path, x_size, y_size, projection, geotransform, no_data_value, data_to_save):
    """Write the processed data to a GeoTIFF file."""
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Float32)

    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)

    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(data_to_save)
    
    # Only set NoData value if it's valid (not None)
    if no_data_value is not None:
        output_band.SetNoDataValue(no_data_value)

    output_dataset.FlushCache()
    output_dataset = None

def calculate_savi(nir_band, red_band, l=0.5):
    """
    Calculate Soil Adjusted Vegetation Index (SAVI)
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    where L is the soil brightness correction factor (typically 0.5)
    """
    # Handle division by zero
    denominator = nir_band + red_band + l
    savi = np.where(denominator == 0., 0, ((nir_band - red_band) / denominator) * (1 + l))
    return savi

def calculate_ndsi(green_band, swir_band):
    """
    Calculate Normalized Difference Snow Index (NDSI)
    NDSI = (Green - SWIR) / (Green + SWIR)
    """
    # Handle division by zero
    denominator = green_band + swir_band
    ndsi = np.where(denominator == 0., 0, (green_band - swir_band) / denominator)
    return ndsi

def process_landsat_image(image_name, output_folder):
    """
    Process a single Landsat image to calculate SAVI and NDSI.
    Returns the image name for file naming.
    """
    # Define band file paths (Landsat-5 bands) - files are in the main directory
    red_path = f"{image_name}_B3.TIF"      # Band 3 (Red)
    nir_path = f"{image_name}_B4.TIF"       # Band 4 (NIR)
    green_path = f"{image_name}_B2.TIF"     # Band 2 (Green)
    swir_path = f"{image_name}_B5.TIF"      # Band 5 (SWIR)
    
    print(f"Processing {image_name}...")
    
    # Read bands and get metadata
    red_band, x_size, y_size, projection, geotransform, no_data_value = read_geotiff_band(red_path)
    nir_band, _, _, _, _, _ = read_geotiff_band(nir_path)
    green_band, _, _, _, _, _ = read_geotiff_band(green_path)
    swir_band, _, _, _, _, _ = read_geotiff_band(swir_path)
    
    # Calculate SAVI
    savi = calculate_savi(nir_band, red_band)
    
    # Calculate NDSI
    ndsi = calculate_ndsi(green_band, swir_band)
    
    # Save SAVI
    savi_output_path = os.path.join(output_folder, f"{image_name}_SAVI.TIF")
    write_geotiff(savi_output_path, x_size, y_size, projection, geotransform, no_data_value, savi)
    print(f"  SAVI saved: {savi_output_path}")
    
    # Save NDSI
    ndsi_output_path = os.path.join(output_folder, f"{image_name}_NDSI.TIF")
    write_geotiff(ndsi_output_path, x_size, y_size, projection, geotransform, no_data_value, ndsi)
    print(f"  NDSI saved: {ndsi_output_path}")
    
    return image_name

# Main processing
def main():
    # Define paths
    current_dir = os.getcwd()
    output_folder = os.path.join(current_dir, "assignment_5_output")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the two Landsat image names (files are in main directory)
    image_names = [
        "LT50290372010217EDC00",
        "LT50290372010233EDC00"
    ]
    
    print("Starting Assignment 1: SAVI and NDSI Calculation")
    print("=" * 50)
    
    # Process each image
    for image_name in image_names:
        # Check if the required band files exist
        required_files = [f"{image_name}_B2.TIF", f"{image_name}_B3.TIF", f"{image_name}_B4.TIF", f"{image_name}_B5.TIF"]
        if all(os.path.exists(f) for f in required_files):
            try:
                processed_name = process_landsat_image(image_name, output_folder)
                print(f"✓ Successfully processed {processed_name}")
            except Exception as e:
                print(f"✗ Error processing {image_name}: {str(e)}")
        else:
            print(f"✗ Required band files not found for {image_name}")
    
    print("=" * 50)
    print("Assignment 1 processing complete!")
    print(f"Results saved in: {output_folder}")

# Run the main function
if __name__ == "__main__":
    main()
