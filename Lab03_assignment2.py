# Assignment 2: DN Matching for Radiometric Correction
# This script performs DN matching between two Landsat-5 images using stable areas

from osgeo import gdal
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def stack_bands(image_name, bands_to_stack):
    """
    Stack multiple bands into a single multi-band GeoTIFF file.
    bands_to_stack: list of band numbers to stack (e.g., [3, 4, 5] for Red, NIR, SWIR1)
    """
    print(f"Stacking bands for {image_name}...")
    
    # Read first band to get metadata
    first_band_path = f"{image_name}_B{bands_to_stack[0]}.TIF"
    first_band, x_size, y_size, projection, geotransform, no_data_value = read_geotiff_band(first_band_path)
    
    # Create stacked array (note: numpy arrays are (height, width) not (width, height))
    stacked_array = np.zeros((len(bands_to_stack), y_size, x_size), dtype=np.float32)
    stacked_array[0] = first_band
    
    # Read remaining bands
    for i, band_num in enumerate(bands_to_stack[1:], 1):
        band_path = f"{image_name}_B{band_num}.TIF"
        band_array, _, _, _, _, _ = read_geotiff_band(band_path)
        stacked_array[i] = band_array
    
    # Save stacked image
    output_path = f"{image_name}_stacked.tif"
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(output_path, x_size, y_size, len(bands_to_stack), gdal.GDT_Float32)
    
    output_dataset.SetProjection(projection)
    output_dataset.SetGeoTransform(geotransform)
    
    for i in range(len(bands_to_stack)):
        output_band = output_dataset.GetRasterBand(i + 1)
        output_band.WriteArray(stacked_array[i])
        if no_data_value is not None:
            output_band.SetNoDataValue(no_data_value)
    
    output_dataset.FlushCache()
    output_dataset = None
    
    print(f"  Stacked image saved: {output_path}")
    return output_path, stacked_array, x_size, y_size, projection, geotransform, no_data_value

def clip_stable_areas(image_path, dark_coords, bright_coords, output_folder):
    """
    Clip stable dark and bright areas from the stacked image.
    dark_coords: (x1, y1, x2, y2) for dark area
    bright_coords: (x1, y1, x2, y2) for bright area
    """
    print("Clipping stable areas...")
    
    # Open the stacked image
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open {image_path}")
    
    # Get image dimensions
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    num_bands = dataset.RasterCount
    
    # Read all bands
    bands_data = []
    for i in range(1, num_bands + 1):
        band_data = dataset.GetRasterBand(i).ReadAsArray()
        bands_data.append(band_data)
    
    # Clip dark area
    dark_x1, dark_y1, dark_x2, dark_y2 = dark_coords
    dark_clip = []
    for band_data in bands_data:
        dark_band = band_data[dark_y1:dark_y2, dark_x1:dark_x2]
        dark_clip.append(dark_band)
    
    # Clip bright area
    bright_x1, bright_y1, bright_x2, bright_y2 = bright_coords
    bright_clip = []
    for band_data in bands_data:
        bright_band = band_data[bright_y1:bright_y2, bright_x1:bright_x2]
        bright_clip.append(bright_band)
    
    dataset = None
    
    print(f"  Dark area clipped: {dark_x1},{dark_y1} to {dark_x2},{dark_y2}")
    print(f"  Bright area clipped: {bright_x1},{bright_y1} to {bright_x2},{bright_y2}")
    
    return dark_clip, bright_clip

def calculate_linear_regression(dark_avg_ref, dark_avg_target, bright_avg_ref, bright_avg_target):
    """
    Calculate linear regression parameters (slope and intercept) for DN correction.
    Returns: slope (a) and intercept (b) for equation: y = ax + b
    """
    # Calculate slope: a = (y2 - y1) / (x2 - x1)
    slope = (bright_avg_target - dark_avg_target) / (bright_avg_ref - dark_avg_ref)
    
    # Calculate intercept: b = y1 - a * x1
    intercept = dark_avg_target - slope * dark_avg_ref
    
    return slope, intercept

def apply_dn_correction(band_data, slope, intercept):
    """
    Apply DN correction using linear regression parameters.
    New_DN = slope * Original_DN + intercept
    """
    corrected_band = slope * band_data + intercept
    return corrected_band

def process_dn_matching():
    """
    Main function to process DN matching between two Landsat images.
    """
    print("Starting Assignment 2: DN Matching for Radiometric Correction")
    print("=" * 60)
    
    # Define image names and bands of interest
    image_names = ["LT50290372010217EDC00", "LT50290372010233EDC00"]
    bands_of_interest = [3, 4, 5]  # Red, NIR, SWIR1
    band_names = ["Red", "NIR", "SWIR1"]
    
    # Create output folder
    output_folder = "assignment_2_output"
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Stack bands for both images
    print("\nStep 1: Stacking bands...")
    stacked_paths = []
    stacked_data = []
    
    for image_name in image_names:
        stacked_path, stacked_array, x_size, y_size, projection, geotransform, no_data_value = stack_bands(
            image_name, bands_of_interest
        )
        stacked_paths.append(stacked_path)
        stacked_data.append(stacked_array)
    
    # Step 2: Define stable areas (you'll need to adjust these coordinates based on your images)
    # These are example coordinates - you should identify actual stable areas in your images
    print("\nStep 2: Defining stable areas...")
    
    # Dark area coordinates (adjust based on your image analysis)
    dark_coords = (100, 100, 200, 200)  # (x1, y1, x2, y2)
    
    # Bright area coordinates (adjust based on your image analysis)  
    bright_coords = (300, 300, 400, 400)  # (x1, y1, x2, y2)
    
    print(f"  Dark area: {dark_coords}")
    print(f"  Bright area: {bright_coords}")
    
    # Step 3: Clip stable areas from both images
    print("\nStep 3: Clipping stable areas...")
    dark_areas = []
    bright_areas = []
    
    for i, stacked_path in enumerate(stacked_paths):
        dark_clip, bright_clip = clip_stable_areas(stacked_path, dark_coords, bright_coords, output_folder)
        dark_areas.append(dark_clip)
        bright_areas.append(bright_clip)
    
    # Step 4: Calculate average DN values for each band
    print("\nStep 4: Calculating average DN values...")
    
    # Reference image (less cloudy - you may need to determine which is less cloudy)
    ref_idx = 0  # Assuming first image is reference
    target_idx = 1
    
    print(f"  Reference image: {image_names[ref_idx]}")
    print(f"  Target image: {image_names[target_idx]}")
    
    # Calculate averages for reference image
    ref_dark_avgs = [np.mean(dark_areas[ref_idx][i]) for i in range(len(bands_of_interest))]
    ref_bright_avgs = [np.mean(bright_areas[ref_idx][i]) for i in range(len(bands_of_interest))]
    
    # Calculate averages for target image
    target_dark_avgs = [np.mean(dark_areas[target_idx][i]) for i in range(len(bands_of_interest))]
    target_bright_avgs = [np.mean(bright_areas[target_idx][i]) for i in range(len(bands_of_interest))]
    
    print("  Average DN values:")
    for i, band_name in enumerate(band_names):
        print(f"    {band_name} Band:")
        print(f"      Reference - Dark: {ref_dark_avgs[i]:.2f}, Bright: {ref_bright_avgs[i]:.2f}")
        print(f"      Target - Dark: {target_dark_avgs[i]:.2f}, Bright: {target_bright_avgs[i]:.2f}")
    
    # Step 5: Calculate linear regression parameters for each band
    print("\nStep 5: Calculating linear regression parameters...")
    slopes = []
    intercepts = []
    
    for i, band_name in enumerate(band_names):
        slope, intercept = calculate_linear_regression(
            ref_dark_avgs[i], target_dark_avgs[i],
            ref_bright_avgs[i], target_bright_avgs[i]
        )
        slopes.append(slope)
        intercepts.append(intercept)
        print(f"  {band_name} Band: slope={slope:.4f}, intercept={intercept:.2f}")
    
    # Step 6: Apply DN correction to target image
    print("\nStep 6: Applying DN correction...")
    
    # Read target image bands
    target_bands = []
    for band_num in bands_of_interest:
        band_path = f"{image_names[target_idx]}_B{band_num}.TIF"
        band_data, _, _, _, _, _ = read_geotiff_band(band_path)
        target_bands.append(band_data)
    
    # Apply correction to each band
    corrected_bands = []
    for i, (band_data, slope, intercept) in enumerate(zip(target_bands, slopes, intercepts)):
        corrected_band = apply_dn_correction(band_data, slope, intercept)
        corrected_bands.append(corrected_band)
        
        # Save corrected band
        output_path = os.path.join(output_folder, f"{image_names[target_idx]}_corrected_{band_names[i]}.TIF")
        write_geotiff(output_path, x_size, y_size, projection, geotransform, no_data_value, corrected_band)
        print(f"  Corrected {band_names[i]} band saved: {output_path}")
    
    # Step 7: Create comparison visualization
    print("\nStep 7: Creating comparison visualization...")
    
    # Create a simple comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DN Matching Results Comparison', fontsize=16)
    
    for i, band_name in enumerate(band_names):
        # Original target band
        axes[0, i].imshow(target_bands[i], cmap='viridis')
        axes[0, i].set_title(f'Original {band_name}')
        axes[0, i].axis('off')
        
        # Corrected target band
        axes[1, i].imshow(corrected_bands[i], cmap='viridis')
        axes[1, i].set_title(f'Corrected {band_name}')
        axes[1, i].axis('off')
    
    # Add rectangles to show stable areas
    for ax in axes.flat:
        # Dark area rectangle
        dark_rect = Rectangle((dark_coords[0], dark_coords[1]), 
                             dark_coords[2]-dark_coords[0], 
                             dark_coords[3]-dark_coords[1],
                             linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(dark_rect)
        
        # Bright area rectangle
        bright_rect = Rectangle((bright_coords[0], bright_coords[1]), 
                               bright_coords[2]-bright_coords[0], 
                               bright_coords[3]-bright_coords[1],
                               linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(bright_rect)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_folder, 'dn_matching_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"  Comparison plot saved: {comparison_path}")
    
    print("\n" + "=" * 60)
    print("Assignment 2: DN Matching Complete!")
    print(f"Results saved in: {output_folder}")
    print("\nNext steps:")
    print("1. Review the stable area coordinates and adjust if needed")
    print("2. Open corrected images in QGIS for detailed analysis")
    print("3. Take screenshots of comparison maps")
    print("4. Push code and results to GitHub")

# Run the DN matching process
if __name__ == "__main__":
    process_dn_matching()
