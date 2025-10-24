# Assignment 4: Band Extraction Script
# Extract bands 5, 4, and 3 from Stacked_8bands.TIF and save as individual files

from osgeo import gdal
import os

# Define the input file path (the provided stacked file)
input_file = '/Users/elijahmanwill/Documents/geoairemotesensing/Lab_01/Stacked_8bands_1.TIF'
output_directory = '/Users/elijahmanwill/Documents/geoairemotesensing/Lab_01/'

# Open the input stacked file
input_dataset = gdal.Open(input_file)

if input_dataset is None:
    print(f"Error: Could not open {input_file}")
else:
    print(f"Successfully opened {input_file}")
    
    # Get basic information about the input file
    x_size = input_dataset.RasterXSize
    y_size = input_dataset.RasterYSize
    num_bands = input_dataset.RasterCount
    projection = input_dataset.GetProjection()
    geotransform = input_dataset.GetGeoTransform()
    data_type = input_dataset.GetRasterBand(1).DataType
    no_data_value = input_dataset.GetRasterBand(1).GetNoDataValue()
    
    print(f"Image properties:")
    print(f"Dimensions: {x_size} x {y_size}")
    print(f"Number of bands: {num_bands}")
    print(f"Data type: {data_type}")
    print(f"No data value: {no_data_value}")
    
    # Define the bands to extract (5, 4, 3)
    bands_to_extract = [5, 4, 3]
    
    # Create GTiff driver
    driver = gdal.GetDriverByName('GTiff')
    
    # Extract each band
    for band_num in bands_to_extract:
        if band_num <= num_bands:
            print(f"\nExtracting band {band_num}...")
            
            # Read the band data
            band_data = input_dataset.GetRasterBand(band_num).ReadAsArray()
            
            # Create output file path
            output_path = os.path.join(output_directory, f'band{band_num}.TIF')
            
            # Create the output file
            output_dataset = driver.Create(output_path, x_size, y_size, 1, data_type)
            
            # Set the projection and geotransform
            output_dataset.SetProjection(projection)
            output_dataset.SetGeoTransform(geotransform)
            
            # Write the band data
            output_dataset.GetRasterBand(1).WriteArray(band_data)
            output_dataset.GetRasterBand(1).SetNoDataValue(no_data_value)
            
            # Save and close
            output_dataset.FlushCache()
            output_dataset = None
            
            print(f"Band {band_num} saved as {output_path}")
        else:
            print(f"Warning: Band {band_num} does not exist in the input file (only {num_bands} bands available)")
    
    # Close the input dataset
    input_dataset = None
    print("\nBand extraction completed successfully!")
    print("Extracted files:")
    print("- band5.TIF (Near Infrared - for Red channel in false color)")
    print("- band4.TIF (Red - for Green channel in false color)")  
    print("- band3.TIF (Green - for Blue channel in false color)")
    print("\nYou can now add these bands in QGIS and create a virtual raster with RGB assignment.")
