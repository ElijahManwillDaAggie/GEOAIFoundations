# Assignment 2: Band Stacking Script
# Modified script to stack B1, B2, B3, B4, B5, B6, B7, and B9 into one TIF file
# while keeping the original projection, spatial resolution, and no-data value

from osgeo import gdal
import glob
import os

# Define the directory containing the Landsat band files
data_directory = '/Users/elijahmanwill/Documents/geoairemotesensing/Lab_01/landsat_data/'

# Use glob to fetch all required band files (B1-B7 and B9)
# Note: B8 (panchromatic) and B10-B11 (thermal) are excluded as per assignment requirements
band_files = glob.glob(os.path.join(data_directory, '*_B[1-7,9].TIF'))

# Sort the list to ensure proper band order
band_files = sorted(band_files)

print(f"Found {len(band_files)} band files:")
for i, file in enumerate(band_files, 1):
    print(f"Band {i}: {os.path.basename(file)}")

# Open the first band to get dimensions and geospatial information
first_band = gdal.Open(band_files[0])
x_size = first_band.RasterXSize
y_size = first_band.RasterYSize
projection = first_band.GetProjection()
geotransform = first_band.GetGeoTransform()
data_type = first_band.GetRasterBand(1).DataType
no_data_value = first_band.GetRasterBand(1).GetNoDataValue()

print(f"\nImage properties:")
print(f"x_size: {x_size}, y_size: {y_size}")
print(f"data_type: {data_type}, no_data_value: {no_data_value}")

# Define output file path
output_path = os.path.join(data_directory, 'Stacked_L8_8bands_elijah.TIF')

# Create the output file with 8 bands
driver = gdal.GetDriverByName('GTiff')
output_dataset = driver.Create(output_path, x_size, y_size, 8, data_type)

# Set the projection and geotransform of the output file
output_dataset.SetProjection(projection)
output_dataset.SetGeoTransform(geotransform)

# Process each band using a for loop
for i, band_file in enumerate(band_files, 1):
    print(f"Processing band {i}: {os.path.basename(band_file)}")
    
    # Open the band file
    band = gdal.Open(band_file)
    band_data = band.GetRasterBand(1).ReadAsArray()
    
    # Write the data to the corresponding band in the output file
    output_dataset.GetRasterBand(i).WriteArray(band_data)
    output_dataset.GetRasterBand(i).SetNoDataValue(no_data_value)

# Save and close the output file
output_dataset.FlushCache()
output_dataset = None

print(f"\nStacked image saved as {output_path}")
print("Band stacking completed successfully!")
print("\nStacked bands:")
print("Band 1: B1 (Coastal/Aerosol)")
print("Band 2: B2 (Blue)")
print("Band 3: B3 (Green)")
print("Band 4: B4 (Red)")
print("Band 5: B5 (Near Infrared)")
print("Band 6: B6 (SWIR1)")
print("Band 7: B7 (SWIR2)")
print("Band 8: B9 (Cirrus)")
