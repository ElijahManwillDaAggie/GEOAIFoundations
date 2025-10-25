import os
import numpy as np
from osgeo import gdal

print("=== Assignment 2: Array Transformation Analysis ===")
print("=" * 60)

# Read in the image *1* data with gdal, and print the original image array and its shape
print("\n1. READING IMAGE 1:")
print("-" * 30)
image1_path = "assignment2_arraytrans_Data/img1.tif"
image1_dataset = gdal.Open(image1_path)
orig_image1_array = image1_dataset.ReadAsArray()
print("Original image 1 array shape:", orig_image1_array.shape)
print("Original image 1 array:")
print(orig_image1_array)
print("Data type:", orig_image1_array.dtype)
print("Min value:", orig_image1_array.min())
print("Max value:", orig_image1_array.max())

# Reshape the image array and print and its shape
print("\n2. RESHAPING IMAGE 1:")
print("-" * 30)
n_bands, n_rows, n_cols = orig_image1_array.shape
reshape_image1_array = orig_image1_array.reshape(n_bands, -1)
print("Reshaped image 1 array shape:", reshape_image1_array.shape)
print("Reshaped image 1 array:")
print(reshape_image1_array)

# Transpose the array and print and its shape
print("\n3. TRANSPOSING IMAGE 1:")
print("-" * 30)
trans_reshape_image1_array = reshape_image1_array.T
print("Transposed image 1 array shape:", trans_reshape_image1_array.shape)
print("Transposed image 1 array:")
print(trans_reshape_image1_array)

# Read in the image *2* data with gdal, and print the original image array and its shape
print("\n4. READING IMAGE 2:")
print("-" * 30)
image2_path = "assignment2_arraytrans_Data/img2.tif"
image2_dataset = gdal.Open(image2_path)
orig_image2_array = image2_dataset.ReadAsArray()
print("Original image 2 array shape:", orig_image2_array.shape)
print("Original image 2 array:")
print(orig_image2_array)
print("Data type:", orig_image2_array.dtype)
print("Min value:", orig_image2_array.min())
print("Max value:", orig_image2_array.max())

# Reshape the image array and print and its shape
print("\n5. RESHAPING IMAGE 2:")
print("-" * 30)
n_bands, n_rows, n_cols = orig_image2_array.shape
reshape_image2_array = orig_image2_array.reshape(n_bands, -1)
print("Reshaped image 2 array shape:", reshape_image2_array.shape)
print("Reshaped image 2 array:")
print(reshape_image2_array)

# Transpose the array and print and its shape
print("\n6. TRANSPOSING IMAGE 2:")
print("-" * 30)
trans_reshape_image2_array = reshape_image2_array.T
print("Transposed image 2 array shape:", trans_reshape_image2_array.shape)
print("Transposed image 2 array:")
print(trans_reshape_image2_array)

# Create an empty list and append the two transposed image lists, print the list and its shape
print("\n7. COMBINING IMAGE ARRAYS:")
print("-" * 30)
img_data_list = []
img_data_list.append(trans_reshape_image1_array)
img_data_list.append(trans_reshape_image2_array)
print("Image data list length:", len(img_data_list))
print("First array in list shape:", img_data_list[0].shape)
print("Second array in list shape:", img_data_list[1].shape)

# Vstack to the list, print the list and its shape
print("\n8. VSTACKING IMAGE ARRAYS:")
print("-" * 30)
img_data_list_vstack = np.vstack(img_data_list)
print("Vstacked image arrays shape:", img_data_list_vstack.shape)
print("Vstacked image arrays:")
print(img_data_list_vstack)

# Read in the label *1* data with gdal, and print the original label array and its shape
print("\n9. READING LABEL 1:")
print("-" * 30)
label1_path = "assignment2_arraytrans_Data/label1.tif"
label1_dataset = gdal.Open(label1_path)
orig_label1_array = label1_dataset.ReadAsArray()
print("Original label 1 array shape:", orig_label1_array.shape)
print("Original label 1 array:")
print(orig_label1_array)
print("Data type:", orig_label1_array.dtype)
print("Unique values:", np.unique(orig_label1_array))

# Flatten the label array and print and its shape
print("\n10. FLATTENING LABEL 1:")
print("-" * 30)
flatten_label1_array = orig_label1_array.flatten()
print("Flattened label 1 array shape:", flatten_label1_array.shape)
print("Flattened label 1 array:")
print(flatten_label1_array)

# Read in the label *2* data with gdal, and print the original label array and its shape
print("\n11. READING LABEL 2:")
print("-" * 30)
label2_path = "assignment2_arraytrans_Data/label2.tif"
label2_dataset = gdal.Open(label2_path)
orig_label2_array = label2_dataset.ReadAsArray()
print("Original label 2 array shape:", orig_label2_array.shape)
print("Original label 2 array:")
print(orig_label2_array)
print("Data type:", orig_label2_array.dtype)
print("Unique values:", np.unique(orig_label2_array))

# Flatten the label array and print and its shape
print("\n12. FLATTENING LABEL 2:")
print("-" * 30)
flatten_label2_array = orig_label2_array.flatten()
print("Flattened label 2 array shape:", flatten_label2_array.shape)
print("Flattened label 2 array:")
print(flatten_label2_array)

# Create an empty list and append the two flattened label lists, print the list and its shape
print("\n13. COMBINING LABEL ARRAYS:")
print("-" * 30)
label_data_list = []
label_data_list.append(flatten_label1_array)
label_data_list.append(flatten_label2_array)
print("Label data list length:", len(label_data_list))
print("First label array in list shape:", label_data_list[0].shape)
print("Second label array in list shape:", label_data_list[1].shape)

# Hstack to the list, print the list and its shape
print("\n14. HSTACKING LABEL ARRAYS:")
print("-" * 30)
label_data_list_hstack = np.hstack(label_data_list)
print("Hstacked label arrays shape:", label_data_list_hstack.shape)
print("Hstacked label arrays:")
print(label_data_list_hstack)

# Clean up datasets
image1_dataset = None
image2_dataset = None
label1_dataset = None
label2_dataset = None

print("\n" + "=" * 60)
print("Array transformation analysis complete!")
print("=" * 60)
