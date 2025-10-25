import os
from osgeo import gdal

def chip_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir, chip_size=128):
    """
    Chip images and labels into smaller tiles while maintaining geo-transform.
    
    Parameters:
    - image_dir: Path to directory containing image files
    - label_dir: Path to directory containing label files  
    - output_image_dir: Path to save chipped images
    - output_label_dir: Path to save chipped labels
    - chip_size: Size of chips (default 128x128)
    """
    
    # Create output directories if they don't exist
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    print(f"Processing {len(image_files)} image files...")
    
    for image_file in image_files:
        print(f"Processing {image_file}...")
        
        # Construct file paths
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file)  # Same filename
        
        # Get base name without extension
        base_name = os.path.splitext(image_file)[0]
        
        # Open the input GeoTIFFs with gdal
        image_dataset = gdal.Open(image_path)
        label_dataset = gdal.Open(label_path)
        
        if image_dataset is None or label_dataset is None:
            print(f"Warning: Could not open {image_file} or its label. Skipping...")
            continue
            
        # Get the size of the input images
        x_size = image_dataset.RasterXSize
        y_size = image_dataset.RasterYSize
        
        # Determine how many chips can be generated
        num_x_chips = x_size // chip_size
        num_y_chips = y_size // chip_size
        
        print(f"  Image size: {x_size}x{y_size}, generating {num_x_chips}x{num_y_chips} chips")
        
        # Use sliding window to chip the images
        for x in range(num_x_chips):
            for y in range(num_y_chips):
                # Calculate the pixel offsets for the current chip
                x_offset = x * chip_size
                y_offset = y * chip_size
                
                # Define the output file names
                output_image_file = os.path.join(output_image_dir, f"{base_name}_{x}_{y}.tif")
                output_label_file = os.path.join(output_label_dir, f"{base_name}_{x}_{y}.tif")
                
                # Use gdal.Translate to create the chips for both image and label
                gdal.Translate(output_image_file, image_dataset, 
                              srcWin=[x_offset, y_offset, chip_size, chip_size])
                gdal.Translate(output_label_file, label_dataset, 
                              srcWin=[x_offset, y_offset, chip_size, chip_size])
        
        # Close the input datasets
        image_dataset.FlushCache()
        label_dataset.FlushCache()
        image_dataset = None
        label_dataset = None
    
    print(f"Chipping completed! Files saved in:")
    print(f"  Images: {output_image_dir}")
    print(f"  Labels: {output_label_dir}")

if __name__ == "__main__":
    # Define paths
    image_dir = "assignment1_train_Data/images"
    label_dir = "assignment1_train_Data/labels"
    output_image_dir = "chipped_image_folder"
    output_label_dir = "chipped_label_folder"
    
    # Run the chipping function
    chip_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir, chip_size=128)
