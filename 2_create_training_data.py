"""
Create Training Data for Impervious Surface Classification
Weber River Watershed - Sentinel-2 Images

This script allows you to:
1. Load and visualize Sentinel-2 images
2. Interactively label points as Impervious (1) or Non-Impervious (0)
3. Extract pixel values from all bands and indices
4. Save training data for machine learning models
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from pathlib import Path

# Configuration
TIFF_FOLDER = 'tiff_images'
OUTPUT_FOLDER = 'training_data'
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, 'training_data.csv')
OUTPUT_LABELS = os.path.join(OUTPUT_FOLDER, 'labeled_points.json')

# Band names in the GeoTIFF files
BAND_NAMES = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 
              'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'IBI']

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class TrainingDataCreator:
    """Interactive tool for creating training data from Sentinel-2 images"""
    
    def __init__(self, tiff_folder=TIFF_FOLDER):
        self.tiff_folder = tiff_folder
        self.tiff_files = self._get_tiff_files()
        self.current_image_idx = 0
        self.current_image = None
        self.current_data = None
        self.current_transform = None
        self.labeled_points = []
        self.fig = None
        self.ax = None
        
    def _get_tiff_files(self):
        """Get list of all GeoTIFF files in the folder"""
        tiff_files = sorted([f for f in os.listdir(self.tiff_folder) 
                            if f.endswith('.tif')])
        print(f"Found {len(tiff_files)} GeoTIFF files:")
        for f in tiff_files:
            print(f"  - {f}")
        return tiff_files
    
    def load_image(self, image_idx=None):
        """Load a GeoTIFF image"""
        if image_idx is None:
            image_idx = self.current_image_idx
        
        if image_idx >= len(self.tiff_files):
            print(f"Image index {image_idx} out of range. Using last image.")
            image_idx = len(self.tiff_files) - 1
        
        self.current_image_idx = image_idx
        tiff_path = os.path.join(self.tiff_folder, self.tiff_files[image_idx])
        
        print(f"\nLoading: {self.tiff_files[image_idx]}")
        
        with rasterio.open(tiff_path) as src:
            # Read all bands
            self.current_data = src.read()
            self.current_transform = src.transform
            self.current_crs = src.crs
            self.current_bounds = src.bounds
            
            print(f"  Shape: {self.current_data.shape}")
            print(f"  Bands: {len(BAND_NAMES)}")
            print(f"  CRS: {self.current_crs}")
            print(f"  Bounds: {self.current_bounds}")
        
        return self.current_data
    
    def display_image(self, composite_type='rgb', band_indices=None):
        """
        Display the image for labeling
        
        Parameters:
        -----------
        composite_type : str
            'rgb' for true color, 'false_color' for NIR-Red-Green, 
            'ndvi' for NDVI, 'ibi' for IBI
        band_indices : tuple
            Custom band indices (R, G, B) for display
        """
        if self.current_data is None:
            print("No image loaded. Loading first image...")
            self.load_image()
        
        # Select bands for display
        if composite_type == 'rgb':
            # True color: B4 (Red), B3 (Green), B2 (Blue)
            display_bands = [3, 2, 1]  # B4, B3, B2 (0-indexed)
        elif composite_type == 'false_color':
            # False color: B8 (NIR), B4 (Red), B3 (Green)
            display_bands = [6, 3, 2]  # B8, B4, B3
        elif composite_type == 'ndvi':
            # NDVI single band
            display_bands = [6]  # NDVI is band 7 (0-indexed)
        elif composite_type == 'ibi':
            # IBI single band
            display_bands = [11]  # IBI is band 12 (0-indexed)
        elif band_indices:
            display_bands = band_indices
        else:
            display_bands = [3, 2, 1]  # Default to RGB
        
        # Prepare data for display
        if len(display_bands) == 1:
            # Single band - use colormap
            display_data = self.current_data[display_bands[0]]
            cmap = 'RdYlGn' if composite_type == 'ndvi' else 'RdYlBu'
        else:
            # Multi-band - stack and normalize
            display_data = np.dstack([
                self.current_data[b] for b in display_bands
            ])
            # Normalize to 0-1 range for display
            for i in range(display_data.shape[2]):
                band = display_data[:, :, i]
                band_min = np.nanpercentile(band, 2)
                band_max = np.nanpercentile(band, 98)
                display_data[:, :, i] = np.clip(
                    (band - band_min) / (band_max - band_min), 0, 1
                )
        
        # Create figure
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        else:
            self.ax.clear()
        
        # Display image
        if len(display_bands) == 1:
            im = self.ax.imshow(display_data, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im, ax=self.ax, label=composite_type.upper())
        else:
            self.ax.imshow(display_data)
        
        self.ax.set_title(f"{self.tiff_files[self.current_image_idx]} - {composite_type.upper()}")
        self.ax.set_xlabel("Pixel X")
        self.ax.set_ylabel("Pixel Y")
        
        # Show existing labels
        self._plot_existing_labels()
        
        plt.tight_layout()
        plt.show(block=False)
        
    def _plot_existing_labels(self):
        """Plot already labeled points on the current image"""
        for point in self.labeled_points:
            if point['image'] == self.tiff_files[self.current_image_idx]:
                color = 'red' if point['label'] == 1 else 'blue'
                marker = 'X' if point['label'] == 1 else 'o'
                self.ax.plot(point['pixel_x'], point['pixel_y'], 
                            color=color, marker=marker, markersize=10,
                            markeredgewidth=2, markeredgecolor='white',
                            label='Impervious' if point['label'] == 1 else 'Non-Impervious')
        
        # Add legend if there are labels
        if any(p['image'] == self.tiff_files[self.current_image_idx] 
               for p in self.labeled_points):
            self.ax.legend()
    
    def label_point_interactive(self, pixel_x, pixel_y, label):
        """
        Label a point at pixel coordinates
        
        Parameters:
        -----------
        pixel_x, pixel_y : int
            Pixel coordinates
        label : int
            1 for Impervious, 0 for Non-Impervious
        """
        # Validate coordinates
        if (pixel_x < 0 or pixel_x >= self.current_data.shape[2] or
            pixel_y < 0 or pixel_y >= self.current_data.shape[1]):
            print(f"Error: Coordinates ({pixel_x}, {pixel_y}) out of bounds")
            return False
        
        # Extract pixel values from all bands
        pixel_values = {}
        for i, band_name in enumerate(BAND_NAMES):
            pixel_values[band_name] = float(self.current_data[i, pixel_y, pixel_x])
        
        # Convert pixel coordinates to geographic coordinates
        lon, lat = rasterio.transform.xy(
            self.current_transform, pixel_y, pixel_x
        )
        
        # Create label entry
        label_entry = {
            'image': self.tiff_files[self.current_image_idx],
            'pixel_x': int(pixel_x),
            'pixel_y': int(pixel_y),
            'lon': float(lon),
            'lat': float(lat),
            'label': int(label),
            'pixel_values': pixel_values
        }
        
        # Check if point already exists (within 5 pixels)
        for existing in self.labeled_points:
            if (existing['image'] == label_entry['image'] and
                abs(existing['pixel_x'] - label_entry['pixel_x']) < 5 and
                abs(existing['pixel_y'] - label_entry['pixel_y']) < 5):
                print(f"Point near ({pixel_x}, {pixel_y}) already labeled. Updating...")
                existing.update(label_entry)
                return True
        
        # Add new label
        self.labeled_points.append(label_entry)
        label_type = "Impervious" if label == 1 else "Non-Impervious"
        print(f"Labeled point ({pixel_x}, {pixel_y}) as {label_type}")
        return True
    
    def label_point_manual(self):
        """Manually input coordinates to label"""
        print("\n=== Manual Point Labeling ===")
        print(f"Current image: {self.tiff_files[self.current_image_idx]}")
        print(f"Image dimensions: {self.current_data.shape[1]} x {self.current_data.shape[2]}")
        
        try:
            pixel_x = int(input("Enter pixel X coordinate: "))
            pixel_y = int(input("Enter pixel Y coordinate: "))
            label = int(input("Enter label (1=Impervious, 0=Non-Impervious): "))
            
            if label not in [0, 1]:
                print("Error: Label must be 0 or 1")
                return False
            
            return self.label_point_interactive(pixel_x, pixel_y, label)
        except ValueError:
            print("Error: Invalid input")
            return False
    
    def save_labels(self):
        """Save labeled points to JSON and CSV"""
        if not self.labeled_points:
            print("No labels to save.")
            return
        
        # Save as JSON (preserves all information)
        with open(OUTPUT_LABELS, 'w') as f:
            json.dump(self.labeled_points, f, indent=2)
        print(f"\nSaved {len(self.labeled_points)} labeled points to {OUTPUT_LABELS}")
        
        # Save as CSV (for easy ML processing)
        rows = []
        for point in self.labeled_points:
            row = {
                'image': point['image'],
                'pixel_x': point['pixel_x'],
                'pixel_y': point['pixel_y'],
                'lon': point['lon'],
                'lat': point['lat'],
                'label': point['label']
            }
            # Add all band values
            row.update(point['pixel_values'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved training data to {OUTPUT_CSV}")
        print(f"\nTraining data summary:")
        print(f"  Total samples: {len(df)}")
        print(f"  Impervious (1): {len(df[df['label'] == 1])}")
        print(f"  Non-Impervious (0): {len(df[df['label'] == 0])}")
        print(f"  Features: {len(BAND_NAMES)} bands/indices")
    
    def load_labels(self):
        """Load previously saved labels"""
        if os.path.exists(OUTPUT_LABELS):
            with open(OUTPUT_LABELS, 'r') as f:
                self.labeled_points = json.load(f)
            print(f"Loaded {len(self.labeled_points)} existing labels")
            return True
        return False
    
    def get_training_data(self):
        """Get training data as numpy arrays (X, y)"""
        if not self.labeled_points:
            print("No labels available. Please label some points first.")
            return None, None
        
        # Extract features and labels
        X = []
        y = []
        
        for point in self.labeled_points:
            # Get pixel values in order of BAND_NAMES
            features = [point['pixel_values'][band] for band in BAND_NAMES]
            X.append(features)
            y.append(point['label'])
        
        return np.array(X), np.array(y)


def main():
    """Main function for interactive labeling"""
    print("=" * 60)
    print("Training Data Creator for Impervious Surface Classification")
    print("=" * 60)
    
    # Initialize creator
    creator = TrainingDataCreator()
    
    # Load existing labels if they exist
    creator.load_labels()
    
    # Load first image
    creator.load_image(0)
    
    print("\n" + "=" * 60)
    print("INSTRUCTIONS:")
    print("=" * 60)
    print("1. Display images using: creator.display_image('rgb')")
    print("   Options: 'rgb', 'false_color', 'ndvi', 'ibi'")
    print("2. Label points using: creator.label_point_manual()")
    print("3. Switch images: creator.load_image(image_index)")
    print("4. Save labels: creator.save_labels()")
    print("5. Get training data: X, y = creator.get_training_data()")
    print("\nExample workflow:")
    print("  creator.display_image('rgb')")
    print("  creator.label_point_manual()  # Repeat for multiple points")
    print("  creator.save_labels()")
    print("=" * 60)
    
    # Interactive loop
    while True:
        print("\nOptions:")
        print("1. Display image (RGB)")
        print("2. Display image (False Color)")
        print("3. Display image (NDVI)")
        print("4. Display image (IBI)")
        print("5. Label point manually")
        print("6. Switch to next image")
        print("7. Switch to previous image")
        print("8. Save labels")
        print("9. Show statistics")
        print("10. Exit")
        
        choice = input("\nEnter choice (1-10): ").strip()
        
        if choice == '1':
            creator.display_image('rgb')
        elif choice == '2':
            creator.display_image('false_color')
        elif choice == '3':
            creator.display_image('ndvi')
        elif choice == '4':
            creator.display_image('ibi')
        elif choice == '5':
            creator.label_point_manual()
        elif choice == '6':
            creator.current_image_idx = (creator.current_image_idx + 1) % len(creator.tiff_files)
            creator.load_image()
            print(f"Switched to: {creator.tiff_files[creator.current_image_idx]}")
        elif choice == '7':
            creator.current_image_idx = (creator.current_image_idx - 1) % len(creator.tiff_files)
            creator.load_image()
            print(f"Switched to: {creator.tiff_files[creator.current_image_idx]}")
        elif choice == '8':
            creator.save_labels()
        elif choice == '9':
            if creator.labeled_points:
                total = len(creator.labeled_points)
                impervious = sum(1 for p in creator.labeled_points if p['label'] == 1)
                non_impervious = total - impervious
                print(f"\nLabeling Statistics:")
                print(f"  Total points: {total}")
                print(f"  Impervious (1): {impervious} ({100*impervious/total:.1f}%)")
                print(f"  Non-Impervious (0): {non_impervious} ({100*non_impervious/total:.1f}%)")
            else:
                print("No labels yet.")
        elif choice == '10':
            if creator.labeled_points:
                save = input("Save labels before exiting? (y/n): ").strip().lower()
                if save == 'y':
                    creator.save_labels()
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1-10.")


if __name__ == "__main__":
    main()

