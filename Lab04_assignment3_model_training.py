import os
import numpy as np
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib  # For saving the trained model

print("=== Assignment 3: Model Training ===")
print("Training models using chipped 128x128 dataset...")
print("=" * 50)

# Paths to your chipped image and label directories
image_dir = "chipped_image_folder"
label_dir = "chipped_label_folder"

# Lists to store data for model training
X = []  # Features
y = []  # Labels

print(f"Loading data from {len(os.listdir(image_dir))} chipped images...")

# Loop through the chipped image files
for filename in os.listdir(image_dir):
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)  # Assuming labels have the same filenames

        # Operate on image
        image_dataset = gdal.Open(image_path)
        image_array = image_dataset.ReadAsArray()  # Read in all bands together as 3d array
        
        # Reshape the image array to (pixels, bands)
        n_bands, n_rows, n_cols = image_array.shape
        image_array = image_array.reshape(n_bands, -1)  # reshape array
        image_array = image_array.T  # Transpose the array
        
        # Operate on label
        label_dataset = gdal.Open(label_path)
        label_array = label_dataset.ReadAsArray()
        label_array = label_array.flatten()
        
        # Append to lists
        X.append(image_array)
        y.append(label_array)
        
        # Clean up GDAL datasets
        image_dataset = None
        label_dataset = None

print(f"Loaded {len(X)} chipped images")
print(f"Total pixels: {sum(arr.shape[0] for arr in X)}")

# Concatenate all data
print("Concatenating all data...")
X = np.vstack(X)  # Shape: (total_valid_pixels, n_bands)
y = np.hstack(y)  # Shape: (total_valid_pixels,)

print(f"Final training data shape: {X.shape}")
print(f"Final label data shape: {y.shape}")
print(f"Number of unique classes: {len(np.unique(y))}")
print(f"Classes: {np.unique(y)}")

# Initialize the classifiers
print("\nInitializing classifiers...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
dt_model = DecisionTreeClassifier(random_state=42)

# Train the models using the valid data
print("\nTraining Random Forest model...")
print("This may take several minutes...")
rf_model.fit(X, y)
print("Random Forest model trained.")

print("\nTraining Decision Tree model...")
dt_model.fit(X, y)
print("Decision Tree model trained.")

# Save the trained models
print("\nSaving models...")
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(dt_model, 'decision_tree_model.pkl')

print("Models trained and saved successfully!")
print("Files saved:")
print("- random_forest_model.pkl")
print("- decision_tree_model.pkl")
