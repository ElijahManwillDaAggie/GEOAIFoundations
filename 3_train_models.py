"""
Train Machine Learning Models for Impervious Surface Classification
Weber River Watershed - Sentinel-2 Images

This script:
1. Loads training data from CSV (created by create_training_data.py)
2. Trains Random Forest and Decision Tree classifiers
3. Evaluates model performance
4. Saves trained models for inference
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("Impervious Surface Classification - Model Training")
print("=" * 60)

# Configuration
TRAINING_DATA_CSV = 'training_data/training_data.csv'
OUTPUT_FOLDER = 'models'
RF_MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'random_forest_model.pkl')
DT_MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'decision_tree_model.pkl')

# Feature names (bands and indices)
FEATURE_NAMES = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 
                 'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'IBI']

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load training data
print("\n1. Loading training data...")
if not os.path.exists(TRAINING_DATA_CSV):
    print(f"Error: Training data file not found: {TRAINING_DATA_CSV}")
    print("Please run create_training_data.py first to create training labels.")
    exit(1)

df = pd.read_csv(TRAINING_DATA_CSV)
print(f"   Loaded {len(df)} training samples")

# Extract features and labels
X = df[FEATURE_NAMES].values
y = df['label'].values

print(f"   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   Number of features: {len(FEATURE_NAMES)}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\n   Class distribution:")
for label, count in zip(unique, counts):
    label_name = "Impervious" if label == 1 else "Non-Impervious"
    print(f"     {label_name} ({label}): {count} samples ({100*count/len(y):.1f}%)")

# Split data into training and validation sets
print("\n2. Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")

# Initialize models
print("\n3. Initializing models...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
    verbose=1
)

dt_model = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

print("   Random Forest: 100 estimators, balanced class weights")
print("   Decision Tree: default parameters, balanced class weights")

# Train models
print("\n4. Training models...")
print("   Training Random Forest (this may take a few minutes)...")
rf_model.fit(X_train, y_train)
print("   ✓ Random Forest trained")

print("   Training Decision Tree...")
dt_model.fit(X_train, y_train)
print("   ✓ Decision Tree trained")

# Make predictions on validation set
print("\n5. Evaluating models on validation set...")
rf_pred = rf_model.predict(X_val)
dt_pred = dt_model.predict(X_val)

# Calculate metrics
def calculate_metrics(y_true, y_pred, model_name):
    """Calculate and print classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n   {model_name} Metrics:")
    print(f"     Accuracy:  {accuracy:.4f}")
    print(f"     Precision: {precision:.4f}")
    print(f"     Recall:    {recall:.4f}")
    print(f"     F1-Score: {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                Non-Imp  Imperv")
    print(f"     Actual Non-Imp  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"            Imperv   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

rf_metrics = calculate_metrics(y_val, rf_pred, "Random Forest")
dt_metrics = calculate_metrics(y_val, dt_pred, "Decision Tree")

# Cross-validation
print("\n6. Performing 5-fold cross-validation...")
print("   Random Forest cross-validation...")
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"   CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")

print("   Decision Tree cross-validation...")
dt_cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"   CV Accuracy: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std() * 2:.4f})")

# Feature importance
print("\n7. Feature importance analysis...")
rf_importance = rf_model.feature_importances_
dt_importance = dt_model.feature_importances_

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'RF_Importance': rf_importance,
    'DT_Importance': dt_importance
}).sort_values('RF_Importance', ascending=False)

print("\n   Top 5 Most Important Features (Random Forest):")
for idx, row in importance_df.head(5).iterrows():
    print(f"     {row['Feature']:8s}: {row['RF_Importance']:.4f}")

# Save feature importance
importance_df.to_csv(os.path.join(OUTPUT_FOLDER, 'feature_importance.csv'), index=False)
print(f"\n   Feature importance saved to: {os.path.join(OUTPUT_FOLDER, 'feature_importance.csv')}")

# Save models
print("\n8. Saving trained models...")
joblib.dump(rf_model, RF_MODEL_PATH)
joblib.dump(dt_model, DT_MODEL_PATH)
print(f"   ✓ Random Forest saved to: {RF_MODEL_PATH}")
print(f"   ✓ Decision Tree saved to: {DT_MODEL_PATH}")

# Create visualization of feature importance
print("\n9. Creating visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest feature importance
axes[0].barh(range(len(FEATURE_NAMES)), importance_df['RF_Importance'].values)
axes[0].set_yticks(range(len(FEATURE_NAMES)))
axes[0].set_yticklabels(importance_df['Feature'].values)
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest - Feature Importance')
axes[0].invert_yaxis()

# Decision Tree feature importance
dt_sorted = importance_df.sort_values('DT_Importance', ascending=False)
axes[1].barh(range(len(FEATURE_NAMES)), dt_sorted['DT_Importance'].values)
axes[1].set_yticks(range(len(FEATURE_NAMES)))
axes[1].set_yticklabels(dt_sorted['Feature'].values)
axes[1].set_xlabel('Importance')
axes[1].set_title('Decision Tree - Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'feature_importance.png'), dpi=150, bbox_inches='tight')
print(f"   ✓ Feature importance plot saved to: {os.path.join(OUTPUT_FOLDER, 'feature_importance.png')}")

# Confusion matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Random Forest confusion matrix
sns.heatmap(rf_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Random Forest - Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Decision Tree confusion matrix
sns.heatmap(dt_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Decision Tree - Confusion Matrix')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
print(f"   ✓ Confusion matrices saved to: {os.path.join(OUTPUT_FOLDER, 'confusion_matrices.png')}")

print("\n" + "=" * 60)
print("Model Training Complete!")
print("=" * 60)
print(f"\nModels saved to: {OUTPUT_FOLDER}/")
print(f"  - random_forest_model.pkl")
print(f"  - decision_tree_model.pkl")
print(f"  - feature_importance.csv")
print(f"  - feature_importance.png")
print(f"  - confusion_matrices.png")
print("\nNext steps:")
print("  1. Review model performance metrics above")
print("  2. Check feature importance to understand which bands/indices are most useful")
print("  3. Run inference script to apply models to full watershed images")

