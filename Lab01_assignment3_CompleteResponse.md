# Assignment 3: False Color Composite Analysis
**Student Name**: Elijah Manwill  
**Date**: 10/24/2025  
**Course**: GeoAI Remote Sensing Lab 1

## Assignment Overview

This assignment demonstrates the use of false color composites in QGIS to visualize Landsat-8 data. The goal is to create two different false color composites where vegetation appears in different colors (red and yellow) and explain the scientific reasoning behind each combination.

## Data Used

- **Input File**: `Stacked_L8_8bands_elijah.TIF` (8-band stacked Landsat-8 image)
- **Location**: Louisiana/Mississippi area (Path 022, Row 039)
- **Date**: October 2, 2020
- **Software**: QGIS for visualization and composite creation

## False Color Composite 1: Vegetation Appears Red

<img src="images/false_color_comp_red.png">


### Band Assignment
- **Red Channel**: Band 5 (Near Infrared, 0.85-0.88 μm)
- **Green Channel**: Band 4 (Red, 0.64-0.67 μm)  
- **Blue Channel**: Band 3 (Green, 0.53-0.59 μm)

### Why Vegetation Appears Red

This is the classic "False Color Infrared" composite (NIR-Red-Green). Healthy vegetation has high reflectance in the near-infrared portion of the spectrum (Band 5) due to the internal structure of plant cells. When we assign the NIR band to the red channel, healthy vegetation appears bright red or magenta. This occurs because:

- **Chlorophyll absorption**: Chlorophyll absorbs red and blue light for photosynthesis, but reflects near-infrared light
- **Leaf structure**: The spongy mesophyll layer in leaves reflects near-infrared light strongly
- **Vegetation health**: Stressed or unhealthy vegetation appears darker red or brown
- **Water contrast**: Water bodies appear dark blue/black due to strong absorption of NIR

### Applications
- Vegetation health monitoring
- Agricultural assessment
- Forest cover analysis
- Wetland delineation

## False Color Composite 2: Vegetation Appears Yellow

<img src="images/false_color_comp_yellow.png">

### Band Assignment
- **Red Channel**: Band 6 (SWIR1, 1.57-1.65 μm)
- **Green Channel**: Band 5 (Near Infrared, 0.85-0.88 μm)
- **Blue Channel**: Band 4 (Red, 0.64-0.67 μm)

### Why Vegetation Appears Yellow

This composite (SWIR1-NIR-Red) creates a different vegetation signature where:
- **Healthy vegetation**: Appears bright yellow-green
- **Vegetation types**: Different vegetation types show varying shades of yellow
- **Stressed vegetation**: Appears orange or red
- **Water bodies**: Appear dark blue
- **Urban areas**: Appear pink/magenta

### Applications
- Vegetation stress detection
- Differentiating vegetation types
- Soil moisture assessment
- Burn scar identification

## QGIS Implementation Steps

### Step 1: Load Data
1. Open QGIS
2. Load the stacked file: `Stacked_L8_8bands_elijah.TIF`
3. Verify the file loads with 8 bands

### Step 2: Create Red Vegetation Composite
1. Go to **Raster → Miscellaneous → Build Virtual Raster**
2. Select the stacked file as input
3. Set band assignments:
   - **Red Channel**: Band 5 (NIR)
   - **Green Channel**: Band 4 (Red)
   - **Blue Channel**: Band 3 (Green)
4. Create the virtual raster
5. **Take screenshot** of the result

### Step 3: Create Yellow Vegetation Composite
1. Create another virtual raster
2. Set band assignments:
   - **Red Channel**: Band 6 (SWIR1)
   - **Green Channel**: Band 5 (NIR)
   - **Blue Channel**: Band 4 (Red)
3. Create the virtual raster
4. **Take screenshot** of the result

## Scientific Analysis

### Spectral Properties

| Band | Wavelength (μm) | Primary Use | Vegetation Response |
|------|----------------|-------------|-------------------|
| B3 (Green) | 0.53-0.59 | Photosynthesis | Chlorophyll absorption |
| B4 (Red) | 0.64-0.67 | Photosynthesis | Chlorophyll absorption |
| B5 (NIR) | 0.85-0.88 | Vegetation health | High reflectance |
| B6 (SWIR1) | 1.57-1.65 | Moisture content | Moderate reflectance |

### Color Interpretation

**Red Vegetation Composite (NIR-Red-Green):**
- **Bright Red**: Healthy, dense vegetation
- **Dark Red/Brown**: Stressed or sparse vegetation
- **Dark Blue/Black**: Water bodies
- **White/Gray**: Urban areas, bare soil

**Yellow Vegetation Composite (SWIR1-NIR-Red):**
- **Bright Yellow-Green**: Healthy vegetation
- **Orange/Red**: Stressed vegetation
- **Dark Blue**: Water bodies
- **Pink/Magenta**: Urban areas

## Comparison and Analysis

The two false color composites provide complementary information:

1. **NIR-Red-Green (Red vegetation)**: 
   - Best for overall vegetation health and density
   - Excellent for agricultural monitoring
   - Clear water body delineation

2. **SWIR1-NIR-Red (Yellow vegetation)**: 
   - Better for detecting vegetation stress and moisture content
   - Useful for differentiating vegetation types
   - Effective for burn scar identification

## Conclusion

Both false color composites effectively highlight vegetation but serve different analytical purposes in remote sensing applications. The red vegetation composite is ideal for general vegetation monitoring, while the yellow vegetation composite provides more detailed information about vegetation stress and moisture content. These techniques are fundamental tools in remote sensing for environmental monitoring, agricultural assessment, and land cover analysis.

---
