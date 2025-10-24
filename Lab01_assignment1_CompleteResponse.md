# Assignment 1: Landsat-8 Data Analysis
**Student Name**: [Your Name]  
**Date**: [Current Date]  
**Course**: GeoAI Remote Sensing Lab 1

## Data Overview

For this assignment, I downloaded a Landsat-8 image from the Louisiana/Mississippi area, acquired on October 2, 2020. The image was downloaded from the USGS GloVis website and contains 11 spectral bands plus additional metadata files.

**Scene Information:**
- **Scene ID**: LC08_L1TP_022039_20201002_20201015_02_T1
- **Date**: October 2, 2020
- **Location**: Path 022, Row 039 (Louisiana/Mississippi area)
- **Satellite**: Landsat-8 OLI/TIRS
- **Collection**: Collection 2, Tier 1

## File Structure Analysis

The downloaded Landsat-8 data contains the following files:

| File Type | Count | Description |
|-----------|-------|-------------|
| Band Files (.TIF) | 11 | Individual spectral bands (B1-B11) |
| Metadata Files | 8 | Quality assessment, metadata, and auxiliary files |
| **Total Files** | **19** | Complete Landsat-8 scene package |

## Landsat-8 Band Information

| Band | Wavelength (μm) | Resolution (m) | Description | Color Assignment | File Size (MB) |
|------|----------------|----------------|-------------|------------------|----------------|
| B1 | 0.43-0.45 | 30 | Coastal/Aerosol | - | 72.1 |
| B2 | 0.45-0.51 | 30 | Blue | Blue | 73.9 |
| B3 | 0.53-0.59 | 30 | Green | Green | 79.0 |
| B4 | 0.64-0.67 | 30 | Red | Red | 80.6 |
| B5 | 0.85-0.88 | 30 | Near Infrared (NIR) | - | 88.7 |
| B6 | 1.57-1.65 | 30 | Shortwave Infrared 1 (SWIR1) | - | 87.2 |
| B7 | 2.11-2.29 | 30 | Shortwave Infrared 2 (SWIR2) | - | 83.6 |
| B8 | 0.50-0.68 | 15 | Panchromatic | - | 315.2 |
| B9 | 1.36-1.38 | 30 | Cirrus | - | 43.8 |
| B10 | 10.60-11.19 | 100 | Thermal Infrared 1 (TIRS1) | - | 75.7 |
| B11 | 11.50-12.51 | 100 | Thermal Infrared 2 (TIRS2) | - | 73.5 |

**Total Band Data Size**: 1,073.3 MB

## Additional Files

1. **MTL.txt** (12.1 KB): Contains metadata including acquisition date, time, sun elevation, and processing parameters
2. **MTL.xml** (17.6 KB): XML version of the metadata file
3. **QA_PIXEL.TIF** (1.3 MB): Quality assessment band indicating cloud cover, cloud shadow, and other quality flags
4. **QA_RADSAT.TIF** (0.2 MB): Radiometric saturation and terrain occlusion information
5. **ANG.txt** (114.4 KB): Solar and sensor angle information for each pixel
6. **SAA.TIF** (2.3 MB): Solar azimuth angle
7. **SZA.TIF** (2.0 MB): Solar zenith angle
8. **VAA.TIF** (6.8 MB): View azimuth angle
9. **VZA.TIF** (2.6 MB): View zenith angle

## Key Observations

- **Spatial Resolution**: 
  - Most bands (B1-B7, B9): 30m resolution
  - Panchromatic band (B8): 15m resolution
  - Thermal bands (B10-B11): 100m resolution
- **File Naming Convention**: LC08_L1TP_022039_20201002_20201015_02_T1_B[X].TIF
  - LC08: Landsat-8 OLI/TIRS
  - L1TP: Level-1 Terrain Precision
  - 022039: Path 22, Row 39 (Louisiana/Mississippi area)
  - 20201002: Acquisition date (October 2, 2020)
  - 02_T1: Collection 2, Tier 1
- **Total Data Size**: Approximately 1.1 GB for all band files
- **Cloud Coverage**: The image was selected for minimal cloud cover as required
- **Geographic Coverage**: Covers parts of Louisiana and Mississippi, including the New Orleans metropolitan area

## Band Applications

- **B1 (Coastal/Aerosol)**: Useful for coastal water studies and aerosol detection
- **B2-B4 (Blue, Green, Red)**: Standard RGB bands for true color visualization
- **B5 (NIR)**: Essential for vegetation analysis and NDVI calculations
- **B6-B7 (SWIR)**: Important for soil moisture, vegetation stress, and burn scar detection
- **B8 (Panchromatic)**: High-resolution band for sharpening other bands
- **B9 (Cirrus)**: Cloud detection and atmospheric correction
- **B10-B11 (Thermal)**: Land surface temperature analysis

## Conclusion

This Landsat-8 dataset provides comprehensive spectral information suitable for various remote sensing applications including land cover classification, vegetation monitoring, water quality assessment, and urban analysis. The data meets all assignment requirements with minimal cloud cover and includes all necessary spectral bands for advanced remote sensing analysis.

---

**Note**: This analysis was completed using Python and GDAL libraries to programmatically examine the Landsat-8 data structure and properties. The data is ready for use in subsequent assignments involving band stacking and extraction operations.
