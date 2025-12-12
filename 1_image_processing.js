//Google Earth Engine Script for Impervious Surface Monitoring
// Weber River Watershed - Sentinel-2 Image Collection
// GEOG 6805 Final Project

// ============================================================================
// 1. STUDY AREA DEFINITION
// ============================================================================
// Load Weber River Watershed shapefile asset
var watershedFC = ee.FeatureCollection('projects/ee-elijahmanwillmeng/assets/WeberWatershed');

// Get geometry from feature collection (handles single or multiple features)
var studyArea = watershedFC.geometry();

// Print watershed info
print('Watershed feature count:', watershedFC.size());
print('Watershed area (sq km):', studyArea.area().divide(1e6));

// ============================================================================
// 2. SENTINEL-2 COLLECTION SETUP
// ============================================================================
// Load Sentinel-2 Harmonized collection (COPERNICUS/S2_HARMONIZED)
var s2Collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  .filterBounds(studyArea)
  .filterDate('2017-01-01', '2024-12-31')  // 8-year period
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));  // Max 10% cloud coverage

print('Total images in collection:', s2Collection.size());

// ============================================================================
// 3. CLOUD MASKING FUNCTION
// ============================================================================
function maskS2Clouds(image) {
  var qa = image.select('QA60');
  
  // Bits 10 and 11 are clouds and cirrus
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  
  // Both flags should be set to zero, indicating clear conditions
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  
  return image.updateMask(mask)
    .divide(10000)  // Scale to reflectance [0, 1]
    .copyProperties(image, ['system:time_start', 'system:time_end']);
}

// ============================================================================
// 4. BAND SELECTION AND PROCESSING
// ============================================================================
// Selected bands for impervious surface detection:
// B2: Blue (10m)
// B3: Green (10m)
// B4: Red (10m)
// B8: Near-Infrared (10m)
// B11: Shortwave Infrared 1 (20m)
// B12: Shortwave Infrared 2 (20m)

function processS2Image(image) {
  // Select relevant bands first
  var bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'QA60']);
  
  // Apply cloud masking
  var qa = bands.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  
  // Apply mask and scale to reflectance [0, 1]
  var masked = bands.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    .updateMask(mask)
    .divide(10000);
  
  // Resample to 30m for consistency (using bilinear)
  var resampled = masked.resample('bilinear').reproject({
    crs: 'EPSG:4326',
    scale: 30
  });
  
  // Clip to study area
  var clipped = resampled.clip(studyArea);
  
  return clipped.copyProperties(image, ['system:time_start', 'system:time_end']);
}

// Apply processing to collection
var processedCollection = s2Collection.map(processS2Image);

print('Processed collection size:', processedCollection.size());

// ============================================================================
// 5. SPECTRAL INDICES CALCULATION
// ============================================================================
function calculateIndices(image) {
  // Extract bands
  var B2 = image.select('B2');  // Blue
  var B3 = image.select('B3');  // Green
  var B4 = image.select('B4');  // Red
  var B8 = image.select('B8');  // NIR
  var B11 = image.select('B11'); // SWIR1
  var B12 = image.select('B12'); // SWIR2
  
  // Normalized Difference Vegetation Index (NDVI)
  var ndvi = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI');
  
  // Normalized Difference Built-up Index (NDBI)
  var ndbi = B12.subtract(B8).divide(B12.add(B8)).rename('NDBI');
  
  // Normalized Difference Water Index (NDWI)
  var ndwi = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI');
  
  // Modified Normalized Difference Water Index (MNDWI)
  var mndwi = B3.subtract(B12).divide(B3.add(B12)).rename('MNDWI');
  
  // Soil Adjusted Vegetation Index (SAVI)
  var L = 0.5;  // Soil adjustment factor
  var savi = B8.subtract(B4).divide(B8.add(B4).add(L))
    .multiply(1 + L).rename('SAVI');
  
  // Index-based Impervious Surface Index (IBI)
  var saviMndwiMean = savi.add(mndwi).divide(2);
  var ibi = ndbi.subtract(saviMndwiMean)
    .divide(ndbi.add(saviMndwiMean)).rename('IBI');
  
  // Add indices as new bands
  var imageWithIndices = image.addBands([ndvi, ndbi, ndwi, mndwi, savi, ibi]);
  
  // Cast all bands to Float32 to ensure consistent data types for export
  var allBands = imageWithIndices.select([
    'B2', 'B3', 'B4', 'B8', 'B11', 'B12', 
    'NDVI', 'NDBI', 'NDWI', 'MNDWI', 'SAVI', 'IBI'
  ]);
  
  return allBands.toFloat();
}

// Calculate indices for all images
var collectionWithIndices = processedCollection.map(calculateIndices);

// ============================================================================
// 6. SELECT BEST IMAGE NEAR JULY 1ST FOR EACH YEAR
// ============================================================================
// Function to find best image near July 1st for a given year
function getBestJulyImage(year) {
  // Define date range: June 15 to July 15 (±15 days from July 1st)
  var startDate = ee.Date.fromYMD(year, 6, 15);
  var endDate = ee.Date.fromYMD(year, 7, 15);
  
  // Filter collection for this year's July window
  var julyImages = s2Collection
    .filterDate(startDate, endDate)
    .sort('CLOUDY_PIXEL_PERCENTAGE');  // Sort by cloud coverage (lowest first)
  
  // Get the image with lowest cloud coverage
  var bestImage = ee.Image(julyImages.first());
  
  // Add year as property for reference
  return bestImage.set('year', year);
}

// Get best July image for each year (2017-2024)
var years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024];
var bestJulyImages = years.map(function(year) {
  return getBestJulyImage(year);
});

// Convert to ImageCollection and process
var bestJulyCollection = ee.ImageCollection.fromImages(bestJulyImages);
var processedJulyImages = bestJulyCollection.map(processS2Image).map(calculateIndices);

// Print info about selected images
print('Best July images selected:', processedJulyImages.size());

// Get dates of selected images
var dates = processedJulyImages.aggregate_array('system:time_start');
print('Selected image dates:', dates);

// ============================================================================
// 6B. COMPOSITE CREATION (Optional - for reference)
// ============================================================================
// Create median composite from all processed images (for comparison)
var medianComposite = collectionWithIndices.median();

// ============================================================================
// 7. VISUALIZATION
// ============================================================================
// Visualization parameters
var rgbVis = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.3,
  gamma: 1.4
};

var falseColorVis = {
  bands: ['B8', 'B4', 'B3'],
  min: 0,
  max: 0.3,
  gamma: 1.4
};

var ndviVis = {
  min: -1,
  max: 1,
  palette: ['blue', 'white', 'green']
};

var ndbiVis = {
  min: -1,
  max: 1,
  palette: ['blue', 'yellow', 'red']
};

var ibiVis = {
  min: -1,
  max: 1,
  palette: ['blue', 'white', 'red']
};

// Center map on study area
Map.centerObject(studyArea, 10);

// Get images as a list and add to map
// Note: We'll iterate client-side to add layers
var imageList = processedJulyImages.toList(processedJulyImages.size());

// Get the size (this will be evaluated client-side)
var collectionSize = processedJulyImages.size().getInfo();

// Add each yearly image layer
// 2017
var img2017 = ee.Image(imageList.get(0));
var date2017 = ee.Date(img2017.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2017, rgbVis, '2017 - RGB', false);
Map.addLayer(img2017, falseColorVis, '2017 - False Color', false);
Map.addLayer(img2017.select('NDVI'), ndviVis, '2017 - NDVI', false);
Map.addLayer(img2017.select('NDBI'), ndbiVis, '2017 - NDBI', false);
Map.addLayer(img2017.select('IBI'), ibiVis, '2017 - IBI', false);

// 2018
var img2018 = ee.Image(imageList.get(1));
var date2018 = ee.Date(img2018.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2018, rgbVis, '2018 - RGB', false);
Map.addLayer(img2018, falseColorVis, '2018 - False Color', false);
Map.addLayer(img2018.select('NDVI'), ndviVis, '2018 - NDVI', false);
Map.addLayer(img2018.select('NDBI'), ndbiVis, '2018 - NDBI', false);
Map.addLayer(img2018.select('IBI'), ibiVis, '2018 - IBI', false);

// 2019
var img2019 = ee.Image(imageList.get(2));
var date2019 = ee.Date(img2019.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2019, rgbVis, '2019 - RGB', false);
Map.addLayer(img2019, falseColorVis, '2019 - False Color', false);
Map.addLayer(img2019.select('NDVI'), ndviVis, '2019 - NDVI', false);
Map.addLayer(img2019.select('NDBI'), ndbiVis, '2019 - NDBI', false);
Map.addLayer(img2019.select('IBI'), ibiVis, '2019 - IBI', false);

// 2020
var img2020 = ee.Image(imageList.get(3));
var date2020 = ee.Date(img2020.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2020, rgbVis, '2020 - RGB', false);
Map.addLayer(img2020, falseColorVis, '2020 - False Color', false);
Map.addLayer(img2020.select('NDVI'), ndviVis, '2020 - NDVI', false);
Map.addLayer(img2020.select('NDBI'), ndbiVis, '2020 - NDBI', false);
Map.addLayer(img2020.select('IBI'), ibiVis, '2020 - IBI', false);

// 2021
var img2021 = ee.Image(imageList.get(4));
var date2021 = ee.Date(img2021.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2021, rgbVis, '2021 - RGB', false);
Map.addLayer(img2021, falseColorVis, '2021 - False Color', false);
Map.addLayer(img2021.select('NDVI'), ndviVis, '2021 - NDVI', false);
Map.addLayer(img2021.select('NDBI'), ndbiVis, '2021 - NDBI', false);
Map.addLayer(img2021.select('IBI'), ibiVis, '2021 - IBI', false);

// 2022
var img2022 = ee.Image(imageList.get(5));
var date2022 = ee.Date(img2022.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2022, rgbVis, '2022 - RGB', false);
Map.addLayer(img2022, falseColorVis, '2022 - False Color', false);
Map.addLayer(img2022.select('NDVI'), ndviVis, '2022 - NDVI', false);
Map.addLayer(img2022.select('NDBI'), ndbiVis, '2022 - NDBI', false);
Map.addLayer(img2022.select('IBI'), ibiVis, '2022 - IBI', false);

// 2023
var img2023 = ee.Image(imageList.get(6));
var date2023 = ee.Date(img2023.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2023, rgbVis, '2023 - RGB', false);
Map.addLayer(img2023, falseColorVis, '2023 - False Color', false);
Map.addLayer(img2023.select('NDVI'), ndviVis, '2023 - NDVI', false);
Map.addLayer(img2023.select('NDBI'), ndbiVis, '2023 - NDBI', false);
Map.addLayer(img2023.select('IBI'), ibiVis, '2023 - IBI', false);

// 2024 (most recent - visible by default)
var img2024 = ee.Image(imageList.get(7));
var date2024 = ee.Date(img2024.get('system:time_start')).format('YYYY-MM-dd');
Map.addLayer(img2024, rgbVis, '2024 - RGB', true);  // Visible by default
Map.addLayer(img2024, falseColorVis, '2024 - False Color', false);
Map.addLayer(img2024.select('NDVI'), ndviVis, '2024 - NDVI', false);
Map.addLayer(img2024.select('NDBI'), ndbiVis, '2024 - NDBI', false);
Map.addLayer(img2024.select('IBI'), ibiVis, '2024 - IBI (Impervious Surface Index)', true);  // Visible by default

// Print dates for reference (will show in console when evaluated)
var allDates = processedJulyImages.aggregate_array('system:time_start');
print('Selected image dates (all years):', allDates);

// Add watershed boundary
Map.addLayer(watershedFC, {color: 'blue', fillColor: '00000000'}, 'Weber River Watershed', false);

// Add median composite for reference (optional)
Map.addLayer(medianComposite, rgbVis, 'Median Composite (All Images)', false);

// ============================================================================
// 8. EXPORT FUNCTIONS - EXPORT ALL YEARLY IMAGES TO GOOGLE DRIVE
// ============================================================================
// Export all yearly July images to Google Drive
// Each image includes: B2, B3, B4, B8, B11, B12, NDVI, NDBI, NDWI, MNDWI, SAVI, IBI

// Function to export a single image
function exportYearlyImage(image, year) {
  return Export.image.toDrive({
    image: image,
    description: 'WeberRiver_S2_' + year + '_July',
    folder: 'WeberRiver_Sentinel2',
    fileNamePrefix: 'WeberRiver_S2_' + year + '_July',
    scale: 30,
    region: studyArea,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
}

// Export each yearly image
// 2017
Export.image.toDrive({
  image: img2017,
  description: 'WeberRiver_S2_2017_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2017_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 2018
Export.image.toDrive({
  image: img2018,
  description: 'WeberRiver_S2_2018_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2018_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 2019
Export.image.toDrive({
  image: img2019,
  description: 'WeberRiver_S2_2019_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2019_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 2020
Export.image.toDrive({
  image: img2020,
  description: 'WeberRiver_S2_2020_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2020_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 2021
Export.image.toDrive({
  image: img2021,
  description: 'WeberRiver_S2_2021_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2021_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 2022
Export.image.toDrive({
  image: img2022,
  description: 'WeberRiver_S2_2022_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2022_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 2023
Export.image.toDrive({
  image: img2023,
  description: 'WeberRiver_S2_2023_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2023_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

// 2024
Export.image.toDrive({
  image: img2024,
  description: 'WeberRiver_S2_2024_July',
  folder: 'WeberRiver_Sentinel2',
  fileNamePrefix: 'WeberRiver_S2_2024_July',
  scale: 30,
  region: studyArea,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

print('✓ Export tasks created for all 8 yearly images (2017-2024)');
print('Check the Tasks tab in GEE to monitor export progress');
print('Files will be saved to: Google Drive > WeberRiver_Sentinel2 folder');

// ============================================================================
// 9. STATISTICS AND SUMMARY
// ============================================================================
// Calculate statistics for each yearly image
function calculateYearlyStats(image) {
  var stats = image.select(['NDVI', 'NDBI', 'IBI']).reduceRegion({
    reducer: ee.Reducer.minMax().combine({
      reducer2: ee.Reducer.mean(),
      sharedInputs: true
    }),
    geometry: studyArea,
    scale: 30,
    maxPixels: 1e9
  });
  return image.set('stats', stats);
}

var imagesWithStats = processedJulyImages.map(calculateYearlyStats);
print('Yearly image statistics calculated');

// Get statistics for most recent image (2024) as example
var latestImage = ee.Image(imageList.get(7));
var latestStats = latestImage.select(['NDVI', 'NDBI', 'IBI']).reduceRegion({
  reducer: ee.Reducer.minMax().combine({
    reducer2: ee.Reducer.mean(),
    sharedInputs: true
  }),
  geometry: studyArea,
  scale: 30,
  maxPixels: 1e9
});
print('2024 Image Statistics:', latestStats);

// ============================================================================
// 10. ADDITIONAL NOTES
// ============================================================================
// The processedJulyImages collection contains one image per year (2017-2024)
// Each image is the clearest available image near July 1st for that year
// All images have been processed with cloud masking, band selection, 
// resampling to 30m, and spectral indices calculation

// ============================================================================
// NOTES:
// ============================================================================
// 1. Run this script in Google Earth Engine Code Editor
// 2. Study area uses shapefile asset: projects/ee-elijahmanwillmeng/assets/WeberWatershed
// 3. All images are clipped to the exact watershed boundary
// 4. Script selects the best clear image near July 1st for each year (2017-2024)
// 5. Each image is processed with cloud masking, band selection, resampling to 30m
// 6. All spectral indices (NDVI, NDBI, NDWI, MNDWI, SAVI, IBI) are calculated
// 7. Images are visualized on the map - toggle layers to compare years
// 8. Modify export paths and asset IDs as needed
// 9. Use the exported images for machine learning model training in Python
// 10. For Python integration, use the earthengine-api package to export data
