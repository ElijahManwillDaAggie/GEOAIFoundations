# Statistical Analysis and Visualization of Impervious Surface Changes
# Weber River Watershed - Temporal Analysis (2017-2024)
#
# This script:
# 1. Loads impervious surface prediction maps
# 2. Calculates impervious surface area for each year
# 3. Performs statistical inference (trend analysis, change detection)
# 4. Creates visualizations of temporal changes

# Load required libraries
library(raster)
library(rgdal)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(trend)
library(Kendall)

# Set working directory (adjust as needed)
# setwd("/path/to/Final_Project")

# Configuration
PREDICTIONS_FOLDER <- "predictions"
OUTPUT_FOLDER <- "r_analysis"
MODEL_TYPE <- "rf"  # "rf" for Random Forest or "dt" for Decision Tree

# Create output folder
dir.create(OUTPUT_FOLDER, showWarnings = FALSE)

# ============================================================================
# 1. LOAD PREDICTION MAPS
# ============================================================================

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Impervious Surface Temporal Analysis\n")
cat("=" %+% strrep("=", 59), "\n\n")

cat("1. Loading prediction maps...\n")

# Get list of prediction files
prediction_files <- list.files(
  PREDICTIONS_FOLDER, 
  pattern = paste0("^", MODEL_TYPE, ".*\\.tif$"),
  full.names = TRUE
)

if (length(prediction_files) == 0) {
  stop("No prediction files found. Please run apply_models.py first.")
}

# Sort files by year
years <- as.numeric(gsub(".*(\\d{4}).*", "\\1", basename(prediction_files)))
prediction_files <- prediction_files[order(years)]
years <- sort(years)

cat(sprintf("   Found %d prediction maps for years: %s\n", 
            length(prediction_files), paste(years, collapse = ", ")))

# Load all prediction rasters
prediction_rasters <- lapply(prediction_files, function(f) {
  cat(sprintf("   Loading: %s\n", basename(f)))
  raster(f)
})

names(prediction_rasters) <- paste0("year_", years)

# ============================================================================
# 2. CALCULATE IMPERVIOUS SURFACE AREA FOR EACH YEAR
# ============================================================================

cat("\n2. Calculating impervious surface area for each year...\n")

# Function to calculate impervious surface area
calculate_impervious_area <- function(r) {
  # Count pixels classified as impervious (value = 1)
  impervious_pixels <- sum(values(r) == 1, na.rm = TRUE)
  total_pixels <- sum(!is.na(values(r)))
  
  # Get pixel area (assuming square pixels)
  pixel_area <- res(r)[1] * res(r)[2]  # in map units (likely degrees)
  
  # Convert to square meters (if needed, adjust based on your CRS)
  # For UTM: pixel_area is already in m²
  # For geographic: need to convert degrees to meters
  # This is a simplification - adjust based on your actual CRS
  
  # Calculate area
  impervious_area_pixels <- impervious_pixels
  impervious_area_km2 <- (impervious_area_pixels * pixel_area) / 1e6
  
  # Calculate percentage
  impervious_percentage <- (impervious_pixels / total_pixels) * 100
  
  return(list(
    impervious_pixels = impervious_pixels,
    total_pixels = total_pixels,
    impervious_area_km2 = impervious_area_km2,
    impervious_percentage = impervious_percentage
  ))
}

# Calculate statistics for each year
yearly_stats <- data.frame(
  year = years,
  impervious_pixels = numeric(length(years)),
  total_pixels = numeric(length(years)),
  impervious_area_km2 = numeric(length(years)),
  impervious_percentage = numeric(length(years))
)

for (i in 1:length(prediction_rasters)) {
  stats <- calculate_impervious_area(prediction_rasters[[i]])
  yearly_stats$impervious_pixels[i] <- stats$impervious_pixels
  yearly_stats$total_pixels[i] <- stats$total_pixels
  yearly_stats$impervious_area_km2[i] <- stats$impervious_area_km2
  yearly_stats$impervious_percentage[i] <- stats$impervious_percentage
  
  cat(sprintf("   %d: %.2f%% impervious (%.2f km²)\n", 
              years[i], stats$impervious_percentage, stats$impervious_area_km2))
}

# Save yearly statistics
write.csv(yearly_stats, 
          file.path(OUTPUT_FOLDER, "yearly_impervious_statistics.csv"),
          row.names = FALSE)
cat(sprintf("\n   Saved: yearly_impervious_statistics.csv\n"))

# ============================================================================
# 3. STATISTICAL INFERENCE - TREND ANALYSIS
# ============================================================================

cat("\n3. Performing statistical inference...\n")

# Mann-Kendall trend test for impervious percentage
cat("   Mann-Kendall trend test for impervious surface percentage...\n")
mk_test_percentage <- MannKendall(yearly_stats$impervious_percentage)
print(mk_test_percentage)

# Mann-Kendall trend test for impervious area
cat("\n   Mann-Kendall trend test for impervious surface area...\n")
mk_test_area <- MannKendall(yearly_stats$impervious_area_km2)
print(mk_test_area)

# Linear regression for trend
cat("\n   Linear regression analysis...\n")
lm_percentage <- lm(impervious_percentage ~ year, data = yearly_stats)
lm_area <- lm(impervious_area_km2 ~ year, data = yearly_stats)

cat("   Impervious Percentage Trend:\n")
print(summary(lm_percentage))

cat("\n   Impervious Area Trend:\n")
print(summary(lm_area))

# Calculate annual change rate
annual_change_percentage <- coef(lm_percentage)[2]  # slope
annual_change_area <- coef(lm_area)[2]  # slope

cat(sprintf("\n   Annual change rate: %.4f%% per year\n", annual_change_percentage))
cat(sprintf("   Annual change rate: %.4f km² per year\n", annual_change_area))

# ============================================================================
# 4. CHANGE DETECTION BETWEEN YEARS
# ============================================================================

cat("\n4. Calculating year-to-year changes...\n")

# Calculate changes between consecutive years
yearly_changes <- data.frame(
  year_start = years[-length(years)],
  year_end = years[-1],
  change_percentage = diff(yearly_stats$impervious_percentage),
  change_area_km2 = diff(yearly_stats$impervious_area_km2),
  percent_change = diff(yearly_stats$impervious_percentage) / 
                   yearly_stats$impervious_percentage[-length(years)] * 100
)

cat("   Year-to-year changes:\n")
print(yearly_changes)

write.csv(yearly_changes,
          file.path(OUTPUT_FOLDER, "yearly_changes.csv"),
          row.names = FALSE)
cat(sprintf("   Saved: yearly_changes.csv\n"))

# ============================================================================
# 5. CREATE VISUALIZATIONS
# ============================================================================

cat("\n5. Creating visualizations...\n")

# 5.1 Time series plot of impervious surface percentage
p1 <- ggplot(yearly_stats, aes(x = year, y = impervious_percentage)) +
  geom_line(color = "blue", size = 1.2) +
  geom_point(color = "red", size = 3) +
  geom_smooth(method = "lm", se = TRUE, color = "green", linetype = "dashed") +
  labs(title = "Impervious Surface Percentage Over Time (2017-2024)",
       subtitle = paste0("Trend: ", sprintf("%.4f%% per year", annual_change_percentage)),
       x = "Year",
       y = "Impervious Surface (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12))

ggsave(file.path(OUTPUT_FOLDER, "impervious_percentage_timeseries.png"),
       p1, width = 10, height = 6, dpi = 300)
cat("   Saved: impervious_percentage_timeseries.png\n")

# 5.2 Time series plot of impervious surface area
p2 <- ggplot(yearly_stats, aes(x = year, y = impervious_area_km2)) +
  geom_line(color = "blue", size = 1.2) +
  geom_point(color = "red", size = 3) +
  geom_smooth(method = "lm", se = TRUE, color = "green", linetype = "dashed") +
  labs(title = "Impervious Surface Area Over Time (2017-2024)",
       subtitle = paste0("Trend: ", sprintf("%.4f km² per year", annual_change_area)),
       x = "Year",
       y = "Impervious Surface Area (km²)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12))

ggsave(file.path(OUTPUT_FOLDER, "impervious_area_timeseries.png"),
       p2, width = 10, height = 6, dpi = 300)
cat("   Saved: impervious_area_timeseries.png\n")

# 5.3 Bar plot of year-to-year changes
p3 <- ggplot(yearly_changes, aes(x = factor(year_start), y = change_percentage)) +
  geom_bar(stat = "identity", fill = ifelse(yearly_changes$change_percentage > 0, "red", "green")) +
  labs(title = "Year-to-Year Change in Impervious Surface Percentage",
       x = "Year",
       y = "Change in Percentage (%)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(OUTPUT_FOLDER, "yearly_changes_barplot.png"),
       p3, width = 10, height = 6, dpi = 300)
cat("   Saved: yearly_changes_barplot.png\n")

# 5.4 Combined plot
p4 <- ggplot(yearly_stats, aes(x = year)) +
  geom_line(aes(y = impervious_percentage, color = "Percentage"), size = 1.2) +
  geom_point(aes(y = impervious_percentage, color = "Percentage"), size = 3) +
  geom_line(aes(y = impervious_area_km2 * 10, color = "Area (×10)"), size = 1.2) +
  geom_point(aes(y = impervious_area_km2 * 10, color = "Area (×10)"), size = 3) +
  scale_y_continuous(
    name = "Impervious Surface (%)",
    sec.axis = sec_axis(~ . / 10, name = "Impervious Surface Area (km²)")
  ) +
  scale_color_manual(values = c("Percentage" = "blue", "Area (×10)" = "red")) +
  labs(title = "Impervious Surface: Percentage and Area Over Time",
       x = "Year",
       color = "Metric") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        legend.position = "bottom")

ggsave(file.path(OUTPUT_FOLDER, "combined_timeseries.png"),
       p4, width = 12, height = 6, dpi = 300)
cat("   Saved: combined_timeseries.png\n")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================

cat("\n6. Summary Statistics\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\nOverall Statistics (2017-2024):\n")
cat(sprintf("  Mean impervious percentage: %.2f%%\n", 
            mean(yearly_stats$impervious_percentage)))
cat(sprintf("  Standard deviation: %.2f%%\n", 
            sd(yearly_stats$impervious_percentage)))
cat(sprintf("  Minimum: %.2f%% (year %d)\n", 
            min(yearly_stats$impervious_percentage),
            yearly_stats$year[which.min(yearly_stats$impervious_percentage)]))
cat(sprintf("  Maximum: %.2f%% (year %d)\n", 
            max(yearly_stats$impervious_percentage),
            yearly_stats$year[which.max(yearly_stats$impervious_percentage)]))
cat(sprintf("  Total change (2017-2024): %.2f%%\n", 
            yearly_stats$impervious_percentage[length(years)] - 
            yearly_stats$impervious_percentage[1]))
cat(sprintf("  Total area change: %.2f km²\n", 
            yearly_stats$impervious_area_km2[length(years)] - 
            yearly_stats$impervious_area_km2[1]))

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("Analysis complete! Results saved to:", OUTPUT_FOLDER, "\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

