/**
 * @file plot_utils.h
 * @brief Utilities for generating plots from SR-UKF examples
 *
 * This module provides a simple interface for creating visualizations
 * from example data. It supports multiple output formats (SVG, CSV, JSON)
 * and uses gnuplot for SVG generation when available.
 *
 * The design philosophy:
 * - CSV for maximum portability and analysis flexibility
 * - SVG for immediate visual feedback and documentation
 * - JSON for structured data exchange with web demos
 * - Gnuplot for professional-quality plots without heavy dependencies
 */

#ifndef PLOT_UTILS_H
#define PLOT_UTILS_H

#include <stdbool.h>
#include <stdio.h>

/**
 * Output format options
 */
typedef enum {
  OUTPUT_CSV,  /**< Comma-separated values - one file per time axis */
  OUTPUT_JSON, /**< JSON format - single file with all data */
  OUTPUT_SVG,  /**< SVG via gnuplot - requires gnuplot installed */
  OUTPUT_ALL   /**< Generate all formats */
} output_format_t;

/**
 * Plot style configuration
 */
typedef struct {
  const char *title;
  const char *xlabel;
  const char *ylabel;
  bool dark_mode; /**< Use dark mode color scheme */
  bool show_grid;
  bool show_legend;
  int width;  /**< SVG width in pixels */
  int height; /**< SVG height in pixels */
} plot_config_t;

/**
 * Data series for plotting
 */
typedef struct {
  const char *name;
  double *timestamps; /**< ISO8601 timestamps converted to doubles (seconds) */
  double *values;
  size_t count;
  const char *style; /**< gnuplot style: "lines", "points", "linespoints" */
  const char *color; /**< Color specification (for dark mode support) */
} data_series_t;

/**
 * Initialize default plot configuration
 *
 * @return Default configuration with dark mode enabled
 */
plot_config_t plot_config_default(void);

/**
 * Write data series to CSV file
 *
 * Multiple series with the same timestamps go in one file.
 * Series with different timestamps get separate files.
 *
 * @param filename Base filename (without extension)
 * @param series Array of data series
 * @param n_series Number of series
 * @return 0 on success, -1 on error
 */
int plot_write_csv(const char *filename, data_series_t *series,
                   size_t n_series);

/**
 * Write data series to JSON file
 *
 * Simple JSON structure suitable for web consumption.
 *
 * @param filename Output filename (with .json extension)
 * @param series Array of data series
 * @param n_series Number of series
 * @return 0 on success, -1 on error
 */
int plot_write_json(const char *filename, data_series_t *series,
                    size_t n_series);

/**
 * Generate SVG plot using gnuplot
 *
 * Creates a gnuplot script, executes it, and optionally opens the result.
 * Falls back gracefully if gnuplot is not available.
 *
 * @param filename Output filename (with .svg extension)
 * @param config Plot configuration
 * @param series Array of data series
 * @param n_series Number of series
 * @param open_viewer If true, attempt to open SVG with xdg-open
 * @return 0 on success, -1 on error, 1 if gnuplot unavailable
 */
int plot_generate_svg(const char *filename, plot_config_t *config,
                      data_series_t *series, size_t n_series, bool open_viewer);

/**
 * Check if gnuplot is available
 *
 * @return true if gnuplot command can be executed
 */
bool plot_has_gnuplot(void);

/**
 * Format timestamp as ISO8601 string
 *
 * @param seconds Timestamp in seconds since start
 * @param buffer Output buffer (must be at least 32 bytes)
 * @return Pointer to buffer
 */
char *plot_format_timestamp(double seconds, char *buffer);

#endif /* PLOT_UTILS_H */
