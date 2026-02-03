/**
 * @file plot_utils.c
 * @brief Implementation of plotting utilities
 */

#include "plot_utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

plot_config_t plot_config_default(void) {
  plot_config_t config = {.title = "SR-UKF Example",
                          .xlabel = "Time (s)",
                          .ylabel = "Value",
                          .dark_mode = true,
                          .show_grid = true,
                          .show_legend = true,
                          .width = 1200,
                          .height = 800};
  return config;
}

bool plot_has_gnuplot(void) {
  int ret = system("command -v gnuplot > /dev/null 2>&1");
  return (ret == 0);
}

char *plot_format_timestamp(double seconds, char *buffer) {
  // For simplicity, format as elapsed time with high precision
  // Full ISO8601 would require a base timestamp
  snprintf(buffer, 32, "%.6f", seconds);
  return buffer;
}

/**
 * Group series by timestamp array (same pointer = same time axis)
 * This lets us write multiple columns that share timestamps in one CSV
 */
static int write_csv_group(FILE *f, data_series_t *series, size_t n_series,
                           double *timestamps) {
  if (n_series == 0)
    return 0;

  // Find all series that use this timestamp array
  size_t group_size = 0;
  size_t *group_indices = malloc(n_series * sizeof(size_t));

  for (size_t i = 0; i < n_series; i++) {
    if (series[i].timestamps == timestamps) {
      group_indices[group_size++] = i;
    }
  }

  if (group_size == 0) {
    free(group_indices);
    return 0;
  }

  // Write header
  fprintf(f, "timestamp");
  for (size_t i = 0; i < group_size; i++) {
    fprintf(f, ",%s", series[group_indices[i]].name);
  }
  fprintf(f, "\n");

  // Write data
  size_t n_rows = series[group_indices[0]].count;
  char ts_buffer[32];
  for (size_t row = 0; row < n_rows; row++) {
    plot_format_timestamp(timestamps[row], ts_buffer);
    fprintf(f, "%s", ts_buffer);
    for (size_t i = 0; i < group_size; i++) {
      fprintf(f, ",%.10g", series[group_indices[i]].values[row]);
    }
    fprintf(f, "\n");
  }

  free(group_indices);
  return 0;
}

int plot_write_csv(const char *filename, data_series_t *series,
                   size_t n_series) {
  if (n_series == 0)
    return 0;

  // Track which timestamp arrays we've already written
  double **written_timestamps = malloc(n_series * sizeof(double *));
  size_t n_written = 0;

  for (size_t i = 0; i < n_series; i++) {
    double *ts = series[i].timestamps;

    // Check if we've already written this timestamp array
    bool already_written = false;
    for (size_t j = 0; j < n_written; j++) {
      if (written_timestamps[j] == ts) {
        already_written = true;
        break;
      }
    }

    if (!already_written) {
      // Create filename for this time axis
      char csv_filename[256];
      if (n_written == 0) {
        snprintf(csv_filename, sizeof(csv_filename), "%s.csv", filename);
      } else {
        snprintf(csv_filename, sizeof(csv_filename), "%s_%zu.csv", filename,
                 n_written);
      }

      FILE *f = fopen(csv_filename, "w");
      if (!f) {
        free(written_timestamps);
        return -1;
      }

      write_csv_group(f, series, n_series, ts);
      fclose(f);

      written_timestamps[n_written++] = ts;
    }
  }

  free(written_timestamps);
  return 0;
}

int plot_write_json(const char *filename, data_series_t *series,
                    size_t n_series) {
  FILE *f = fopen(filename, "w");
  if (!f)
    return -1;

  fprintf(f, "{\n");
  fprintf(f, "  \"series\": [\n");

  for (size_t i = 0; i < n_series; i++) {
    fprintf(f, "    {\n");
    fprintf(f, "      \"name\": \"%s\",\n", series[i].name);
    fprintf(f, "      \"data\": [\n");

    for (size_t j = 0; j < series[i].count; j++) {
      char ts_buffer[32];
      plot_format_timestamp(series[i].timestamps[j], ts_buffer);
      fprintf(f, "        {\"t\": %s, \"y\": %.10g}%s\n", ts_buffer,
              series[i].values[j], (j < series[i].count - 1) ? "," : "");
    }

    fprintf(f, "      ]\n");
    fprintf(f, "    }%s\n", (i < n_series - 1) ? "," : "");
  }

  fprintf(f, "  ]\n");
  fprintf(f, "}\n");

  fclose(f);
  return 0;
}

int plot_generate_svg(const char *filename, plot_config_t *config,
                      data_series_t *series, size_t n_series,
                      bool open_viewer) {
  if (!plot_has_gnuplot()) {
    fprintf(stderr, "Warning: gnuplot not found. Cannot generate SVG.\n");
    fprintf(stderr,
            "Install gnuplot to enable SVG output: apt install gnuplot\n");
    return 1;
  }

  // Write data to temporary file
  char data_file[256];
  snprintf(data_file, sizeof(data_file), "/tmp/srukf_plot_%d.dat", getpid());

  FILE *f = fopen(data_file, "w");
  if (!f)
    return -1;

  // Write all series (assumes they share timestamps for simplicity)
  // For multi-timebase, would need multiple plot commands
  size_t n_rows = series[0].count;
  for (size_t i = 0; i < n_rows; i++) {
    fprintf(f, "%.10g", series[0].timestamps[i]);
    for (size_t j = 0; j < n_series; j++) {
      fprintf(f, " %.10g", series[j].values[i]);
    }
    fprintf(f, "\n");
  }
  fclose(f);

  // Generate gnuplot script
  char script_file[256];
  snprintf(script_file, sizeof(script_file), "/tmp/srukf_plot_%d.plt",
           getpid());

  f = fopen(script_file, "w");
  if (!f) {
    remove(data_file);
    return -1;
  }

  // Dark mode color scheme
  if (config->dark_mode) {
    fprintf(f, "set terminal svg size %d,%d enhanced background '#1a1a1a'\n",
            config->width, config->height);
    fprintf(f, "set border linecolor rgb '#e0e0e0'\n");
    fprintf(f, "set key textcolor rgb '#e0e0e0'\n");
    fprintf(f, "set xlabel textcolor rgb '#e0e0e0'\n");
    fprintf(f, "set ylabel textcolor rgb '#e0e0e0'\n");
    fprintf(f, "set title textcolor rgb '#e0e0e0'\n");
    fprintf(f, "set grid linecolor rgb '#404040'\n");
  } else {
    fprintf(f, "set terminal svg size %d,%d enhanced\n", config->width,
            config->height);
  }

  fprintf(f, "set output '%s'\n", filename);
  fprintf(f, "set title '%s'\n", config->title);
  fprintf(f, "set xlabel '%s'\n", config->xlabel);
  fprintf(f, "set ylabel '%s'\n", config->ylabel);

  if (config->show_grid) {
    fprintf(f, "set grid\n");
  }

  // Plot commands
  fprintf(f, "plot ");
  for (size_t i = 0; i < n_series; i++) {
    const char *style = series[i].style ? series[i].style : "lines";
    const char *color = series[i].color ? series[i].color : "";

    fprintf(f, "'%s' using 1:%zu with %s %s title '%s'", data_file, i + 2,
            style, color, series[i].name);

    if (i < n_series - 1) {
      fprintf(f, ", \\\n     ");
    }
  }
  fprintf(f, "\n");

  fclose(f);

  // Execute gnuplot
  char cmd[512];
  snprintf(cmd, sizeof(cmd), "gnuplot '%s'", script_file);
  int ret = system(cmd);

  // Clean up temp files
  remove(data_file);
  remove(script_file);

  if (ret != 0) {
    fprintf(stderr, "Error: gnuplot execution failed\n");
    return -1;
  }

  // Optionally open the result
  if (open_viewer) {
    snprintf(cmd, sizeof(cmd), "xdg-open '%s' > /dev/null 2>&1 &", filename);
    system(cmd);
  }

  return 0;
}
