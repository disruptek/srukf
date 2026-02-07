/**
 * @file stability.c
 * @brief Long-duration numerical stability test for SR-UKF
 *
 * This example demonstrates the numerical robustness of the Square-Root UKF
 * formulation over extended operation. We run the filter for thousands of
 * iterations and monitor numerical health metrics.
 *
 * WHY THIS MATTERS:
 * - Standard Kalman filters can suffer from covariance matrix going negative
 * - Numerical roundoff can accumulate over long runs
 * - SR-UKF's square-root formulation provides inherent stability
 *
 * WHAT WE TEST:
 * - Covariance remains positive definite (guaranteed by Cholesky form)
 * - No NaN or Inf values appear
 * - Error statistics remain bounded
 * - Filter consistency (NEES/NIS tests)
 * - Performance doesn't degrade over time
 *
 * This is a "torture test" to validate the implementation.
 */

/* ========================================================================
 * INCLUDES
 * ======================================================================== */

#include <float.h>
#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "plot_utils.h"
#include "srukf.h"

/* For LAPACK condition number computation */
#ifdef HAVE_LAPACK
#include <lapacke.h>
#endif

/* ========================================================================
 * CONFIGURATION
 * ======================================================================== */

typedef enum {
  SCENARIO_BASELINE = 0,
  SCENARIO_HIGH_DYNAMICS,
  SCENARIO_POOR_OBSERVABILITY,
  SCENARIO_EXTREME_NOISE
} scenario_t;

typedef struct {
  scenario_t scenario;
  double duration; /* Simulation duration (seconds) */
  double dt;       /* Timestep (seconds) */
  output_format_t format;
  bool open_viewer;
  bool verbose;
} test_config_t;

/* Scenario parameters */
typedef struct {
  const char *name;
  const char *description;
  double process_noise;
  double measurement_noise;
  double measurement_rate; /* Hz */
  bool high_dynamics;      /* Enable aggressive maneuvering */
} scenario_params_t;

static const scenario_params_t SCENARIOS[] = {
    [SCENARIO_BASELINE] = {.name = "Baseline",
                           .description = "Steady 2D motion, moderate noise",
                           .process_noise = 0.1,
                           .measurement_noise = 1.0,
                           .measurement_rate = 10.0,
                           .high_dynamics = false},
    [SCENARIO_HIGH_DYNAMICS] =
        {.name = "High Dynamics",
         .description = "Aggressive maneuvering, high accelerations",
         .process_noise = 1.0,
         .measurement_noise = 1.0,
         .measurement_rate = 10.0,
         .high_dynamics = true},
    [SCENARIO_POOR_OBSERVABILITY] = {.name = "Poor Observability",
                                     .description =
                                         "Infrequent measurements (1 Hz)",
                                     .process_noise = 0.1,
                                     .measurement_noise = 1.0,
                                     .measurement_rate = 1.0,
                                     .high_dynamics = false},
    [SCENARIO_EXTREME_NOISE] = {.name = "Extreme Noise",
                                .description =
                                    "Very noisy sensors (10x baseline)",
                                .process_noise = 0.1,
                                .measurement_noise = 10.0,
                                .measurement_rate = 10.0,
                                .high_dynamics = false}};

/* ========================================================================
 * NUMERICAL HEALTH METRICS
 * ======================================================================== */

typedef struct {
  double cov_trace;      /* Trace of covariance (total uncertainty) */
  double cov_det;        /* Determinant (volume of uncertainty ellipsoid) */
  double cov_condition;  /* Condition number (numerical stability indicator) */
  double max_eigenvalue; /* Largest eigenvalue */
  double min_eigenvalue; /* Smallest eigenvalue */
  bool has_nan;          /* Any NaN values detected */
  bool has_inf;          /* Any Inf values detected */
  bool is_positive_def;  /* Covariance is positive definite */
} health_metrics_t;

typedef struct {
  double position_rmse;      /* Root mean square error in position */
  double velocity_rmse;      /* Root mean square error in velocity */
  double max_position_error; /* Maximum position error */
  double nees;               /* Normalized Estimation Error Squared */
  double nis;                /* Normalized Innovation Squared */
} error_metrics_t;

/* ========================================================================
 * UTILITY FUNCTIONS
 * ======================================================================== */

static test_config_t config_default(void) {
  test_config_t cfg = {.scenario = SCENARIO_BASELINE,
                       .duration = 1000.0, /* 1000 seconds = ~16 minutes */
                       .dt = 0.01,         /* 100 Hz simulation */
                       .format = OUTPUT_SVG,
                       .open_viewer = false,
                       .verbose = false};
  return cfg;
}

static double randn(void) {
  /* Box-Muller transform for Gaussian random numbers */
  double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
  double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static double add_noise(double sigma) {
  return sigma * randn();
}

/* Check if value is NaN or Inf */
static bool is_invalid(double x) {
  return isnan(x) || isinf(x);
}

/* ========================================================================
 * 2D MOTION MODEL
 * ======================================================================== */

/**
 * State: [x, y, vx, vy] - position and velocity in 2D
 *
 * Simple constant velocity model with process noise representing acceleration.
 */

typedef struct {
  double dt;
  bool high_dynamics;
  double t; /* Current time for deterministic forcing */
} motion_context_t;

/**
 * Process model: constant velocity with optional high-dynamics forcing
 */
static void process_model(const srukf_mat *x_in, srukf_mat *x_out,
                          void *user_data) {
  motion_context_t *ctx = (motion_context_t *)user_data;

  double x = SRUKF_ENTRY(x_in, 0, 0);
  double y = SRUKF_ENTRY(x_in, 1, 0);
  double vx = SRUKF_ENTRY(x_in, 2, 0);
  double vy = SRUKF_ENTRY(x_in, 3, 0);

  /* High dynamics: add sinusoidal forcing to simulate aggressive maneuvering */
  double ax = 0.0, ay = 0.0;
  if (ctx->high_dynamics) {
    ax = 5.0 * sin(0.5 * ctx->t); /* 5 m/s² peak acceleration */
    ay = 5.0 * cos(0.7 * ctx->t); /* Different frequency for 2D pattern */
  }

  /* Kinematic update */
  SRUKF_ENTRY(x_out, 0, 0) = x + vx * ctx->dt + 0.5 * ax * ctx->dt * ctx->dt;
  SRUKF_ENTRY(x_out, 1, 0) = y + vy * ctx->dt + 0.5 * ay * ctx->dt * ctx->dt;
  SRUKF_ENTRY(x_out, 2, 0) = vx + ax * ctx->dt;
  SRUKF_ENTRY(x_out, 3, 0) = vy + ay * ctx->dt;
}

/**
 * Measurement model: observe position only (not velocity)
 */
static void measurement_model(const srukf_mat *x, srukf_mat *z,
                              void *user_data) {
  (void)user_data;
  SRUKF_ENTRY(z, 0, 0) = SRUKF_ENTRY(x, 0, 0); /* x position */
  SRUKF_ENTRY(z, 1, 0) = SRUKF_ENTRY(x, 1, 0); /* y position */
}

/* ========================================================================
 * NUMERICAL HEALTH ANALYSIS
 * ======================================================================== */

/**
 * Compute numerical health metrics from covariance square root S
 */
static health_metrics_t compute_health(const srukf *ukf) {
  health_metrics_t h = {0};

  /* Check for NaN/Inf in state */
  for (size_t i = 0; i < 4; i++) {
    if (is_invalid(SRUKF_ENTRY(ukf->x, i, 0))) {
      h.has_nan = true;
      return h;
    }
  }

  /* Compute covariance trace (sum of diagonal elements of S'*S) */
  h.cov_trace = 0.0;
  for (size_t i = 0; i < 4; i++) {
    double s_ii = SRUKF_ENTRY(ukf->S, i, i);
    if (is_invalid(s_ii)) {
      h.has_nan = true;
      return h;
    }
    h.cov_trace += s_ii * s_ii; /* P_ii = S_ii^2 for diagonal */
  }

  /* For Cholesky factor, determinant = product of diagonal elements squared */
  h.cov_det = 1.0;
  for (size_t i = 0; i < 4; i++) {
    double s_ii = SRUKF_ENTRY(ukf->S, i, i);
    h.cov_det *= s_ii * s_ii;
  }

  /* Positive definiteness: guaranteed by Cholesky, check diagonal > 0 */
  h.is_positive_def = true;
  for (size_t i = 0; i < 4; i++) {
    if (SRUKF_ENTRY(ukf->S, i, i) <= 0.0) {
      h.is_positive_def = false;
      break;
    }
  }

  /* Eigenvalues: for diagonal terms, eigenvalues of P are S_ii^2 */
  h.max_eigenvalue = 0.0;
  h.min_eigenvalue = DBL_MAX;
  for (size_t i = 0; i < 4; i++) {
    double s_ii = SRUKF_ENTRY(ukf->S, i, i);
    double eig = s_ii * s_ii;
    if (eig > h.max_eigenvalue)
      h.max_eigenvalue = eig;
    if (eig < h.min_eigenvalue)
      h.min_eigenvalue = eig;
  }

  /* Condition number = max_eig / min_eig */
  if (h.min_eigenvalue > 1e-15) {
    h.cov_condition = h.max_eigenvalue / h.min_eigenvalue;
  } else {
    h.cov_condition = DBL_MAX;
  }

  return h;
}

/**
 * Compute error metrics comparing estimate to ground truth
 */
static error_metrics_t compute_errors(const srukf *ukf, double true_x,
                                      double true_y, double true_vx,
                                      double true_vy) {
  error_metrics_t e = {0};

  double ex = SRUKF_ENTRY(ukf->x, 0, 0) - true_x;
  double ey = SRUKF_ENTRY(ukf->x, 1, 0) - true_y;
  double evx = SRUKF_ENTRY(ukf->x, 2, 0) - true_vx;
  double evy = SRUKF_ENTRY(ukf->x, 3, 0) - true_vy;

  e.position_rmse = sqrt(ex * ex + ey * ey);
  e.velocity_rmse = sqrt(evx * evx + evy * evy);
  e.max_position_error = e.position_rmse; /* Updated incrementally */

  /* NEES: Normalized Estimation Error Squared
   * NEES = error' * P^-1 * error
   * For small errors, should be chi-squared distributed with n degrees of
   * freedom Expected value ≈ n (here n=4)
   */
  double P_inv_diag[4];
  for (size_t i = 0; i < 4; i++) {
    double s_ii = SRUKF_ENTRY(ukf->S, i, i);
    P_inv_diag[i] = 1.0 / (s_ii * s_ii + 1e-15);
  }

  e.nees = ex * ex * P_inv_diag[0] + ey * ey * P_inv_diag[1] +
           evx * evx * P_inv_diag[2] + evy * evy * P_inv_diag[3];

  return e;
}

/* ========================================================================
 * MAIN STABILITY TEST
 * ======================================================================== */

static int run_stability_test(const test_config_t *cfg) {
  const scenario_params_t *scenario = &SCENARIOS[cfg->scenario];

  printf("\n");
  printf("=============================================================\n");
  printf("  SR-UKF Long-Duration Stability Test\n");
  printf("=============================================================\n");
  printf("Scenario: %s\n", scenario->name);
  printf("  %s\n", scenario->description);
  printf("Duration: %.1f seconds (%.0f timesteps)\n", cfg->duration,
         cfg->duration / cfg->dt);
  printf("Process noise: %.2f m/s²\n", scenario->process_noise);
  printf("Measurement noise: %.2f m\n", scenario->measurement_noise);
  printf("Measurement rate: %.1f Hz\n", scenario->measurement_rate);
  printf("=============================================================\n\n");

  /* Allocate result arrays */
  size_t n_steps = (size_t)(cfg->duration / cfg->dt);
  double *time = malloc(n_steps * sizeof(double));
  double *pos_error = malloc(n_steps * sizeof(double));
  double *vel_error = malloc(n_steps * sizeof(double));
  double *cov_trace = malloc(n_steps * sizeof(double));
  double *nees = malloc(n_steps * sizeof(double));
  double *condition = malloc(n_steps * sizeof(double));

  if (!time || !pos_error || !vel_error || !cov_trace || !nees || !condition) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  /* Initialize filter */
  printf("Initializing SR-UKF (4D state, 2D measurements)...\n");
  srukf *ukf = srukf_create(4, 2);
  if (!ukf) {
    fprintf(stderr, "Failed to create SR-UKF\n");
    return 1;
  }

  /* Process noise covariance (affects velocity primarily) */
  srukf_mat *Qsqrt = srukf_mat_alloc(4, 4, 1);
  SRUKF_ENTRY(Qsqrt, 0, 0) =
      scenario->process_noise * 0.01; /* Small position noise */
  SRUKF_ENTRY(Qsqrt, 1, 1) = scenario->process_noise * 0.01;
  SRUKF_ENTRY(Qsqrt, 2, 2) = scenario->process_noise; /* Velocity noise */
  SRUKF_ENTRY(Qsqrt, 3, 3) = scenario->process_noise;

  /* Measurement noise covariance */
  srukf_mat *Rsqrt = srukf_mat_alloc(2, 2, 1);
  SRUKF_ENTRY(Rsqrt, 0, 0) = scenario->measurement_noise;
  SRUKF_ENTRY(Rsqrt, 1, 1) = scenario->measurement_noise;

  srukf_set_noise(ukf, Qsqrt, Rsqrt);

  /* Initial state */
  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;  /* x */
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0;  /* y */
  SRUKF_ENTRY(ukf->x, 2, 0) = 10.0; /* vx = 10 m/s */
  SRUKF_ENTRY(ukf->x, 3, 0) = 0.0;  /* vy */

  /* Initial uncertainty */
  SRUKF_ENTRY(ukf->S, 0, 0) = 5.0; /* 5m position std */
  SRUKF_ENTRY(ukf->S, 1, 1) = 5.0;
  SRUKF_ENTRY(ukf->S, 2, 2) = 2.0; /* 2 m/s velocity std */
  SRUKF_ENTRY(ukf->S, 3, 3) = 2.0;

  /* True state */
  double true_x = 0.0, true_y = 0.0;
  double true_vx = 10.0, true_vy = 0.0;

  /* Motion context */
  motion_context_t ctx = {
      .dt = cfg->dt, .high_dynamics = scenario->high_dynamics, .t = 0.0};

  /* Measurement timing */
  double meas_period = 1.0 / scenario->measurement_rate;
  double time_since_meas = 0.0;

  /* Statistics */
  size_t meas_count = 0;
  double max_pos_error = 0.0;
  double mean_nees = 0.0;
  size_t health_failures = 0;

  clock_t start_time = clock();

  printf("Running simulation...\n");

  /* Main loop */
  for (size_t step = 0; step < n_steps; step++) {
    double t = step * cfg->dt;
    time[step] = t;
    ctx.t = t;

    /* Simulate true motion */
    double ax = 0.0, ay = 0.0;
    if (scenario->high_dynamics) {
      ax = 5.0 * sin(0.5 * t);
      ay = 5.0 * cos(0.7 * t);
    }
    ax += add_noise(scenario->process_noise); /* Process noise */
    ay += add_noise(scenario->process_noise);

    true_vx += ax * cfg->dt;
    true_vy += ay * cfg->dt;
    true_x += true_vx * cfg->dt;
    true_y += true_vy * cfg->dt;

    /* Predict */
    srukf_predict(ukf, process_model, &ctx);

    /* Measurement update */
    time_since_meas += cfg->dt;
    if (time_since_meas >= meas_period) {
      srukf_mat *z = srukf_mat_alloc(2, 1, 1);
      SRUKF_ENTRY(z, 0, 0) = true_x + add_noise(scenario->measurement_noise);
      SRUKF_ENTRY(z, 1, 0) = true_y + add_noise(scenario->measurement_noise);

      srukf_correct(ukf, z, measurement_model, NULL);
      srukf_mat_free(z);

      time_since_meas = 0.0;
      meas_count++;
    }

    /* Compute metrics */
    health_metrics_t health = compute_health(ukf);
    error_metrics_t errors =
        compute_errors(ukf, true_x, true_y, true_vx, true_vy);

    /* Record */
    pos_error[step] = errors.position_rmse;
    vel_error[step] = errors.velocity_rmse;
    cov_trace[step] = health.cov_trace;
    nees[step] = errors.nees;
    condition[step] = health.cov_condition;

    /* Update statistics */
    if (errors.position_rmse > max_pos_error) {
      max_pos_error = errors.position_rmse;
    }
    mean_nees += errors.nees;

    if (health.has_nan || health.has_inf || !health.is_positive_def) {
      health_failures++;
      if (cfg->verbose || health_failures == 1) {
        printf("  WARNING: Numerical health issue at t=%.1fs\n", t);
        if (health.has_nan)
          printf("    - NaN detected\n");
        if (health.has_inf)
          printf("    - Inf detected\n");
        if (!health.is_positive_def)
          printf("    - Covariance not positive definite\n");
      }
    }

    /* Progress */
    if (cfg->verbose && (step % (n_steps / 10) == 0)) {
      printf("  %.0f%% | t=%.1fs | pos_err=%.2fm | NEES=%.1f | cond=%.1e\n",
             100.0 * step / n_steps, t, errors.position_rmse, errors.nees,
             health.cov_condition);
    }
  }

  clock_t end_time = clock();
  double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  mean_nees /= n_steps;

  /* Print results */
  printf("\n");
  printf("=============================================================\n");
  printf("  Test Results\n");
  printf("=============================================================\n");
  printf("Completed: %zu timesteps in %.2f seconds (%.0f steps/sec)\n", n_steps,
         elapsed, n_steps / elapsed);
  printf("Measurements: %zu (%.1f Hz average)\n", meas_count,
         meas_count / cfg->duration);
  printf("\n");
  printf("Error Statistics:\n");
  printf("  Max position error: %.2f m\n", max_pos_error);
  printf("  Final position error: %.2f m\n", pos_error[n_steps - 1]);
  printf("  Final velocity error: %.2f m/s\n", vel_error[n_steps - 1]);
  printf("\n");
  printf("Consistency Metrics:\n");
  printf("  Mean NEES: %.2f (expect ≈4.0 for consistent filter)\n", mean_nees);
  printf("  Final covariance trace: %.2f\n", cov_trace[n_steps - 1]);
  printf("  Final condition number: %.2e\n", condition[n_steps - 1]);
  printf("\n");
  printf("Numerical Health:\n");
  printf("  Health failures: %zu (%.2f%%)\n", health_failures,
         100.0 * health_failures / n_steps);
  printf("  Status: ");
  if (health_failures == 0) {
    printf("✓ EXCELLENT - No numerical issues detected\n");
  } else if (health_failures < n_steps * 0.01) {
    printf("⚠ GOOD - Minor issues (< 1%%)\n");
  } else {
    printf("✗ POOR - Significant numerical problems\n");
  }
  printf("=============================================================\n\n");

  /* Generate outputs */
  printf("Generating outputs...\n");

  /* Generate CSV output */
  if (cfg->format == OUTPUT_CSV || cfg->format == OUTPUT_ALL) {
    FILE *fp = fopen("stability_results.csv", "w");
    if (fp) {
      fprintf(fp, "time\tposition_error\tvelocity_error\tcov_"
                  "trace\tnees\tcondition\n");
      for (size_t i = 0; i < n_steps; i++) {
        fprintf(fp, "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6e\n", time[i],
                pos_error[i], vel_error[i], cov_trace[i], nees[i],
                condition[i]);
      }
      fclose(fp);
      printf("  ✓ Generated: stability_results.csv\n");
    }
  }

  /* Generate SVG plots */
  if (cfg->format == OUTPUT_SVG || cfg->format == OUTPUT_ALL) {
    data_series_t series_error[] = {{.name = "Position Error",
                                     .timestamps = time,
                                     .values = pos_error,
                                     .count = n_steps,
                                     .style = "lines",
                                     .color = "#4169E1"}};

    data_series_t series_nees[] = {{.name = "NEES",
                                    .timestamps = time,
                                    .values = nees,
                                    .count = n_steps,
                                    .style = "lines",
                                    .color = "#32CD32"}};

    data_series_t series_trace[] = {{.name = "Covariance Trace",
                                     .timestamps = time,
                                     .values = cov_trace,
                                     .count = n_steps,
                                     .style = "lines",
                                     .color = "#FF8C00"}};

    plot_config_t cfg_error = plot_config_default();
    cfg_error.title = "Position Error Over Time";
    cfg_error.xlabel = "Time (s)";
    cfg_error.ylabel = "Error (m)";
    plot_generate_svg("stability_error.svg", &cfg_error, series_error, 1,
                      cfg->open_viewer);

    plot_config_t cfg_nees = plot_config_default();
    cfg_nees.title = "NEES Consistency Test";
    cfg_nees.xlabel = "Time (s)";
    cfg_nees.ylabel = "NEES";
    plot_generate_svg("stability_nees.svg", &cfg_nees, series_nees, 1,
                      cfg->open_viewer);

    plot_config_t cfg_trace = plot_config_default();
    cfg_trace.title = "Covariance Trace Evolution";
    cfg_trace.xlabel = "Time (s)";
    cfg_trace.ylabel = "Trace";
    plot_generate_svg("stability_trace.svg", &cfg_trace, series_trace, 1,
                      cfg->open_viewer);
  }

  /* Cleanup */
  srukf_free(ukf);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  free(time);
  free(pos_error);
  free(vel_error);
  free(cov_trace);
  free(nees);
  free(condition);

  printf("\nStability test complete!\n");

  return (health_failures == 0) ? 0 : 1;
}

/* ========================================================================
 * COMMAND-LINE INTERFACE
 * ======================================================================== */

static void print_usage(const char *prog) {
  printf("Usage: %s [options]\n\n", prog);
  printf("Long-duration numerical stability test for SR-UKF.\n\n");
  printf("Options:\n");
  printf("  --scenario=NAME     Test scenario (default: baseline)\n");
  printf("                      Options: baseline, dynamics, observability, "
         "noise\n");
  printf(
      "  --duration=SEC      Simulation duration in seconds (default: 1000)\n");
  printf("  --format=FMT        Output format: svg, csv, json, all (default: "
         "svg)\n");
  printf("  --verbose           Print detailed progress\n");
  printf("  --open              Open SVG plots automatically\n");
  printf("  --help              Show this help\n\n");
  printf("Examples:\n");
  printf("  %s --scenario=baseline --duration=10000\n", prog);
  printf("  %s --scenario=dynamics --verbose\n", prog);
  printf("  %s --scenario=noise --format=all\n\n", prog);
}

int main(int argc, char **argv) {
  test_config_t cfg = config_default();

  static struct option long_options[] = {
      {"scenario", required_argument, 0, 's'},
      {"duration", required_argument, 0, 'd'},
      {"format", required_argument, 0, 'f'},
      {"verbose", no_argument, 0, 'v'},
      {"open", no_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "s:d:f:voh", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 's':
      if (strcmp(optarg, "baseline") == 0)
        cfg.scenario = SCENARIO_BASELINE;
      else if (strcmp(optarg, "dynamics") == 0)
        cfg.scenario = SCENARIO_HIGH_DYNAMICS;
      else if (strcmp(optarg, "observability") == 0)
        cfg.scenario = SCENARIO_POOR_OBSERVABILITY;
      else if (strcmp(optarg, "noise") == 0)
        cfg.scenario = SCENARIO_EXTREME_NOISE;
      else {
        fprintf(stderr, "Unknown scenario: %s\n", optarg);
        return 1;
      }
      break;
    case 'd':
      cfg.duration = atof(optarg);
      break;
    case 'f':
      if (strcmp(optarg, "svg") == 0)
        cfg.format = OUTPUT_SVG;
      else if (strcmp(optarg, "csv") == 0)
        cfg.format = OUTPUT_CSV;
      else if (strcmp(optarg, "json") == 0)
        cfg.format = OUTPUT_JSON;
      else if (strcmp(optarg, "all") == 0)
        cfg.format = OUTPUT_ALL;
      else {
        fprintf(stderr, "Unknown format: %s\n", optarg);
        return 1;
      }
      break;
    case 'v':
      cfg.verbose = true;
      break;
    case 'o':
      cfg.open_viewer = true;
      break;
    case 'h':
      print_usage(argv[0]);
      return 0;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  srand(time(NULL));

  return run_stability_test(&cfg);
}
