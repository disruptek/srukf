/**
 * @file pendulum.c
 * @brief Nonlinear Pendulum Tracking with SR-UKF
 *
 * This example demonstrates the Square-Root Unscented Kalman Filter tracking
 * a simple pendulum with nonlinear dynamics. The pendulum is a perfect
 * pedagogical example because:
 *
 * 1. **Nonlinearity is Obvious**: The restoring force involves sin(θ), making
 *    linear approximations fail visibly when the angle is large.
 *
 * 2. **Easy to Visualize**: Phase space (angle vs angular velocity) and time
 *    series plots clearly show filter performance.
 *
 * 3. **Physical Intuition**: Everyone understands a swinging pendulum, making
 *    the example relatable.
 *
 * ## The Physics
 *
 * A damped pendulum follows:
 *    dθ/dt = ω                    (angular velocity)
 *    dω/dt = -(g/L)sin(θ) - b*ω   (gravity + damping)
 *
 * Where:
 *   θ = angle from vertical (radians)
 *   ω = angular velocity (rad/s)
 *   g = gravitational acceleration (9.81 m/s²)
 *   L = pendulum length (meters)
 *   b = damping coefficient
 *
 * ## Why SR-UKF vs Linear KF?
 *
 * A linear Kalman filter would approximate sin(θ) ≈ θ, which works for small
 * angles but fails dramatically for large swings. The UKF handles the full
 * nonlinearity without requiring us to compute Jacobians.
 *
 * ## The Filtering Problem
 *
 * **State**: x = [θ, ω]ᵀ (angle and angular velocity)
 *
 * **Process Model**: f(x) propagates state forward using RK4 integration
 *   of the pendulum ODEs
 *
 * **Measurement Model**: h(x) = θ + noise (we measure angle with a noisy
 * sensor)
 *
 * **Process Noise**: Small random perturbations (wind, friction variations)
 *
 * **Measurement Noise**: Sensor noise (e.g., encoder quantization, noise)
 *
 * ## What You'll See
 *
 * The output plots show:
 * - True pendulum motion (simulated)
 * - Noisy measurements (what the sensor sees)
 * - SR-UKF estimate (filtered, smooth)
 * - Uncertainty bounds (±1σ confidence)
 *
 * Notice how the filter "smooths" the measurements and provides velocity
 * estimates even though we only measure angle!
 */

#define _USE_MATH_DEFINES
#include "../common/plot_utils.h"
#include "srukf.h"
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * CONFIGURATION & PARAMETERS
 * ======================================================================== */

/** Pendulum physical parameters */
typedef struct {
  double g; /**< Gravitational acceleration (m/s²) */
  double L; /**< Pendulum length (m) */
  double b; /**< Damping coefficient */
} pendulum_params_t;

/** Simulation configuration */
typedef struct {
  double duration;          /**< Total simulation time (seconds) */
  double dt;                /**< Timestep (seconds) */
  double measurement_rate;  /**< Measurement frequency (Hz) */
  double process_noise;     /**< Process noise std dev */
  double measurement_noise; /**< Measurement noise std dev */
  double initial_angle;     /**< Initial angle (radians) */
  double initial_velocity;  /**< Initial angular velocity (rad/s) */
  output_format_t format;   /**< Output format */
  bool open_viewer;         /**< Auto-open SVG */
} sim_config_t;

/* Default configuration */
static sim_config_t config_default(void) {
  sim_config_t cfg = {.duration = 10.0,
                      .dt = 0.01,
                      .measurement_rate = 20.0, /* 20 Hz measurements */
                      .process_noise = 0.01,
                      .measurement_noise =
                          0.1, /* 0.1 radian noise (~5.7 degrees) */
                      .initial_angle = M_PI / 4, /* 45 degrees */
                      .initial_velocity = 0.0,
                      .format = OUTPUT_SVG,
                      .open_viewer = false};
  return cfg;
}

/* ========================================================================
 * PENDULUM DYNAMICS
 * ======================================================================== */

/**
 * Pendulum ODE: dθ/dt = ω, dω/dt = -(g/L)sin(θ) - b*ω
 *
 * This function computes the time derivatives given the current state.
 * We use this for both simulation and as the process model for the UKF.
 *
 * @param theta Current angle (rad)
 * @param omega Current angular velocity (rad/s)
 * @param params Physical parameters
 * @param dtheta_dt Output: dθ/dt
 * @param domega_dt Output: dω/dt
 */
static void pendulum_derivatives(double theta, double omega,
                                 const pendulum_params_t *params,
                                 double *dtheta_dt, double *domega_dt) {
  *dtheta_dt = omega;
  *domega_dt = -(params->g / params->L) * sin(theta) - params->b * omega;
}

/**
 * RK4 integration step for pendulum
 *
 * Runge-Kutta 4th order is a standard ODE integration method that's more
 * accurate than simple Euler integration. We use this to propagate the
 * pendulum state forward in time.
 *
 * Why RK4? It gives us smooth, accurate trajectories even with relatively
 * large timesteps (dt = 0.01s). Euler would require much smaller steps.
 *
 * @param theta Current angle
 * @param omega Current angular velocity
 * @param dt Timestep
 * @param params Physical parameters
 * @param theta_next Output: angle at t+dt
 * @param omega_next Output: angular velocity at t+dt
 */
static void pendulum_rk4_step(double theta, double omega, double dt,
                              const pendulum_params_t *params,
                              double *theta_next, double *omega_next) {
  double k1_theta, k1_omega;
  double k2_theta, k2_omega;
  double k3_theta, k3_omega;
  double k4_theta, k4_omega;

  /* k1 = f(t, y) */
  pendulum_derivatives(theta, omega, params, &k1_theta, &k1_omega);

  /* k2 = f(t + dt/2, y + k1*dt/2) */
  pendulum_derivatives(theta + k1_theta * dt / 2, omega + k1_omega * dt / 2,
                       params, &k2_theta, &k2_omega);

  /* k3 = f(t + dt/2, y + k2*dt/2) */
  pendulum_derivatives(theta + k2_theta * dt / 2, omega + k2_omega * dt / 2,
                       params, &k3_theta, &k3_omega);

  /* k4 = f(t + dt, y + k3*dt) */
  pendulum_derivatives(theta + k3_theta * dt, omega + k3_omega * dt, params,
                       &k4_theta, &k4_omega);

  /* Combine: y_next = y + (dt/6)(k1 + 2*k2 + 2*k3 + k4) */
  *theta_next =
      theta + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta);
  *omega_next =
      omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega);
}

/* ========================================================================
 * SR-UKF PROCESS AND MEASUREMENT MODELS
 * ======================================================================== */

/** Context passed to process/measurement models */
typedef struct {
  pendulum_params_t params;
  double dt;
} model_context_t;

/**
 * Process model for SR-UKF: f(x_k) -> x_{k+1}
 *
 * This function tells the UKF how the state evolves over time. We simply
 * apply one RK4 integration step to each sigma point.
 *
 * The UKF will call this function multiple times (once per sigma point) to
 * understand how uncertainty propagates through the nonlinear dynamics.
 *
 * @param x_in Input state [θ, ω]
 * @param x_out Output state [θ', ω'] after one timestep
 * @param user_data Pointer to model_context_t with params and dt
 */
static void process_model(const srukf_mat *x_in, srukf_mat *x_out,
                          void *user_data) {
  model_context_t *ctx = (model_context_t *)user_data;

  double theta = SRUKF_ENTRY(x_in, 0, 0);
  double omega = SRUKF_ENTRY(x_in, 1, 0);

  double theta_next, omega_next;
  pendulum_rk4_step(theta, omega, ctx->dt, &ctx->params, &theta_next,
                    &omega_next);

  SRUKF_ENTRY(x_out, 0, 0) = theta_next;
  SRUKF_ENTRY(x_out, 1, 0) = omega_next;
}

/**
 * Measurement model for SR-UKF: h(x) -> z
 *
 * This function tells the UKF what we actually observe. In this case,
 * we measure angle but not velocity.
 *
 * The measurement model is linear here (z = θ), but the *process* model
 * is nonlinear, which is enough to require a UKF.
 *
 * @param x State [θ, ω]
 * @param z Output measurement [θ_measured]
 * @param user_data Unused
 */
static void measurement_model(const srukf_mat *x, srukf_mat *z,
                              void *user_data) {
  (void)user_data;
  SRUKF_ENTRY(z, 0, 0) = SRUKF_ENTRY(x, 0, 0); /* measure angle only */
}

/* ========================================================================
 * SIMULATION AND FILTERING
 * ======================================================================== */

/** Add Gaussian noise with given standard deviation */
static double add_noise(double std_dev) {
  /* Box-Muller transform for Gaussian noise */
  double u1 = (double)rand() / RAND_MAX;
  double u2 = (double)rand() / RAND_MAX;
  return std_dev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * Run the pendulum simulation with SR-UKF filtering
 *
 * This is the main simulation loop. At each timestep, we:
 * 1. Propagate the true pendulum state (simulation)
 * 2. Occasionally take a noisy measurement
 * 3. Run the UKF predict step
 * 4. Run the UKF correct step when measurements arrive
 * 5. Record everything for plotting
 *
 * @param cfg Simulation configuration
 * @return 0 on success, non-zero on error
 */
static int run_simulation(const sim_config_t *cfg) {
  /* Initialize pendulum parameters */
  pendulum_params_t params = {.g = 9.81, .L = 1.0, .b = 0.1};

  /* Set up model context */
  model_context_t ctx = {.params = params, .dt = cfg->dt};

  /* Calculate number of timesteps */
  size_t n_steps = (size_t)(cfg->duration / cfg->dt);
  double meas_period = 1.0 / cfg->measurement_rate;

  /* Allocate arrays for results */
  double *time = malloc(n_steps * sizeof(double));
  double *true_angle = malloc(n_steps * sizeof(double));
  double *true_velocity = malloc(n_steps * sizeof(double));
  double *measurements = malloc(n_steps * sizeof(double));
  double *est_angle = malloc(n_steps * sizeof(double));
  double *est_velocity = malloc(n_steps * sizeof(double));
  double *uncertainty = malloc(n_steps * sizeof(double));

  if (!time || !true_angle || !true_velocity || !measurements || !est_angle ||
      !est_velocity || !uncertainty) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  /* Initialize measurement array with NaN (no measurement) */
  for (size_t i = 0; i < n_steps; i++) {
    measurements[i] = NAN;
  }

  /* ====================================================================
   * CREATE AND INITIALIZE SR-UKF
   * ==================================================================== */

  printf("Initializing SR-UKF...\n");

  /* State dimension: 2 (angle, angular velocity) */
  /* Measurement dimension: 1 (angle only) */
  srukf *ukf = srukf_create(2, 1);
  if (!ukf) {
    fprintf(stderr, "Failed to create SR-UKF\n");
    return 1;
  }

  /* Set process noise covariance (square root form)
   *
   * Q = diag([q_angle², q_velocity²])
   * Qsqrt = diag([q_angle, q_velocity])
   *
   * Process noise represents unmodeled dynamics (wind gusts, friction
   * variations, etc.). We set it relatively small since our model is good.
   */
  srukf_mat *Qsqrt = srukf_mat_alloc(2, 2, 1);
  SRUKF_ENTRY(Qsqrt, 0, 0) = cfg->process_noise;
  SRUKF_ENTRY(Qsqrt, 1, 1) = cfg->process_noise;

  /* Set measurement noise covariance (square root form)
   *
   * R = [r²]
   * Rsqrt = [r]
   *
   * Measurement noise is the sensor's noise level.
   */
  srukf_mat *Rsqrt = srukf_mat_alloc(1, 1, 1);
  SRUKF_ENTRY(Rsqrt, 0, 0) = cfg->measurement_noise;

  srukf_set_noise(ukf, Qsqrt, Rsqrt);

  /* Set initial state estimate
   *
   * We assume we start with some uncertainty about the initial conditions.
   */
  SRUKF_ENTRY(ukf->x, 0, 0) = cfg->initial_angle + add_noise(0.1);
  SRUKF_ENTRY(ukf->x, 1, 0) = cfg->initial_velocity + add_noise(0.1);

  /* Set initial uncertainty */
  SRUKF_ENTRY(ukf->S, 0, 0) = 0.2;
  SRUKF_ENTRY(ukf->S, 1, 1) = 0.2;

  /* ====================================================================
   * MAIN SIMULATION LOOP
   * ==================================================================== */

  printf("Running simulation for %.1f seconds...\n", cfg->duration);

  /* Initialize true state */
  double theta_true = cfg->initial_angle;
  double omega_true = cfg->initial_velocity;
  double time_since_meas = 0.0;

  for (size_t step = 0; step < n_steps; step++) {
    double t = step * cfg->dt;
    time[step] = t;

    /* ==============================================================
     * 1. SIMULATE TRUE PENDULUM MOTION
     * ============================================================== */

    /* Add small random perturbations (process noise) */
    double theta_noisy = theta_true + add_noise(cfg->process_noise * cfg->dt);
    double omega_noisy = omega_true + add_noise(cfg->process_noise * cfg->dt);

    /* Propagate forward */
    pendulum_rk4_step(theta_noisy, omega_noisy, cfg->dt, &params, &theta_true,
                      &omega_true);

    true_angle[step] = theta_true;
    true_velocity[step] = omega_true;

    /* ==============================================================
     * 2. GENERATE MEASUREMENTS (at specified rate)
     * ============================================================== */

    time_since_meas += cfg->dt;
    bool have_measurement = (time_since_meas >= meas_period);

    if (have_measurement) {
      measurements[step] = theta_true + add_noise(cfg->measurement_noise);
      time_since_meas = 0.0;
    }

    /* ==============================================================
     * 3. SR-UKF PREDICT STEP
     *
     * Every timestep, we predict forward based on our process model.
     * This propagates both the state estimate and the uncertainty.
     * ============================================================== */

    srukf_predict(ukf, process_model, &ctx);

    /* ==============================================================
     * 4. SR-UKF CORRECT STEP (when measurement available)
     *
     * When we get a measurement, we correct our prediction.
     * This typically reduces uncertainty.
     * ============================================================== */

    if (have_measurement) {
      srukf_mat *z = srukf_mat_alloc(1, 1, 1);
      SRUKF_ENTRY(z, 0, 0) = measurements[step];

      srukf_correct(ukf, z, measurement_model, NULL);

      srukf_mat_free(z);
    }

    /* ==============================================================
     * 5. RECORD RESULTS
     * ============================================================== */

    est_angle[step] = SRUKF_ENTRY(ukf->x, 0, 0);
    est_velocity[step] = SRUKF_ENTRY(ukf->x, 1, 0);

    /* Uncertainty (±1σ) for angle */
    uncertainty[step] = SRUKF_ENTRY(ukf->S, 0, 0);

    /* Progress indicator */
    if (step % (n_steps / 10) == 0) {
      printf("  Progress: %zu%%\n", (step * 100) / n_steps);
    }
  }

  printf("Simulation complete.\n");

  /* ====================================================================
   * GENERATE OUTPUTS
   * ==================================================================== */

  printf("Generating outputs...\n");

  /* Prepare data series */
  data_series_t series[] = {{
                                .name = "True Angle",
                                .timestamps = time,
                                .values = true_angle,
                                .count = n_steps,
                                .style = "lines",
                                .color = "linecolor rgb '#00ff00'" /* Green */
                            },
                            {
                                .name = "Measurements",
                                .timestamps = time,
                                .values = measurements,
                                .count = n_steps,
                                .style = "points pt 7 ps 0.5",
                                .color = "linecolor rgb '#ff6b6b'" /* Red */
                            },
                            {
                                .name = "SR-UKF Estimate",
                                .timestamps = time,
                                .values = est_angle,
                                .count = n_steps,
                                .style = "lines lw 2",
                                .color = "linecolor rgb '#4dabf7'" /* Blue */
                            }};

  /* Output based on requested format */
  if (cfg->format == OUTPUT_CSV || cfg->format == OUTPUT_ALL) {
    printf("  Writing CSV...\n");
    plot_write_csv("pendulum", series, 3);
  }

  if (cfg->format == OUTPUT_JSON || cfg->format == OUTPUT_ALL) {
    printf("  Writing JSON...\n");
    plot_write_json("pendulum.json", series, 3);
  }

  if (cfg->format == OUTPUT_SVG || cfg->format == OUTPUT_ALL) {
    printf("  Generating SVG...\n");
    plot_config_t plot_cfg = plot_config_default();
    plot_cfg.title = "Pendulum Tracking with SR-UKF";
    plot_cfg.xlabel = "Time (s)";
    plot_cfg.ylabel = "Angle (radians)";
    plot_cfg.dark_mode = true;

    int ret = plot_generate_svg("pendulum.svg", &plot_cfg, series, 3,
                                cfg->open_viewer);
    if (ret == 0) {
      printf("  ✓ Generated: pendulum.svg\n");
    } else if (ret == 1) {
      printf("  ⚠ SVG generation skipped (gnuplot not available)\n");
      printf("    A pre-generated SVG is included in the repository.\n");
    }
  }

  /* Cleanup */
  srukf_free(ukf);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  free(time);
  free(true_angle);
  free(true_velocity);
  free(measurements);
  free(est_angle);
  free(est_velocity);
  free(uncertainty);

  printf("\nDone! Check the output files for results.\n");

  return 0;
}

/* ========================================================================
 * COMMAND-LINE INTERFACE
 * ======================================================================== */

static void print_usage(const char *prog_name) {
  printf("Usage: %s [options]\n", prog_name);
  printf("\n");
  printf("Simulate a nonlinear pendulum and track it with SR-UKF.\n");
  printf("\n");
  printf("Options:\n");
  printf("  --duration=SECONDS    Simulation duration (default: 10.0)\n");
  printf("  --noise=LEVEL         Measurement noise std dev (default: 0.1)\n");
  printf("  --format=FORMAT       Output format: svg, csv, json, all (default: "
         "svg)\n");
  printf("  --open                Auto-open SVG after generation\n");
  printf("  --help                Show this help message\n");
  printf("\n");
  printf("Examples:\n");
  printf("  %s --duration=20 --open\n", prog_name);
  printf("  %s --noise=0.2 --format=all\n", prog_name);
  printf("\n");
}

int main(int argc, char **argv) {
  sim_config_t cfg = config_default();

  static struct option long_options[] = {
      {"duration", required_argument, 0, 'd'},
      {"noise", required_argument, 0, 'n'},
      {"format", required_argument, 0, 'f'},
      {"open", no_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "d:n:f:oh", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'd':
      cfg.duration = atof(optarg);
      break;
    case 'n':
      cfg.measurement_noise = atof(optarg);
      break;
    case 'f':
      if (strcmp(optarg, "csv") == 0)
        cfg.format = OUTPUT_CSV;
      else if (strcmp(optarg, "json") == 0)
        cfg.format = OUTPUT_JSON;
      else if (strcmp(optarg, "svg") == 0)
        cfg.format = OUTPUT_SVG;
      else if (strcmp(optarg, "all") == 0)
        cfg.format = OUTPUT_ALL;
      else {
        fprintf(stderr, "Unknown format: %s\n", optarg);
        return 1;
      }
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

  /* Seed random number generator */
  srand(time(NULL));

  /* Run simulation */
  return run_simulation(&cfg);
}
