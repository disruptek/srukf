/**
 * @file gps_imu.c
 * @brief GPS + IMU Sensor Fusion in 1D with SR-UKF
 *
 * This example demonstrates a critical real-world use case: fusing continuous
 * IMU measurements with intermittent GPS updates. This scenario appears in:
 * - Autonomous vehicles navigating urban canyons
 * - Drones flying under bridges or through tunnels
 * - Smartphones switching between GPS and dead reckoning
 *
 * ## The Problem
 *
 * You're tracking a vehicle moving in 1D (think: along a straight road).
 *
 * **Sensors:**
 * - **IMU (accelerometer)**: Measures acceleration continuously at 100 Hz
 *   - Always available, but drifts over time (integration errors accumulate)
 *   - Moderate noise (~0.5 m/s²)
 *
 * - **GPS**: Measures position intermittently at 1 Hz
 *   - Accurate when available (~5m noise)
 *   - **Drops out** unpredictably (urban canyons, tunnels, buildings)
 *   - Crucial for correcting IMU drift
 *
 * **Challenge**: Maintain accurate position and velocity estimates even during
 * GPS dropouts that may last several seconds.
 *
 * ## Why This Matters (from the README)
 *
 * This directly addresses the README's motivation: "GPS provides position
 * fixes, but cuts out unpredictably under bridges, in urban canyons, and near
 * buildings."
 *
 * During GPS dropout:
 * - Uncertainty **grows** (we're relying on drifty IMU)
 * - Standard UKF might lose positive-definiteness after many dropout cycles
 * - **SR-UKF stays numerically stable indefinitely**
 *
 * When GPS returns:
 * - Uncertainty **shrinks** (GPS correction resets the estimate)
 * - This repeated grow/shrink cycle stresses the covariance matrix
 * - Square-root formulation prevents numerical issues
 *
 * ## The State and Dynamics
 *
 * **State**: x = [position, velocity, acceleration]ᵀ (3D)
 *
 * We track acceleration as part of the state because:
 * 1. The IMU measures it (so we can correct acceleration biases)
 * 2. It allows modeling acceleration changes (jerk)
 *
 * **Process Model**: Constant acceleration with jerk noise
 *   position_{k+1} = position_k + velocity_k * dt + 0.5 * accel_k * dt²
 *   velocity_{k+1} = velocity_k + accel_k * dt
 *   accel_{k+1}    = accel_k + jerk_noise
 *
 * This is a "nearly constant acceleration" model - we assume acceleration
 * doesn't change much between timesteps (process noise accounts for changes).
 *
 * **Measurement Models**:
 * 1. GPS: z_gps = position + noise (when available)
 * 2. IMU: z_imu = acceleration + noise (always available)
 *
 * ## What You'll See
 *
 * The plots show three key phenomena:
 *
 * 1. **Position Plot**: SR-UKF estimate vs true position vs GPS measurements
 *    - Notice how estimate stays smooth even when GPS drops out
 *    - Raw GPS has gaps; SR-UKF fills them with IMU integration
 *
 * 2. **Uncertainty Plot**: ±1σ bands around position estimate
 *    - **Grows** during GPS dropout (relying on noisy IMU)
 *    - **Shrinks** when GPS returns (measurement correction)
 *    - This "breathing" pattern is characteristic of intermittent updates
 *
 * 3. **GPS Availability**: Timeline showing when GPS is available
 *    - Correlate this with uncertainty changes
 *    - During long dropouts, uncertainty can grow significantly
 *
 * ## Numerical Stability Insight
 *
 * This example demonstrates why SR-UKF matters for long-duration operation:
 * - Each GPS dropout → measurement update → cycle stresses the covariance
 * - Standard UKF: After thousands of cycles, P loses positive-definiteness
 * - **SR-UKF: Maintains S=sqrt(P), numerically stable indefinitely**
 *
 * Try running with `--duration=3600` (1 hour) to see long-term stability.
 */

#define _USE_MATH_DEFINES
#include "../common/plot_utils.h"
#include "srukf.h"
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

/* ========================================================================
 * CONFIGURATION & PARAMETERS
 * ======================================================================== */

/** Simulation configuration */
typedef struct {
  double duration;           /**< Total simulation time (seconds) */
  double dt;                 /**< Timestep (seconds) */
  double imu_rate;           /**< IMU sample rate (Hz) */
  double gps_rate;           /**< GPS update rate when available (Hz) */
  double gps_dropout_prob;   /**< Probability per second of GPS dropout */
  double gps_return_prob;    /**< Probability per second of GPS return */
  double imu_noise;          /**< IMU noise std dev (m/s²) */
  double gps_noise;          /**< GPS noise std dev (m) */
  double process_noise_jerk; /**< Process noise for jerk (m/s³) */
  double initial_position;   /**< Initial position (m) */
  double initial_velocity;   /**< Initial velocity (m/s) */
  double initial_accel;      /**< Initial acceleration (m/s²) */
  output_format_t format;    /**< Output format */
  bool open_viewer;          /**< Auto-open SVG */
} sim_config_t;

/* Default configuration - simulates urban driving with GPS dropouts */
static sim_config_t config_default(void) {
  sim_config_t cfg = {
      .duration = 60.0,          /* 1 minute */
      .dt = 0.01,                /* 100 Hz simulation */
      .imu_rate = 100.0,         /* 100 Hz IMU */
      .gps_rate = 1.0,           /* 1 Hz GPS */
      .gps_dropout_prob = 0.1,   /* ~10% chance per second of dropout */
      .gps_return_prob = 0.3,    /* ~30% chance per second of return */
      .imu_noise = 0.5,          /* 0.5 m/s² IMU noise */
      .gps_noise = 5.0,          /* 5 m GPS noise */
      .process_noise_jerk = 0.5, /* 0.5 m/s³ jerk noise */
      .initial_position = 0.0,
      .initial_velocity = 10.0, /* 10 m/s (~36 km/h) */
      .initial_accel = 0.0,
      .format = OUTPUT_SVG,
      .open_viewer = false};
  return cfg;
}

/* ========================================================================
 * MOTION DYNAMICS
 * ======================================================================== */

/**
 * 1D motion with constant acceleration model
 *
 * This is a kinematic model assuming acceleration is piecewise constant.
 * The "jerk" (change in acceleration) is modeled as process noise.
 *
 * Equations:
 *   p' = p + v*dt + 0.5*a*dt²
 *   v' = v + a*dt
 *   a' = a + jerk*dt (jerk is process noise)
 *
 * @param pos Current position
 * @param vel Current velocity
 * @param acc Current acceleration
 * @param dt Timestep
 * @param jerk Jerk (acceleration change) - from process noise
 * @param pos_next Output: position at t+dt
 * @param vel_next Output: velocity at t+dt
 * @param acc_next Output: acceleration at t+dt
 */
static void motion_step(double pos, double vel, double acc, double dt,
                        double jerk, double *pos_next, double *vel_next,
                        double *acc_next) {
  /* Constant acceleration kinematics */
  *pos_next = pos + vel * dt + 0.5 * acc * dt * dt;
  *vel_next = vel + acc * dt;
  *acc_next = acc + jerk * dt;
}

/* ========================================================================
 * SR-UKF PROCESS AND MEASUREMENT MODELS
 * ======================================================================== */

/** Context for process model */
typedef struct {
  double dt;
} process_context_t;

/**
 * Process model: propagate [position, velocity, acceleration] forward
 *
 * This tells the UKF how the state evolves. We use the constant acceleration
 * kinematic equations with zero jerk (process noise models jerk separately).
 *
 * @param x_in Input state [p, v, a]
 * @param x_out Output state [p', v', a']
 * @param user_data Pointer to process_context_t with dt
 */
static void process_model(const srukf_mat *x_in, srukf_mat *x_out,
                          void *user_data) {
  process_context_t *ctx = (process_context_t *)user_data;

  double pos = SRUKF_ENTRY(x_in, 0, 0);
  double vel = SRUKF_ENTRY(x_in, 1, 0);
  double acc = SRUKF_ENTRY(x_in, 2, 0);

  double pos_next, vel_next, acc_next;
  motion_step(pos, vel, acc, ctx->dt, 0.0, &pos_next, &vel_next, &acc_next);

  SRUKF_ENTRY(x_out, 0, 0) = pos_next;
  SRUKF_ENTRY(x_out, 1, 0) = vel_next;
  SRUKF_ENTRY(x_out, 2, 0) = acc_next;
}

/**
 * GPS measurement model: h(x) = position
 *
 * GPS directly measures position (with noise).
 *
 * @param x State [p, v, a]
 * @param z Output measurement [p]
 * @param user_data Unused
 */
static void gps_measurement_model(const srukf_mat *x, srukf_mat *z,
                                  void *user_data) {
  (void)user_data;
  SRUKF_ENTRY(z, 0, 0) = SRUKF_ENTRY(x, 0, 0); /* measure position */
}

/**
 * IMU measurement model: h(x) = acceleration
 *
 * IMU measures acceleration (with noise and bias).
 *
 * @param x State [p, v, a]
 * @param z Output measurement [a]
 * @param user_data Unused
 */
static void imu_measurement_model(const srukf_mat *x, srukf_mat *z,
                                  void *user_data) {
  (void)user_data;
  SRUKF_ENTRY(z, 0, 0) = SRUKF_ENTRY(x, 2, 0); /* measure acceleration */
}

/* ========================================================================
 * GPS DROPOUT SIMULATION
 * ======================================================================== */

/**
 * Simulate GPS availability with dropout/return dynamics
 *
 * Models urban canyon scenario where GPS availability changes stochastically:
 * - When available: some probability of losing signal per second
 * - When unavailable: some probability of regaining signal per second
 *
 * @param is_available Current GPS availability state
 * @param dt Timestep
 * @param dropout_prob Probability per second of dropout (when available)
 * @param return_prob Probability per second of return (when unavailable)
 * @return Updated GPS availability
 */
static bool update_gps_availability(bool is_available, double dt,
                                    double dropout_prob, double return_prob) {
  double rand_val = (double)rand() / RAND_MAX;

  if (is_available) {
    /* Chance of dropout */
    if (rand_val < dropout_prob * dt) {
      return false;
    }
  } else {
    /* Chance of return */
    if (rand_val < return_prob * dt) {
      return true;
    }
  }

  return is_available;
}

/* ========================================================================
 * NOISE GENERATION
 * ======================================================================== */

/** Add Gaussian noise with given standard deviation */
static double add_noise(double std_dev) {
  double u1 = (double)rand() / RAND_MAX;
  double u2 = (double)rand() / RAND_MAX;
  return std_dev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ========================================================================
 * SIMULATION AND FILTERING
 * ======================================================================== */

/**
 * Run GPS+IMU fusion simulation with SR-UKF
 *
 * Main simulation loop:
 * 1. Propagate true motion (with process noise)
 * 2. Generate IMU measurement (always)
 * 3. Generate GPS measurement (when available)
 * 4. Update GPS availability state
 * 5. SR-UKF predict step
 * 6. SR-UKF correct with IMU
 * 7. SR-UKF correct with GPS (if available)
 * 8. Record results
 *
 * @param cfg Simulation configuration
 * @return 0 on success
 */
static int run_simulation(const sim_config_t *cfg) {
  size_t n_steps = (size_t)(cfg->duration / cfg->dt);
  double imu_period = 1.0 / cfg->imu_rate;
  double gps_period = 1.0 / cfg->gps_rate;

  /* Allocate result arrays */
  double *time = malloc(n_steps * sizeof(double));
  double *true_pos = malloc(n_steps * sizeof(double));
  double *true_vel = malloc(n_steps * sizeof(double));
  double *gps_meas = malloc(n_steps * sizeof(double));
  double *est_pos = malloc(n_steps * sizeof(double));
  double *est_vel = malloc(n_steps * sizeof(double));
  double *pos_uncertainty = malloc(n_steps * sizeof(double));
  double *gps_available = malloc(n_steps * sizeof(double));

  if (!time || !true_pos || !true_vel || !gps_meas || !est_pos || !est_vel ||
      !pos_uncertainty || !gps_available) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  /* Initialize with NaN (no measurement) */
  for (size_t i = 0; i < n_steps; i++) {
    gps_meas[i] = NAN;
    gps_available[i] = 0.0;
  }

  /* ====================================================================
   * CREATE AND INITIALIZE SR-UKF
   * ==================================================================== */

  printf("Initializing SR-UKF for GPS+IMU fusion...\n");
  printf("  State: [position, velocity, acceleration]\n");
  printf("  Measurements: GPS (position) + IMU (acceleration)\n");
  printf("  GPS dropout simulation enabled\n\n");

  srukf *ukf = srukf_create(3, 1); /* 3D state, 1D measurements */
  if (!ukf) {
    fprintf(stderr, "Failed to create SR-UKF\n");
    return 1;
  }

  /* Process noise: mainly affects acceleration (jerk)
   *
   * Q = diag([q_p², q_v², q_a²])
   * We set q_p and q_v small (position/velocity are well-modeled)
   * and q_a larger (acceleration changes are less predictable)
   */
  srukf_mat *Qsqrt = srukf_mat_alloc(3, 3, 1);
  SRUKF_ENTRY(Qsqrt, 0, 0) =
      cfg->process_noise_jerk * 0.01;                       /* position noise */
  SRUKF_ENTRY(Qsqrt, 1, 1) = cfg->process_noise_jerk * 0.1; /* velocity noise */
  SRUKF_ENTRY(Qsqrt, 2, 2) =
      cfg->process_noise_jerk; /* acceleration (jerk) noise */

  /* Create measurement noise matrices for GPS and IMU */
  srukf_mat *R_gps = srukf_mat_alloc(1, 1, 1);
  SRUKF_ENTRY(R_gps, 0, 0) = cfg->gps_noise;

  srukf_mat *R_imu = srukf_mat_alloc(1, 1, 1);
  SRUKF_ENTRY(R_imu, 0, 0) = cfg->imu_noise;

  /* Set initial state estimate */
  SRUKF_ENTRY(ukf->x, 0, 0) = cfg->initial_position;
  SRUKF_ENTRY(ukf->x, 1, 0) = cfg->initial_velocity;
  SRUKF_ENTRY(ukf->x, 2, 0) = cfg->initial_accel;

  /* Set initial uncertainty */
  SRUKF_ENTRY(ukf->S, 0, 0) = 10.0; /* 10m position uncertainty */
  SRUKF_ENTRY(ukf->S, 1, 1) = 2.0;  /* 2 m/s velocity uncertainty */
  SRUKF_ENTRY(ukf->S, 2, 2) = 1.0;  /* 1 m/s² acceleration uncertainty */

  /* Set initial noise covariance (will be updated before each measurement) */
  srukf_set_noise(ukf, Qsqrt, R_imu);

  /* ====================================================================
   * MAIN SIMULATION LOOP
   * ==================================================================== */

  printf("Running simulation for %.1f seconds...\n", cfg->duration);
  printf("  IMU: %.0f Hz (always available)\n", cfg->imu_rate);
  printf("  GPS: %.0f Hz (intermittent)\n", cfg->gps_rate);
  printf("  Dropout probability: %.1f%% per second\n",
         cfg->gps_dropout_prob * 100);
  printf("  Return probability: %.1f%% per second\n\n",
         cfg->gps_return_prob * 100);

  /* Initialize true state */
  double pos = cfg->initial_position;
  double vel = cfg->initial_velocity;
  double acc = cfg->initial_accel;

  /* GPS availability state */
  bool gps_is_available = true;

  /* Measurement timing */
  double time_since_imu = 0.0;
  double time_since_gps = 0.0;

  /* Process model context */
  process_context_t proc_ctx = {.dt = cfg->dt};

  /* Statistics */
  size_t gps_updates = 0;
  size_t imu_updates = 0;

  for (size_t step = 0; step < n_steps; step++) {
    double t = step * cfg->dt;
    time[step] = t;

    /* ==============================================================
     * 1. SIMULATE TRUE MOTION
     * ============================================================== */

    /* Add process noise (jerk) */
    double jerk = add_noise(cfg->process_noise_jerk);

    double pos_next, vel_next, acc_next;
    motion_step(pos, vel, acc, cfg->dt, jerk, &pos_next, &vel_next, &acc_next);

    pos = pos_next;
    vel = vel_next;
    acc = acc_next;

    true_pos[step] = pos;
    true_vel[step] = vel;

    /* ==============================================================
     * 2. UPDATE GPS AVAILABILITY
     * ============================================================== */

    gps_is_available = update_gps_availability(
        gps_is_available, cfg->dt, cfg->gps_dropout_prob, cfg->gps_return_prob);
    gps_available[step] = gps_is_available ? 1.0 : 0.0;

    /* ==============================================================
     * 3. SR-UKF PREDICT STEP (every timestep)
     * ============================================================== */

    srukf_predict(ukf, process_model, &proc_ctx);

    /* ==============================================================
     * 4. IMU MEASUREMENT UPDATE (at IMU rate)
     * ============================================================== */

    time_since_imu += cfg->dt;
    if (time_since_imu >= imu_period) {
      double imu_meas = acc + add_noise(cfg->imu_noise);

      srukf_mat *z_imu = srukf_mat_alloc(1, 1, 1);
      SRUKF_ENTRY(z_imu, 0, 0) = imu_meas;

      srukf_set_noise(ukf, Qsqrt, R_imu); /* Set measurement noise for IMU */
      srukf_correct(ukf, z_imu, imu_measurement_model, NULL);

      srukf_mat_free(z_imu);

      time_since_imu = 0.0;
      imu_updates++;
    }

    /* ==============================================================
     * 5. GPS MEASUREMENT UPDATE (at GPS rate, when available)
     *
     * This is the key: GPS updates are intermittent.
     * During dropouts, uncertainty grows.
     * When GPS returns, uncertainty shrinks.
     * ============================================================== */

    time_since_gps += cfg->dt;
    bool do_gps_update = (time_since_gps >= gps_period) && gps_is_available;

    if (do_gps_update) {
      double gps_pos = pos + add_noise(cfg->gps_noise);
      gps_meas[step] = gps_pos;

      srukf_mat *z_gps = srukf_mat_alloc(1, 1, 1);
      SRUKF_ENTRY(z_gps, 0, 0) = gps_pos;

      srukf_set_noise(ukf, Qsqrt, R_gps); /* Set measurement noise for GPS */
      srukf_correct(ukf, z_gps, gps_measurement_model, NULL);

      srukf_mat_free(z_gps);

      time_since_gps = 0.0;
      gps_updates++;
    }

    /* ==============================================================
     * 6. RECORD RESULTS
     * ============================================================== */

    est_pos[step] = SRUKF_ENTRY(ukf->x, 0, 0);
    est_vel[step] = SRUKF_ENTRY(ukf->x, 1, 0);
    pos_uncertainty[step] = SRUKF_ENTRY(ukf->S, 0, 0);

    /* Progress indicator */
    if (step % (n_steps / 10) == 0) {
      printf("  Progress: %zu%% (GPS updates: %zu, IMU updates: %zu)\n",
             (step * 100) / n_steps, gps_updates, imu_updates);
    }
  }

  printf("\nSimulation complete.\n");
  printf("  Total GPS updates: %zu\n", gps_updates);
  printf("  Total IMU updates: %zu\n", imu_updates);
  printf("  GPS availability: %.1f%%\n",
         (100.0 * gps_updates) / (n_steps / (gps_period / cfg->dt)));

  /* ====================================================================
   * GENERATE OUTPUTS
   * ==================================================================== */

  printf("\nGenerating outputs...\n");

  /* Position plot */
  data_series_t pos_series[] = {{.name = "True Position",
                                 .timestamps = time,
                                 .values = true_pos,
                                 .count = n_steps,
                                 .style = "lines",
                                 .color = "linecolor rgb '#00ff00'"},
                                {.name = "GPS Measurements",
                                 .timestamps = time,
                                 .values = gps_meas,
                                 .count = n_steps,
                                 .style = "points pt 7 ps 0.5",
                                 .color = "linecolor rgb '#ff6b6b'"},
                                {.name = "SR-UKF Estimate",
                                 .timestamps = time,
                                 .values = est_pos,
                                 .count = n_steps,
                                 .style = "lines lw 2",
                                 .color = "linecolor rgb '#4dabf7'"}};

  if (cfg->format == OUTPUT_CSV || cfg->format == OUTPUT_ALL) {
    printf("  Writing CSV...\n");
    /* Write truth vs estimate separately */
    data_series_t truth_series[] = {pos_series[0]};    /* True Position */
    data_series_t estimate_series[] = {pos_series[2]}; /* SR-UKF Estimate */
    plot_write_csv("gps_imu_truth", truth_series, 1);
    plot_write_csv("gps_imu_estimate", estimate_series, 1);
  }

  if (cfg->format == OUTPUT_JSON || cfg->format == OUTPUT_ALL) {
    printf("  Writing JSON...\n");
    plot_write_json("gps_imu.json", pos_series, 3);
  }

  if (cfg->format == OUTPUT_SVG || cfg->format == OUTPUT_ALL) {
    printf("  Generating position plot...\n");
    plot_config_t plot_cfg = plot_config_default();
    plot_cfg.title = "GPS + IMU Fusion with SR-UKF";
    plot_cfg.xlabel = "Time (s)";
    plot_cfg.ylabel = "Position (m)";

    int ret = plot_generate_svg("gps_imu_position.svg", &plot_cfg, pos_series,
                                3, false);
    if (ret == 0)
      printf("  ✓ Generated: gps_imu_position.svg\n");

    /* Uncertainty plot */
    printf("  Generating uncertainty plot...\n");
    data_series_t unc_series[] = {{.name = "Position Uncertainty (±1σ)",
                                   .timestamps = time,
                                   .values = pos_uncertainty,
                                   .count = n_steps,
                                   .style = "lines lw 2",
                                   .color = "linecolor rgb '#ffd43b'"}};

    plot_cfg.title = "Position Uncertainty Over Time";
    plot_cfg.ylabel = "Uncertainty (m, ±1σ)";
    ret = plot_generate_svg("gps_imu_uncertainty.svg", &plot_cfg, unc_series, 1,
                            false);
    if (ret == 0)
      printf("  ✓ Generated: gps_imu_uncertainty.svg\n");

    /* GPS availability plot */
    printf("  Generating GPS availability timeline...\n");
    data_series_t gps_series[] = {{.name = "GPS Available",
                                   .timestamps = time,
                                   .values = gps_available,
                                   .count = n_steps,
                                   .style = "lines lw 2",
                                   .color = "linecolor rgb '#51cf66'"}};

    plot_cfg.title = "GPS Availability";
    plot_cfg.ylabel = "Available (1=yes, 0=no)";
    ret = plot_generate_svg("gps_imu_availability.svg", &plot_cfg, gps_series,
                            1, cfg->open_viewer);
    if (ret == 0)
      printf("  ✓ Generated: gps_imu_availability.svg\n");
  }

  /* Cleanup */
  srukf_free(ukf);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(R_gps);
  srukf_mat_free(R_imu);
  free(time);
  free(true_pos);
  free(true_vel);
  free(gps_meas);
  free(est_pos);
  free(est_vel);
  free(pos_uncertainty);
  free(gps_available);

  printf("\nDone! Check the output files.\n");
  printf("\nKey observations:\n");
  printf("  • Position plot: Estimate bridges GPS gaps\n");
  printf(
      "  • Uncertainty plot: Grows during dropouts, shrinks on GPS return\n");
  printf("  • Availability plot: Correlate dropout periods with uncertainty\n");

  return 0;
}

/* ========================================================================
 * COMMAND-LINE INTERFACE
 * ======================================================================== */

static void print_usage(const char *prog_name) {
  printf("Usage: %s [options]\n", prog_name);
  printf("\n");
  printf("Simulate GPS+IMU sensor fusion in 1D with intermittent GPS.\n");
  printf("\n");
  printf("Options:\n");
  printf("  --duration=SECONDS    Simulation duration (default: 60.0)\n");
  printf(
      "  --gps-noise=LEVEL     GPS noise std dev in meters (default: 5.0)\n");
  printf("  --imu-noise=LEVEL     IMU noise std dev in m/s² (default: 0.5)\n");
  printf("  --dropout=PROB        GPS dropout probability per second (default: "
         "0.1)\n");
  printf("  --format=FORMAT       Output format: svg, csv, json, all (default: "
         "svg)\n");
  printf("  --open                Auto-open SVG after generation\n");
  printf("  --help                Show this help message\n");
  printf("\n");
  printf("Examples:\n");
  printf("  %s --duration=120 --dropout=0.3\n", prog_name);
  printf("  %s --gps-noise=10 --format=all\n", prog_name);
  printf("\n");
}

int main(int argc, char **argv) {
  sim_config_t cfg = config_default();

  static struct option long_options[] = {
      {"duration", required_argument, 0, 'd'},
      {"gps-noise", required_argument, 0, 'g'},
      {"imu-noise", required_argument, 0, 'i'},
      {"dropout", required_argument, 0, 'r'},
      {"format", required_argument, 0, 'f'},
      {"open", no_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "d:g:i:r:f:oh", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'd':
      cfg.duration = atof(optarg);
      break;
    case 'g':
      cfg.gps_noise = atof(optarg);
      break;
    case 'i':
      cfg.imu_noise = atof(optarg);
      break;
    case 'r':
      cfg.gps_dropout_prob = atof(optarg);
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
