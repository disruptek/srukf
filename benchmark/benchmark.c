/* --------------------------------------------------------------------
 * benchmark.c - Performance benchmark for SR-UKF
 *
 * Measures predict/correct performance across various dimensions.
 * Reports timing statistics (mean, stddev, min, max).
 * -------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "srukf.h"

/* Number of iterations per benchmark */
#define WARMUP_ITERS 100
#define BENCH_ITERS 10000

/* ---------------- Timing helpers ------------------------------------ */
static double get_time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

typedef struct {
  double mean;
  double stddev;
  double min;
  double max;
} stats_t;

static stats_t compute_stats(const double *samples, int n) {
  stats_t s = {0, 0, samples[0], samples[0]};

  /* Mean */
  for (int i = 0; i < n; i++) {
    s.mean += samples[i];
    if (samples[i] < s.min)
      s.min = samples[i];
    if (samples[i] > s.max)
      s.max = samples[i];
  }
  s.mean /= n;

  /* Stddev */
  for (int i = 0; i < n; i++) {
    double d = samples[i] - s.mean;
    s.stddev += d * d;
  }
  s.stddev = sqrt(s.stddev / n);

  return s;
}

/* ---------------- Process/measurement models ------------------------ */
/* Simple linear process: x' = x (identity) */
static void process_identity(const lah_mat *x, lah_mat *xp, void *user) {
  (void)user;
  for (lah_index i = 0; i < x->nR; i++)
    LAH_ENTRY(xp, i, 0) = LAH_ENTRY(x, i, 0);
}

/* Simple linear measurement: z = H*x where H selects first M states */
static void meas_linear(const lah_mat *x, lah_mat *z, void *user) {
  (void)user;
  for (lah_index i = 0; i < z->nR; i++)
    LAH_ENTRY(z, i, 0) = LAH_ENTRY(x, i, 0);
}

/* Nonlinear process: x' = x + 0.01*sin(x) */
static void process_nonlinear(const lah_mat *x, lah_mat *xp, void *user) {
  (void)user;
  for (lah_index i = 0; i < x->nR; i++)
    LAH_ENTRY(xp, i, 0) = LAH_ENTRY(x, i, 0) + 0.01 * sin(LAH_ENTRY(x, i, 0));
}

/* Nonlinear measurement: z = x^2 */
static void meas_nonlinear(const lah_mat *x, lah_mat *z, void *user) {
  (void)user;
  for (lah_index i = 0; i < z->nR; i++) {
    lah_value v = LAH_ENTRY(x, i, 0);
    LAH_ENTRY(z, i, 0) = v * v;
  }
}

/* ---------------- Benchmark routines -------------------------------- */
static void bench_predict(int N, int M, int iters, double *samples,
                          void (*f)(const lah_mat *, lah_mat *, void *)) {
  srukf *ukf = srukf_create(N, M);
  if (!ukf) {
    fprintf(stderr, "Failed to create filter\n");
    return;
  }

  /* Initialize noise matrices */
  lah_mat *Q = allocMatrixNow(N, N);
  lah_mat *R = allocMatrixNow(M, M);
  for (int i = 0; i < N; i++)
    LAH_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; i++)
    LAH_ENTRY(R, i, i) = 0.1;
  srukf_set_noise(ukf, Q, R);
  srukf_set_scale(ukf, 1e-3, 2.0, 0.0);

  /* Initialize state */
  for (int i = 0; i < N; i++)
    LAH_ENTRY(ukf->x, i, 0) = 0.1 * (i + 1);

  /* Pre-allocate workspace */
  srukf_alloc_workspace(ukf);

  /* Warmup */
  for (int i = 0; i < WARMUP_ITERS; i++)
    srukf_predict(ukf, f, NULL);

  /* Benchmark */
  for (int i = 0; i < iters; i++) {
    double t0 = get_time_ns();
    srukf_predict(ukf, f, NULL);
    double t1 = get_time_ns();
    samples[i] = t1 - t0;
  }

  lah_matFree(Q);
  lah_matFree(R);
  srukf_free(ukf);
}

static void bench_correct(int N, int M, int iters, double *samples,
                          void (*h)(const lah_mat *, lah_mat *, void *)) {
  srukf *ukf = srukf_create(N, M);
  if (!ukf) {
    fprintf(stderr, "Failed to create filter\n");
    return;
  }

  /* Initialize noise matrices */
  lah_mat *Q = allocMatrixNow(N, N);
  lah_mat *R = allocMatrixNow(M, M);
  for (int i = 0; i < N; i++)
    LAH_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; i++)
    LAH_ENTRY(R, i, i) = 0.1;
  srukf_set_noise(ukf, Q, R);
  srukf_set_scale(ukf, 1e-3, 2.0, 0.0);

  /* Initialize state */
  for (int i = 0; i < N; i++)
    LAH_ENTRY(ukf->x, i, 0) = 0.1 * (i + 1);

  /* Measurement vector */
  lah_mat *z = allocMatrixNow(M, 1);
  for (int i = 0; i < M; i++)
    LAH_ENTRY(z, i, 0) = 0.1 * (i + 1);

  /* Pre-allocate workspace */
  srukf_alloc_workspace(ukf);

  /* Warmup */
  for (int i = 0; i < WARMUP_ITERS; i++)
    srukf_correct(ukf, z, h, NULL);

  /* Benchmark */
  for (int i = 0; i < iters; i++) {
    double t0 = get_time_ns();
    srukf_correct(ukf, z, h, NULL);
    double t1 = get_time_ns();
    samples[i] = t1 - t0;
  }

  lah_matFree(Q);
  lah_matFree(R);
  lah_matFree(z);
  srukf_free(ukf);
}

static void bench_predict_to(int N, int M, int iters, double *samples,
                             void (*f)(const lah_mat *, lah_mat *, void *)) {
  srukf *ukf = srukf_create(N, M);
  if (!ukf) {
    fprintf(stderr, "Failed to create filter\n");
    return;
  }

  /* Initialize noise matrices */
  lah_mat *Q = allocMatrixNow(N, N);
  lah_mat *R = allocMatrixNow(M, M);
  for (int i = 0; i < N; i++)
    LAH_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; i++)
    LAH_ENTRY(R, i, i) = 0.1;
  srukf_set_noise(ukf, Q, R);
  srukf_set_scale(ukf, 1e-3, 2.0, 0.0);

  /* User-managed state buffers */
  lah_mat *x = allocMatrixNow(N, 1);
  lah_mat *S = allocMatrixNow(N, N);
  for (int i = 0; i < N; i++) {
    LAH_ENTRY(x, i, 0) = 0.1 * (i + 1);
    LAH_ENTRY(S, i, i) = 0.001;
  }

  /* Pre-allocate workspace */
  srukf_alloc_workspace(ukf);

  /* Warmup */
  for (int i = 0; i < WARMUP_ITERS; i++)
    srukf_predict_to(ukf, x, S, f, NULL);

  /* Benchmark */
  for (int i = 0; i < iters; i++) {
    double t0 = get_time_ns();
    srukf_predict_to(ukf, x, S, f, NULL);
    double t1 = get_time_ns();
    samples[i] = t1 - t0;
  }

  lah_matFree(Q);
  lah_matFree(R);
  lah_matFree(x);
  lah_matFree(S);
  srukf_free(ukf);
}

static void bench_correct_to(int N, int M, int iters, double *samples,
                             void (*h)(const lah_mat *, lah_mat *, void *)) {
  srukf *ukf = srukf_create(N, M);
  if (!ukf) {
    fprintf(stderr, "Failed to create filter\n");
    return;
  }

  /* Initialize noise matrices */
  lah_mat *Q = allocMatrixNow(N, N);
  lah_mat *R = allocMatrixNow(M, M);
  for (int i = 0; i < N; i++)
    LAH_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; i++)
    LAH_ENTRY(R, i, i) = 0.1;
  srukf_set_noise(ukf, Q, R);
  srukf_set_scale(ukf, 1e-3, 2.0, 0.0);

  /* User-managed state buffers */
  lah_mat *x = allocMatrixNow(N, 1);
  lah_mat *S = allocMatrixNow(N, N);
  for (int i = 0; i < N; i++) {
    LAH_ENTRY(x, i, 0) = 0.1 * (i + 1);
    LAH_ENTRY(S, i, i) = 0.001;
  }

  /* Measurement vector */
  lah_mat *z = allocMatrixNow(M, 1);
  for (int i = 0; i < M; i++)
    LAH_ENTRY(z, i, 0) = 0.1 * (i + 1);

  /* Pre-allocate workspace */
  srukf_alloc_workspace(ukf);

  /* Warmup */
  for (int i = 0; i < WARMUP_ITERS; i++)
    srukf_correct_to(ukf, x, S, z, h, NULL);

  /* Benchmark */
  for (int i = 0; i < iters; i++) {
    double t0 = get_time_ns();
    srukf_correct_to(ukf, x, S, z, h, NULL);
    double t1 = get_time_ns();
    samples[i] = t1 - t0;
  }

  lah_matFree(Q);
  lah_matFree(R);
  lah_matFree(x);
  lah_matFree(S);
  lah_matFree(z);
  srukf_free(ukf);
}

/* ---------------- Main ---------------------------------------------- */
int main(void) {
  double *samples = (double *)malloc(BENCH_ITERS * sizeof(double));
  if (!samples) {
    fprintf(stderr, "Failed to allocate samples buffer\n");
    return 1;
  }

  printf("SR-UKF Benchmark\n");
  printf("================\n");
  printf("Warmup iterations: %d\n", WARMUP_ITERS);
  printf("Benchmark iterations: %d\n\n", BENCH_ITERS);

  /* Test configurations: (N, M) pairs */
  int configs[][2] = {{3, 2}, {6, 3}, {10, 5}, {15, 8}, {20, 10}};
  int n_configs = sizeof(configs) / sizeof(configs[0]);

  printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "Dim", "Operation", "Mean (us)",
         "Stddev (us)", "Min (us)", "Max (us)");
  printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "---", "---------", "---------",
         "-----------", "--------", "--------");

  for (int c = 0; c < n_configs; c++) {
    int N = configs[c][0];
    int M = configs[c][1];
    char dim_str[32];
    snprintf(dim_str, sizeof(dim_str), "%dx%d", N, M);

    /* Predict (linear) */
    bench_predict(N, M, BENCH_ITERS, samples, process_identity);
    stats_t s = compute_stats(samples, BENCH_ITERS);
    printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", dim_str, "predict",
           s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

    /* Predict_to (linear) */
    bench_predict_to(N, M, BENCH_ITERS, samples, process_identity);
    s = compute_stats(samples, BENCH_ITERS);
    printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", "", "predict_to",
           s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

    /* Correct (linear) */
    bench_correct(N, M, BENCH_ITERS, samples, meas_linear);
    s = compute_stats(samples, BENCH_ITERS);
    printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", "", "correct",
           s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

    /* Correct_to (linear) */
    bench_correct_to(N, M, BENCH_ITERS, samples, meas_linear);
    s = compute_stats(samples, BENCH_ITERS);
    printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", "", "correct_to",
           s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

    printf("\n");
  }

  /* Nonlinear benchmark for a typical robotics size */
  printf("\nNonlinear models (N=6, M=3):\n");
  printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "Model", "Operation",
         "Mean (us)", "Stddev (us)", "Min (us)", "Max (us)");
  printf("%-8s %-12s %-12s %-12s %-12s %-12s\n", "-----", "---------",
         "---------", "-----------", "--------", "--------");

  bench_predict(6, 3, BENCH_ITERS, samples, process_nonlinear);
  stats_t s = compute_stats(samples, BENCH_ITERS);
  printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", "nonlin", "predict",
         s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

  bench_predict_to(6, 3, BENCH_ITERS, samples, process_nonlinear);
  s = compute_stats(samples, BENCH_ITERS);
  printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", "", "predict_to",
         s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

  bench_correct(6, 3, BENCH_ITERS, samples, meas_nonlinear);
  s = compute_stats(samples, BENCH_ITERS);
  printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", "", "correct",
         s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

  bench_correct_to(6, 3, BENCH_ITERS, samples, meas_nonlinear);
  s = compute_stats(samples, BENCH_ITERS);
  printf("%-8s %-12s %12.3f %12.3f %12.3f %12.3f\n", "", "correct_to",
         s.mean / 1000.0, s.stddev / 1000.0, s.min / 1000.0, s.max / 1000.0);

  free(samples);
  printf("\nBenchmark complete.\n");
  return 0;
}
