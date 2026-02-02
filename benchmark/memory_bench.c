/* --------------------------------------------------------------------
 * memory_bench.c - Memory usage benchmark for SR-UKF
 *
 * Measures and reports memory consumption for various filter dimensions.
 * Compares theoretical (calculated) vs measured memory usage.
 * -------------------------------------------------------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "srukf.h"

/* ---------------- Memory calculation -------------------------------- */

/* Calculate theoretical memory usage for a filter of given dimensions.
 * This matches the allocations in srukf.c exactly. */
static size_t calc_filter_memory(int N, int M) {
  size_t n_sigma = 2 * N + 1;
  size_t val_size = sizeof(srukf_value);
  size_t mat_struct = sizeof(srukf_mat);
  size_t total = 0;

  /* Filter struct itself */
  total += sizeof(srukf);

  /* Core matrices (x, S, Qsqrt, Rsqrt) */
  total += mat_struct + N * 1 * val_size; /* x: N x 1 */
  total += mat_struct + N * N * val_size; /* S: N x N */
  total += mat_struct + N * N * val_size; /* Qsqrt: N x N */
  total += mat_struct + M * M * val_size; /* Rsqrt: M x M */

  /* Weight vectors */
  total += n_sigma * val_size; /* wm */
  total += n_sigma * val_size; /* wc */

  return total;
}

static size_t calc_workspace_memory(int N, int M) {
  size_t n_sigma = 2 * N + 1;
  size_t val_size = sizeof(srukf_value);
  size_t mat_struct = sizeof(srukf_mat);
  size_t total = 0;

  /* Workspace struct (opaque, estimate based on known contents:
   * 2 size_t + 19 pointers + padding ≈ 168 bytes on 64-bit) */
  total += 168;

  /* Predict temporaries */
  total += mat_struct + N * n_sigma * val_size; /* Xsig: N x (2N+1) */
  total += mat_struct + N * n_sigma * val_size; /* Ysig_N: N x (2N+1) */
  total += mat_struct + N * 1 * val_size;       /* x_pred: N x 1 */
  total += mat_struct + N * N * val_size;       /* S_tmp: N x N */

  /* Correct temporaries */
  total += mat_struct + M * n_sigma * val_size; /* Ysig_M: M x (2N+1) */
  total += mat_struct + M * 1 * val_size;       /* y_mean: M x 1 */
  total += mat_struct + N * M * val_size;       /* Pxz: N x M */
  total += mat_struct + N * M * val_size;       /* K: N x M */
  total += mat_struct + M * 1 * val_size;       /* innov: M x 1 */
  total += mat_struct + N * 1 * val_size;       /* x_new: N x 1 */
  total += mat_struct + N * N * val_size;       /* S_new: N x N */
  total += mat_struct + N * 1 * val_size;       /* dx: N x 1 */
  total += mat_struct + N * M * val_size;       /* tmp1: N x M */

  /* SR-UKF specific */
  total += mat_struct + N * n_sigma * val_size; /* Dev_N: N x (2N+1) */
  total += mat_struct + M * n_sigma * val_size; /* Dev_M: M x (2N+1) */
  total += mat_struct + (n_sigma + N + 1) * N * val_size; /* qr_work_N */
  total += mat_struct + (n_sigma + M + 1) * M * val_size; /* qr_work_M */
  total += mat_struct + M * M * val_size;                 /* Syy: M x M */

  /* Small buffers */
  total += N * val_size; /* tau_N */
  total += M * val_size; /* tau_M */
  size_t max_dim = (N > M) ? N : M;
  total += max_dim * val_size; /* downdate_work */
  total += N * val_size;       /* dev0_N */
  total += M * val_size;       /* dev0_M */

  return total;
}

/* ---------------- Memory measurement -------------------------------- */

#ifdef __linux__
#include <unistd.h>

/* Get current process memory usage from /proc/self/statm (Linux only) */
static size_t get_rss_bytes(void) {
  FILE *f = fopen("/proc/self/statm", "r");
  if (!f)
    return 0;

  long pages = 0;
  long rss = 0;
  if (fscanf(f, "%ld %ld", &pages, &rss) != 2) {
    fclose(f);
    return 0;
  }
  fclose(f);

  long page_size = sysconf(_SC_PAGESIZE);
  return (size_t)(rss * page_size);
}
#else
/* Fallback for non-Linux: return 0 (unknown) */
static size_t get_rss_bytes(void) {
  return 0;
}
#endif

/* Number of filters to create for averaging (reduces page granularity noise) */
#define NUM_FILTERS 1000

/* Identity process model for memory population */
static void identity_f(const srukf_mat *x, srukf_mat *xp, void *user) {
  (void)user;
  for (srukf_index i = 0; i < x->n_rows; i++)
    SRUKF_ENTRY(xp, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Identity measurement model for memory population */
static void identity_h(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; i++)
    SRUKF_ENTRY(z, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Measure actual memory by creating multiple filters, running them, and
 * averaging. Running predict/correct ensures all workspace memory is touched
 * (defeating lazy alloc). */
static size_t measure_filter_memory(int N, int M, int include_workspace) {
  srukf *filters[NUM_FILTERS];

  /* Set up noise matrices (reused) */
  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  if (!Q || !R || !z) {
    srukf_mat_free(Q);
    srukf_mat_free(R);
    srukf_mat_free(z);
    return 0;
  }
  for (int i = 0; i < N; i++)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; i++) {
    SRUKF_ENTRY(R, i, i) = 0.1;
    SRUKF_ENTRY(z, i, 0) = 0.5;
  }

  /* Baseline measurement */
  size_t before = get_rss_bytes();

  /* Create multiple filters */
  for (int f = 0; f < NUM_FILTERS; f++) {
    filters[f] = srukf_create(N, M);
    if (!filters[f]) {
      /* Cleanup on failure */
      for (int j = 0; j < f; j++)
        srukf_free(filters[j]);
      srukf_mat_free(Q);
      srukf_mat_free(R);
      srukf_mat_free(z);
      return 0;
    }
    srukf_set_noise(filters[f], Q, R);
    if (include_workspace) {
      srukf_alloc_workspace(filters[f]);
      /* Run multiple predict + correct cycles to touch all workspace memory */
      for (int cycle = 0; cycle < 5; cycle++) {
        srukf_predict(filters[f], identity_f, NULL);
        srukf_correct(filters[f], z, identity_h, NULL);
      }
    }
  }

  /* Measure after allocation and population */
  size_t after = get_rss_bytes();

  /* Cleanup */
  for (int f = 0; f < NUM_FILTERS; f++) {
    srukf_free(filters[f]);
  }
  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);

  /* Return average per filter */
  if (after > before) {
    return (after - before) / NUM_FILTERS;
  }
  return 0;
}

/* ---------------- Main ---------------------------------------------- */
int main(void) {
  printf("SR-UKF Memory Benchmark\n");
  printf("=======================\n");
  printf("sizeof(srukf_value) = %zu bytes (%s precision)\n",
         sizeof(srukf_value), sizeof(srukf_value) == 4 ? "single" : "double");
  printf("sizeof(srukf_mat) = %zu bytes\n\n", sizeof(srukf_mat));

  /* Test configurations */
  int configs[][2] = {{3, 2},   {6, 3},   {10, 5},  {15, 8},
                      {20, 10}, {50, 25}, {100, 50}};
  int n_configs = sizeof(configs) / sizeof(configs[0]);

  printf("%-10s %12s %12s %12s %12s\n", "Dim", "Filter (KB)", "Workspace",
         "Total Calc", "Total Meas");
  printf("%-10s %12s %12s %12s %12s\n", "---", "-----------", "---------",
         "----------", "----------");

  for (int c = 0; c < n_configs; c++) {
    int N = configs[c][0];
    int M = configs[c][1];

    size_t filter_mem = calc_filter_memory(N, M);
    size_t workspace_mem = calc_workspace_memory(N, M);
    size_t total_calc = filter_mem + workspace_mem;
    size_t total_meas = measure_filter_memory(N, M, 1);

    char dim_str[32];
    snprintf(dim_str, sizeof(dim_str), "%dx%d", N, M);

    printf("%-10s %12.2f %12.2f %12.2f %12.2f\n", dim_str, filter_mem / 1024.0,
           workspace_mem / 1024.0, total_calc / 1024.0, total_meas / 1024.0);
  }

  printf("\n");
  printf("Note: 'Total Meas' uses RSS from /proc/self/statm (Linux).\n");
  printf("      Measured values may be 0 on non-Linux or due to page "
         "granularity.\n");
  printf("      'Total Calc' is the authoritative value based on actual "
         "allocations.\n");

  return 0;
}
