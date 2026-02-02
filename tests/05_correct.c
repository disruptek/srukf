/* --------------------------------------------------------------------
 * 05_correct.c - Correction step tests for srukf_correct
 *
 * Tests the measurement update (correction) step with various scenarios:
 * - Basic correction with known expected values
 * - Multiple sequential corrections
 * - Covariance reduction verification
 * - High/low measurement noise ratios
 * - Zero measurement
 * - Large measurement values
 * - Error handling
 * -------------------------------------------------------------------- */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "srukf.h"

#define EPS       1e-6
#define EPS_LOOSE 1e-3

/* Helper: compute trace of S*S' (sum of squared singular values = sum of
 * eigenvalues of P) */
static double covariance_trace(const srukf_mat *S) {
  double trace = 0.0;
  for (srukf_index i = 0; i < S->n_rows; i++)
    for (srukf_index j = 0; j < S->n_cols; j++) {
      double v = SRUKF_ENTRY(S, i, j);
      trace += v * v;
    }
  return trace;
}

/* Measurement model: identity on first M states */
static void meas_identity(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; i++)
    SRUKF_ENTRY(z, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Measurement model: sum of states */
static void meas_sum(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  double sum = 0.0;
  for (srukf_index i = 0; i < x->n_rows; i++)
    sum += SRUKF_ENTRY(x, i, 0);
  SRUKF_ENTRY(z, 0, 0) = sum;
}

/* Test 1: Basic correction with identity measurement */
static void test_basic_correction(void) {
  const srukf_index N = 2, M = 2;

  /* Create noise matrices */
  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  /* Set initial state to [0, 0] with identity covariance */
  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;
  SRUKF_ENTRY(ukf->S, 0, 1) = 0.0;
  SRUKF_ENTRY(ukf->S, 1, 0) = 0.0;

  /* Measurement: [1, 2] */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 1.0;
  SRUKF_ENTRY(z, 1, 0) = 2.0;

  /* Save prior covariance trace */
  double prior_trace = covariance_trace(ukf->S);

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should move toward measurement */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) > 0.0);
  assert(SRUKF_ENTRY(ukf->x, 1, 0) > 0.0);

  /* Covariance should decrease after correction */
  double post_trace = covariance_trace(ukf->S);
  assert(post_trace < prior_trace);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_basic_correction OK\n");
}

/* Test 2: Multiple sequential corrections should converge */
static void test_sequential_corrections(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.01;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  /* Initial state far from true value */
  SRUKF_ENTRY(ukf->x, 0, 0) = 10.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 10.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;

  /* True state is [1, 2] - measure it repeatedly */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 1.0;
  SRUKF_ENTRY(z, 1, 0) = 2.0;

  /* Run multiple corrections */
  double prev_error = INFINITY;
  for (int iter = 0; iter < 10; iter++) {
    srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);

    double err0 = SRUKF_ENTRY(ukf->x, 0, 0) - 1.0;
    double err1 = SRUKF_ENTRY(ukf->x, 1, 0) - 2.0;
    double error = sqrt(err0 * err0 + err1 * err1);

    /* Error should decrease (or stay small) */
    assert(error <= prev_error + EPS);
    prev_error = error;
  }

  /* After 10 corrections, should be close to true value */
  assert(fabs(SRUKF_ENTRY(ukf->x, 0, 0) - 1.0) < 0.5);
  assert(fabs(SRUKF_ENTRY(ukf->x, 1, 0) - 2.0) < 0.5);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_sequential_corrections OK\n");
}

/* Test 3: Zero measurement */
static void test_zero_measurement(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 5.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 5.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;

  /* Zero measurement */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 0.0;
  SRUKF_ENTRY(z, 1, 0) = 0.0;

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should move toward zero */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) < 5.0);
  assert(SRUKF_ENTRY(ukf->x, 1, 0) < 5.0);

  /* Values should still be finite */
  assert(isfinite(SRUKF_ENTRY(ukf->x, 0, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 1, 0)));

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_zero_measurement OK\n");
}

/* Test 4: Large measurement values */
static void test_large_measurement(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;

  /* Large measurement */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 1e6;
  SRUKF_ENTRY(z, 1, 0) = 1e6;

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should move toward large values */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) > 0.0);
  assert(SRUKF_ENTRY(ukf->x, 1, 0) > 0.0);

  /* All values should be finite */
  for (srukf_index i = 0; i < N; i++) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
    for (srukf_index j = 0; j < N; j++)
      assert(isfinite(SRUKF_ENTRY(ukf->S, i, j)));
  }

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_large_measurement OK\n");
}

/* Test 5: High measurement noise (low trust in measurement) */
static void test_high_measurement_noise(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.01;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 100.0; /* Very high measurement noise */

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;

  double x0_prior = SRUKF_ENTRY(ukf->x, 0, 0);
  double x1_prior = SRUKF_ENTRY(ukf->x, 1, 0);

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 100.0;
  SRUKF_ENTRY(z, 1, 0) = 100.0;

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* With high measurement noise, state should barely change */
  double change0 = fabs(SRUKF_ENTRY(ukf->x, 0, 0) - x0_prior);
  double change1 = fabs(SRUKF_ENTRY(ukf->x, 1, 0) - x1_prior);
  assert(change0 < 1.0); /* Small change despite large innovation */
  assert(change1 < 1.0);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_high_meas_noise OK\n");
}

/* Test 6: Low measurement noise (high trust in measurement) */
static void test_low_measurement_noise(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.001; /* Very low measurement noise */

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 5.0;
  SRUKF_ENTRY(z, 1, 0) = 5.0;

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* With low measurement noise, state should jump close to measurement */
  assert(fabs(SRUKF_ENTRY(ukf->x, 0, 0) - 5.0) < 0.5);
  assert(fabs(SRUKF_ENTRY(ukf->x, 1, 0) - 5.0) < 0.5);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_low_meas_noise  OK\n");
}

/* Test 7: Partial observability (M < N) */
static void test_partial_observability(void) {
  const srukf_index N = 3, M = 1;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  SRUKF_ENTRY(Rsqrt, 0, 0) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  /* Initial state [1, 2, 3] */
  SRUKF_ENTRY(ukf->x, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 2.0;
  SRUKF_ENTRY(ukf->x, 2, 0) = 3.0;
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(ukf->S, i, i) = 1.0;

  /* Only observe sum of all states */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 10.0; /* Measured sum */

  srukf_return rc = srukf_correct(ukf, z, meas_sum, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* All states should be updated (sum measurement couples all) */
  /* State should adjust to match the measurement sum */
  assert(isfinite(SRUKF_ENTRY(ukf->x, 0, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 1, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 2, 0)));

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_partial_obs     OK\n");
}

/* Test 8: 1D case */
static void test_1d(void) {
  const srukf_index N = 1, M = 1;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  SRUKF_ENTRY(Qsqrt, 0, 0) = 0.1;
  SRUKF_ENTRY(Rsqrt, 0, 0) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 5.0;

  double prior_var = SRUKF_ENTRY(ukf->S, 0, 0) * SRUKF_ENTRY(ukf->S, 0, 0);

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should move toward measurement */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) > 0.0);
  assert(SRUKF_ENTRY(ukf->x, 0, 0) < 5.0);

  /* Variance should decrease */
  double post_var = SRUKF_ENTRY(ukf->S, 0, 0) * SRUKF_ENTRY(ukf->S, 0, 0);
  assert(post_var < prior_var);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_1d              OK\n");
}

/* Test 9: Higher dimension (5D state, 3D measurement) */
static void test_5d(void) {
  const srukf_index N = 5, M = 3;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  /* Set non-trivial initial state */
  for (srukf_index i = 0; i < N; i++) {
    SRUKF_ENTRY(ukf->x, i, 0) = (double)(i + 1);
    SRUKF_ENTRY(ukf->S, i, i) = 1.0;
  }

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(z, i, 0) = 10.0;

  double prior_trace = covariance_trace(ukf->S);

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* All values should be finite */
  for (srukf_index i = 0; i < N; i++) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
    for (srukf_index j = 0; j < N; j++)
      assert(isfinite(SRUKF_ENTRY(ukf->S, i, j)));
  }

  /* Covariance should decrease */
  double post_trace = covariance_trace(ukf->S);
  assert(post_trace < prior_trace);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_5d              OK\n");
}

/* Test 10: Measurement that matches prediction (no innovation) */
static void test_no_innovation(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  /* State is [2, 3] */
  SRUKF_ENTRY(ukf->x, 0, 0) = 2.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 3.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 0.5;
  SRUKF_ENTRY(ukf->S, 1, 1) = 0.5;

  /* Measurement exactly matches state (identity measurement, zero innovation)
   */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 2.0;
  SRUKF_ENTRY(z, 1, 0) = 3.0;

  double x0_prior = SRUKF_ENTRY(ukf->x, 0, 0);
  double x1_prior = SRUKF_ENTRY(ukf->x, 1, 0);

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should stay close to prior (measurement matches prediction) */
  assert(fabs(SRUKF_ENTRY(ukf->x, 0, 0) - x0_prior) < 0.1);
  assert(fabs(SRUKF_ENTRY(ukf->x, 1, 0) - x1_prior) < 0.1);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_no_innovation   OK\n");
}

/* Test 11: Covariance should always decrease (or stay same) */
static void test_covariance_reduction(void) {
  const srukf_index N = 3, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.2;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  SRUKF_ENTRY(ukf->S, 0, 0) = 2.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 2.0;
  SRUKF_ENTRY(ukf->S, 2, 2) = 2.0;

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 1.0;
  SRUKF_ENTRY(z, 1, 0) = 1.0;

  /* Run multiple corrections with random measurements */
  for (int iter = 0; iter < 5; iter++) {
    double prior_trace = covariance_trace(ukf->S);

    SRUKF_ENTRY(z, 0, 0) = (double)(iter + 1);
    SRUKF_ENTRY(z, 1, 0) = (double)(iter + 2);

    srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);

    double post_trace = covariance_trace(ukf->S);
    /* Covariance should not increase significantly */
    assert(post_trace <= prior_trace + EPS);
  }

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_cov_reduction   OK\n");
}

/* Test 12: Error - NULL filter */
static void test_error_null_filter(void) {
  srukf_mat *z = SRUKF_MAT_ALLOC(2, 1);
  SRUKF_ENTRY(z, 0, 0) = 1.0;
  SRUKF_ENTRY(z, 1, 0) = 1.0;

  srukf_return rc = srukf_correct(NULL, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(z);
  printf("  test_err_null_filter OK\n");
}

/* Test 13: Error - NULL measurement */
static void test_error_null_measurement(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  srukf_return rc = srukf_correct(ukf, NULL, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_err_null_meas   OK\n");
}

/* Test 14: Error - NULL measurement function */
static void test_error_null_meas_func(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 1.0;
  SRUKF_ENTRY(z, 1, 0) = 1.0;

  srukf_return rc = srukf_correct(ukf, z, NULL, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_err_null_func   OK\n");
}

/* Test 15: Original basic test (from old 05_correct.c) */
static void test_original(void) {
  const srukf_index N = 3, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  SRUKF_ENTRY(Qsqrt, 0, 0) = 0.1;
  SRUKF_ENTRY(Qsqrt, 1, 1) = 0.1;
  SRUKF_ENTRY(Qsqrt, 2, 2) = 0.1;

  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  SRUKF_ENTRY(Rsqrt, 0, 0) = 0.2;
  SRUKF_ENTRY(Rsqrt, 1, 1) = 0.2;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 0.1;
  SRUKF_ENTRY(z, 1, 0) = 0.2;

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Basic sanity checks */
  assert(isfinite(SRUKF_ENTRY(ukf->x, 0, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 1, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 2, 0)));

  srukf_free(ukf);
  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  printf("  test_original        OK\n");
}

/* Test 16: correct_to - transactional in-place API */
static void test_correct_to(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  /* Set initial state */
  SRUKF_ENTRY(ukf->x, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 2.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;

  /* Save filter state */
  double x0_orig = SRUKF_ENTRY(ukf->x, 0, 0);
  double x1_orig = SRUKF_ENTRY(ukf->x, 1, 0);
  double S00_orig = SRUKF_ENTRY(ukf->S, 0, 0);
  double S11_orig = SRUKF_ENTRY(ukf->S, 1, 1);

  /* Allocate user buffers, copy state */
  srukf_mat *x_user = SRUKF_MAT_ALLOC(N, 1);
  srukf_mat *S_user = SRUKF_MAT_ALLOC(N, N);
  for (srukf_index i = 0; i < N; i++) {
    SRUKF_ENTRY(x_user, i, 0) = SRUKF_ENTRY(ukf->x, i, 0);
    for (srukf_index j = 0; j < N; j++)
      SRUKF_ENTRY(S_user, i, j) = SRUKF_ENTRY(ukf->S, i, j);
  }

  /* Measurement */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z, 0, 0) = 5.0;
  SRUKF_ENTRY(z, 1, 0) = 6.0;

  /* Call transactional API */
  srukf_return rc = srukf_correct_to(ukf, x_user, S_user, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Filter state should be unchanged */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) == x0_orig);
  assert(SRUKF_ENTRY(ukf->x, 1, 0) == x1_orig);
  assert(SRUKF_ENTRY(ukf->S, 0, 0) == S00_orig);
  assert(SRUKF_ENTRY(ukf->S, 1, 1) == S11_orig);

  /* User buffers should be updated (state moves toward measurement) */
  assert(SRUKF_ENTRY(x_user, 0, 0) > x0_orig);
  assert(SRUKF_ENTRY(x_user, 1, 0) > x1_orig);

  /* Covariance should decrease */
  double prior_trace = S00_orig * S00_orig + S11_orig * S11_orig;
  double post_trace = covariance_trace(S_user);
  assert(post_trace < prior_trace);

  srukf_mat_free(x_user);
  srukf_mat_free(S_user);
  srukf_mat_free(z);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_correct_to      OK\n");
}

/* Test 17: correct_to chaining - speculative measurement comparison */
static void test_correct_to_chaining(void) {
  const srukf_index N = 2, M = 2;

  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (srukf_index i = 0; i < M; i++)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  assert(ukf);

  /* Set initial state with large uncertainty */
  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0;

  /* Save filter state */
  double x0_orig = SRUKF_ENTRY(ukf->x, 0, 0);
  double x1_orig = SRUKF_ENTRY(ukf->x, 1, 0);

  /* Three candidate measurements - evaluate which brings us closest to target
   */
  srukf_mat *z1 = SRUKF_MAT_ALLOC(M, 1);
  srukf_mat *z2 = SRUKF_MAT_ALLOC(M, 1);
  srukf_mat *z3 = SRUKF_MAT_ALLOC(M, 1);
  SRUKF_ENTRY(z1, 0, 0) = 1.0;
  SRUKF_ENTRY(z1, 1, 0) = 1.0; /* measurement 1 */
  SRUKF_ENTRY(z2, 0, 0) = 5.0;
  SRUKF_ENTRY(z2, 1, 0) = 5.0; /* measurement 2 */
  SRUKF_ENTRY(z3, 0, 0) = 3.0;
  SRUKF_ENTRY(z3, 1, 0) = 3.0; /* measurement 3 */

  /* Target state we want to reach */
  double target_x0 = 3.0, target_x1 = 3.0;

  /* Speculatively evaluate each measurement */
  srukf_mat *x_test = SRUKF_MAT_ALLOC(N, 1);
  srukf_mat *S_test = SRUKF_MAT_ALLOC(N, N);
  double best_dist = INFINITY;
  int best_idx = -1;

  srukf_mat *measurements[3] = {z1, z2, z3};
  for (int i = 0; i < 3; i++) {
    /* Reset to filter state */
    for (srukf_index k = 0; k < N; k++) {
      SRUKF_ENTRY(x_test, k, 0) = SRUKF_ENTRY(ukf->x, k, 0);
      for (srukf_index l = 0; l < N; l++)
        SRUKF_ENTRY(S_test, k, l) = SRUKF_ENTRY(ukf->S, k, l);
    }

    srukf_return rc = srukf_correct_to(ukf, x_test, S_test, measurements[i],
                                     meas_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);

    double dx = SRUKF_ENTRY(x_test, 0, 0) - target_x0;
    double dy = SRUKF_ENTRY(x_test, 1, 0) - target_x1;
    double dist = sqrt(dx * dx + dy * dy);

    if (dist < best_dist) {
      best_dist = dist;
      best_idx = i;
    }
  }

  /* z3 (3,3) should be best since it equals target */
  assert(best_idx == 2);

  /* Filter state should be unchanged after all speculative evaluations */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) == x0_orig);
  assert(SRUKF_ENTRY(ukf->x, 1, 0) == x1_orig);

  srukf_mat_free(x_test);
  srukf_mat_free(S_test);
  srukf_mat_free(z1);
  srukf_mat_free(z2);
  srukf_mat_free(z3);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  srukf_free(ukf);
  printf("  test_correct_to_chain OK\n");
}

int main(void) {
  printf("Running srukf_correct tests...\n");

  test_basic_correction();
  test_sequential_corrections();
  test_zero_measurement();
  test_large_measurement();
  test_high_measurement_noise();
  test_low_measurement_noise();
  test_partial_observability();
  test_1d();
  test_5d();
  test_no_innovation();
  test_covariance_reduction();
  test_error_null_filter();
  test_error_null_measurement();
  test_error_null_meas_func();
  test_original();
  test_correct_to();
  test_correct_to_chaining();

  printf("correction tests passed.\n");
  return 0;
}
