/* --------------------------------------------------------------------
 * 45_accessors.c - Tests for state/covariance accessor functions
 *
 * Tests srukf_get_state, srukf_set_state, srukf_get_sqrt_cov,
 * srukf_set_sqrt_cov, srukf_reset, srukf_state_dim, srukf_meas_dim.
 *
 * This test links against the public API (libsrukf.so).
 * -------------------------------------------------------------------- */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "srukf.h"

#define EPS 1e-12

/* Helper: create filter with initialized noise */
static srukf *create_test_filter(int N, int M) {
  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  assert(Qsqrt && Rsqrt);

  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Qsqrt, i, i) = 0.1;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(Rsqrt, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Qsqrt, Rsqrt);
  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
  return ukf;
}

/* ========================= srukf_state_dim ========================= */

static void test_state_dim_valid(void) {
  srukf *ukf = create_test_filter(5, 3);
  assert(ukf);

  assert(srukf_state_dim(ukf) == 5);

  srukf_free(ukf);
  printf("  test_state_dim_valid OK\n");
}

static void test_state_dim_null(void) {
  assert(srukf_state_dim(NULL) == 0);
  printf("  test_state_dim_null  OK\n");
}

/* ========================= srukf_meas_dim ========================== */

static void test_meas_dim_valid(void) {
  srukf *ukf = create_test_filter(5, 3);
  assert(ukf);

  assert(srukf_meas_dim(ukf) == 3);

  srukf_free(ukf);
  printf("  test_meas_dim_valid  OK\n");
}

static void test_meas_dim_null(void) {
  assert(srukf_meas_dim(NULL) == 0);
  printf("  test_meas_dim_null   OK\n");
}

/* ========================= srukf_get_state ========================= */

static void test_get_state_valid(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Set internal state directly for testing */
  SRUKF_ENTRY(ukf->x, 0, 0) = 1.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 2.0;
  SRUKF_ENTRY(ukf->x, 2, 0) = 3.0;

  /* Get state into buffer */
  srukf_mat *x_out = SRUKF_MAT_ALLOC(3, 1);
  assert(x_out);

  srukf_return rc = srukf_get_state(ukf, x_out);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify copy */
  assert(fabs(SRUKF_ENTRY(x_out, 0, 0) - 1.0) < EPS);
  assert(fabs(SRUKF_ENTRY(x_out, 1, 0) - 2.0) < EPS);
  assert(fabs(SRUKF_ENTRY(x_out, 2, 0) - 3.0) < EPS);

  srukf_mat_free(x_out);
  srukf_free(ukf);
  printf("  test_get_state_valid OK\n");
}

static void test_get_state_null_ukf(void) {
  srukf_mat *x_out = SRUKF_MAT_ALLOC(3, 1);
  assert(x_out);

  srukf_return rc = srukf_get_state(NULL, x_out);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(x_out);
  printf("  test_get_state_null_ukf OK\n");
}

static void test_get_state_null_out(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  srukf_return rc = srukf_get_state(ukf, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(ukf);
  printf("  test_get_state_null_out OK\n");
}

static void test_get_state_dim_mismatch(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Wrong number of rows */
  srukf_mat *x_wrong = SRUKF_MAT_ALLOC(4, 1);
  assert(x_wrong);

  srukf_return rc = srukf_get_state(ukf, x_wrong);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(x_wrong);

  /* Wrong number of columns */
  srukf_mat *x_wrong2 = SRUKF_MAT_ALLOC(3, 2);
  assert(x_wrong2);

  rc = srukf_get_state(ukf, x_wrong2);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(x_wrong2);
  srukf_free(ukf);
  printf("  test_get_state_dim   OK\n");
}

/* ========================= srukf_set_state ========================= */

static void test_set_state_valid(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Create input state */
  srukf_mat *x_in = SRUKF_MAT_ALLOC(3, 1);
  assert(x_in);
  SRUKF_ENTRY(x_in, 0, 0) = 10.0;
  SRUKF_ENTRY(x_in, 1, 0) = 20.0;
  SRUKF_ENTRY(x_in, 2, 0) = 30.0;

  srukf_return rc = srukf_set_state(ukf, x_in);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify internal state was updated */
  assert(fabs(SRUKF_ENTRY(ukf->x, 0, 0) - 10.0) < EPS);
  assert(fabs(SRUKF_ENTRY(ukf->x, 1, 0) - 20.0) < EPS);
  assert(fabs(SRUKF_ENTRY(ukf->x, 2, 0) - 30.0) < EPS);

  srukf_mat_free(x_in);
  srukf_free(ukf);
  printf("  test_set_state_valid OK\n");
}

static void test_set_state_null_ukf(void) {
  srukf_mat *x_in = SRUKF_MAT_ALLOC(3, 1);
  assert(x_in);

  srukf_return rc = srukf_set_state(NULL, x_in);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(x_in);
  printf("  test_set_state_null_ukf OK\n");
}

static void test_set_state_null_in(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  srukf_return rc = srukf_set_state(ukf, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(ukf);
  printf("  test_set_state_null_in OK\n");
}

static void test_set_state_dim_mismatch(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Wrong dimensions */
  srukf_mat *x_wrong = SRUKF_MAT_ALLOC(5, 1);
  assert(x_wrong);

  srukf_return rc = srukf_set_state(ukf, x_wrong);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(x_wrong);
  srukf_free(ukf);
  printf("  test_set_state_dim   OK\n");
}

/* ========================= srukf_get_sqrt_cov ====================== */

static void test_get_sqrt_cov_valid(void) {
  srukf *ukf = create_test_filter(2, 1);
  assert(ukf);

  /* Set internal S directly */
  SRUKF_ENTRY(ukf->S, 0, 0) = 0.5;
  SRUKF_ENTRY(ukf->S, 0, 1) = 0.0;
  SRUKF_ENTRY(ukf->S, 1, 0) = 0.1;
  SRUKF_ENTRY(ukf->S, 1, 1) = 0.4;

  srukf_mat *S_out = SRUKF_MAT_ALLOC(2, 2);
  assert(S_out);

  srukf_return rc = srukf_get_sqrt_cov(ukf, S_out);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify copy */
  assert(fabs(SRUKF_ENTRY(S_out, 0, 0) - 0.5) < EPS);
  assert(fabs(SRUKF_ENTRY(S_out, 1, 0) - 0.1) < EPS);
  assert(fabs(SRUKF_ENTRY(S_out, 1, 1) - 0.4) < EPS);

  srukf_mat_free(S_out);
  srukf_free(ukf);
  printf("  test_get_sqrt_cov_valid OK\n");
}

static void test_get_sqrt_cov_null_ukf(void) {
  srukf_mat *S_out = SRUKF_MAT_ALLOC(2, 2);
  assert(S_out);

  srukf_return rc = srukf_get_sqrt_cov(NULL, S_out);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(S_out);
  printf("  test_get_sqrt_cov_null_ukf OK\n");
}

static void test_get_sqrt_cov_null_out(void) {
  srukf *ukf = create_test_filter(2, 1);
  assert(ukf);

  srukf_return rc = srukf_get_sqrt_cov(ukf, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(ukf);
  printf("  test_get_sqrt_cov_null_out OK\n");
}

static void test_get_sqrt_cov_dim_mismatch(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Wrong dimensions */
  srukf_mat *S_wrong = SRUKF_MAT_ALLOC(4, 4);
  assert(S_wrong);

  srukf_return rc = srukf_get_sqrt_cov(ukf, S_wrong);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(S_wrong);
  srukf_free(ukf);
  printf("  test_get_sqrt_cov_dim OK\n");
}

/* ========================= srukf_set_sqrt_cov ====================== */

static void test_set_sqrt_cov_valid(void) {
  srukf *ukf = create_test_filter(2, 1);
  assert(ukf);

  /* Create input S */
  srukf_mat *S_in = SRUKF_MAT_ALLOC(2, 2);
  assert(S_in);
  SRUKF_ENTRY(S_in, 0, 0) = 1.0;
  SRUKF_ENTRY(S_in, 0, 1) = 0.0;
  SRUKF_ENTRY(S_in, 1, 0) = 0.2;
  SRUKF_ENTRY(S_in, 1, 1) = 0.8;

  srukf_return rc = srukf_set_sqrt_cov(ukf, S_in);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify internal S was updated */
  assert(fabs(SRUKF_ENTRY(ukf->S, 0, 0) - 1.0) < EPS);
  assert(fabs(SRUKF_ENTRY(ukf->S, 1, 0) - 0.2) < EPS);
  assert(fabs(SRUKF_ENTRY(ukf->S, 1, 1) - 0.8) < EPS);

  srukf_mat_free(S_in);
  srukf_free(ukf);
  printf("  test_set_sqrt_cov_valid OK\n");
}

static void test_set_sqrt_cov_null_ukf(void) {
  srukf_mat *S_in = SRUKF_MAT_ALLOC(2, 2);
  assert(S_in);

  srukf_return rc = srukf_set_sqrt_cov(NULL, S_in);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(S_in);
  printf("  test_set_sqrt_cov_null_ukf OK\n");
}

static void test_set_sqrt_cov_null_in(void) {
  srukf *ukf = create_test_filter(2, 1);
  assert(ukf);

  srukf_return rc = srukf_set_sqrt_cov(ukf, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(ukf);
  printf("  test_set_sqrt_cov_null_in OK\n");
}

static void test_set_sqrt_cov_dim_mismatch(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Wrong dimensions */
  srukf_mat *S_wrong = SRUKF_MAT_ALLOC(2, 2);
  assert(S_wrong);

  srukf_return rc = srukf_set_sqrt_cov(ukf, S_wrong);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(S_wrong);
  srukf_free(ukf);
  printf("  test_set_sqrt_cov_dim OK\n");
}

/* ========================= srukf_reset ============================= */

static void test_reset_valid(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Set non-trivial state */
  SRUKF_ENTRY(ukf->x, 0, 0) = 100.0;
  SRUKF_ENTRY(ukf->x, 1, 0) = 200.0;
  SRUKF_ENTRY(ukf->x, 2, 0) = 300.0;

  srukf_return rc = srukf_reset(ukf, 0.5);
  assert(rc == SRUKF_RETURN_OK);

  /* State should be zeroed */
  for (int i = 0; i < 3; ++i)
    assert(fabs(SRUKF_ENTRY(ukf->x, i, 0)) < EPS);

  /* S should be diagonal with init_std on diagonal */
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j)
        assert(fabs(SRUKF_ENTRY(ukf->S, i, j) - 0.5) < EPS);
      else
        assert(fabs(SRUKF_ENTRY(ukf->S, i, j)) < EPS);
    }
  }

  srukf_free(ukf);
  printf("  test_reset_valid     OK\n");
}

static void test_reset_null_ukf(void) {
  srukf_return rc = srukf_reset(NULL, 1.0);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
  printf("  test_reset_null_ukf  OK\n");
}

static void test_reset_zero_std(void) {
  srukf *ukf = create_test_filter(2, 1);
  assert(ukf);

  srukf_return rc = srukf_reset(ukf, 0.0);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(ukf);
  printf("  test_reset_zero_std  OK\n");
}

static void test_reset_negative_std(void) {
  srukf *ukf = create_test_filter(2, 1);
  assert(ukf);

  srukf_return rc = srukf_reset(ukf, -1.0);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(ukf);
  printf("  test_reset_neg_std   OK\n");
}

/* ========================= Round-trip tests ======================== */

static void test_state_roundtrip(void) {
  srukf *ukf = create_test_filter(4, 2);
  assert(ukf);

  /* Set state */
  srukf_mat *x_in = SRUKF_MAT_ALLOC(4, 1);
  assert(x_in);
  for (int i = 0; i < 4; ++i)
    SRUKF_ENTRY(x_in, i, 0) = (double)(i * 10 + 1);

  srukf_return rc = srukf_set_state(ukf, x_in);
  assert(rc == SRUKF_RETURN_OK);

  /* Get state back */
  srukf_mat *x_out = SRUKF_MAT_ALLOC(4, 1);
  assert(x_out);

  rc = srukf_get_state(ukf, x_out);
  assert(rc == SRUKF_RETURN_OK);

  /* Should match */
  for (int i = 0; i < 4; ++i)
    assert(fabs(SRUKF_ENTRY(x_out, i, 0) - SRUKF_ENTRY(x_in, i, 0)) < EPS);

  srukf_mat_free(x_in);
  srukf_mat_free(x_out);
  srukf_free(ukf);
  printf("  test_state_roundtrip OK\n");
}

static void test_sqrt_cov_roundtrip(void) {
  srukf *ukf = create_test_filter(3, 2);
  assert(ukf);

  /* Set S */
  srukf_mat *S_in = SRUKF_MAT_ALLOC(3, 3);
  assert(S_in);
  /* Lower triangular */
  SRUKF_ENTRY(S_in, 0, 0) = 1.0;
  SRUKF_ENTRY(S_in, 1, 0) = 0.1;
  SRUKF_ENTRY(S_in, 1, 1) = 0.9;
  SRUKF_ENTRY(S_in, 2, 0) = 0.2;
  SRUKF_ENTRY(S_in, 2, 1) = 0.1;
  SRUKF_ENTRY(S_in, 2, 2) = 0.8;

  srukf_return rc = srukf_set_sqrt_cov(ukf, S_in);
  assert(rc == SRUKF_RETURN_OK);

  /* Get S back */
  srukf_mat *S_out = SRUKF_MAT_ALLOC(3, 3);
  assert(S_out);

  rc = srukf_get_sqrt_cov(ukf, S_out);
  assert(rc == SRUKF_RETURN_OK);

  /* Should match */
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      assert(fabs(SRUKF_ENTRY(S_out, i, j) - SRUKF_ENTRY(S_in, i, j)) < EPS);

  srukf_mat_free(S_in);
  srukf_mat_free(S_out);
  srukf_free(ukf);
  printf("  test_sqrt_cov_roundtrip OK\n");
}

int main(void) {
  printf("Running accessor function tests...\n");

  /* Dimension accessors */
  test_state_dim_valid();
  test_state_dim_null();
  test_meas_dim_valid();
  test_meas_dim_null();

  /* srukf_get_state */
  test_get_state_valid();
  test_get_state_null_ukf();
  test_get_state_null_out();
  test_get_state_dim_mismatch();

  /* srukf_set_state */
  test_set_state_valid();
  test_set_state_null_ukf();
  test_set_state_null_in();
  test_set_state_dim_mismatch();

  /* srukf_get_sqrt_cov */
  test_get_sqrt_cov_valid();
  test_get_sqrt_cov_null_ukf();
  test_get_sqrt_cov_null_out();
  test_get_sqrt_cov_dim_mismatch();

  /* srukf_set_sqrt_cov */
  test_set_sqrt_cov_valid();
  test_set_sqrt_cov_null_ukf();
  test_set_sqrt_cov_null_in();
  test_set_sqrt_cov_dim_mismatch();

  /* srukf_reset */
  test_reset_valid();
  test_reset_null_ukf();
  test_reset_zero_std();
  test_reset_negative_std();

  /* Round-trip */
  test_state_roundtrip();
  test_sqrt_cov_roundtrip();

  printf("accessor function tests passed.\n");
  return 0;
}
