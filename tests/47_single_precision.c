/* --------------------------------------------------------------------
 * 47_single_precision.c - Tests for single-precision (float) mode
 *
 * This file is compiled with -DSRUKF_SINGLE to test float precision.
 * Tests verify that the filter works correctly with reduced precision.
 *
 * This test includes srukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#ifndef SRUKF_SINGLE
#define SRUKF_SINGLE /* Enable single precision */
#endif

#include "srukf.c"
#include "tests/test_helpers.h"

/* Looser tolerance for single precision */
#define FLOAT_EPS 1e-4

/* Identity process model */
static void process_identity(const srukf_mat *x, srukf_mat *xp, void *user) {
  (void)user;
  for (srukf_index i = 0; i < x->n_rows; ++i)
    SRUKF_ENTRY(xp, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Constant velocity model */
static void process_const_vel(const srukf_mat *x, srukf_mat *xp, void *user) {
  srukf_value dt = *(srukf_value *)user;
  SRUKF_ENTRY(xp, 0, 0) = SRUKF_ENTRY(x, 0, 0) + dt * SRUKF_ENTRY(x, 1, 0);
  SRUKF_ENTRY(xp, 1, 0) = SRUKF_ENTRY(x, 1, 0);
}

/* Identity measurement model */
static void meas_identity(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; ++i)
    SRUKF_ENTRY(z, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Helper: create filter with noise.
 * Uses alpha=1.0 for better numerical stability in single precision.
 * The default alpha=1e-3 creates very negative wc[0] which causes
 * downdate failures in float arithmetic. */
static srukf *create_test_filter(int N, int M, srukf_value q, srukf_value r) {
  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);

  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = q;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = r;

  srukf *ukf = srukf_create_from_noise(Q, R);
  srukf_mat_free(Q);
  srukf_mat_free(R);

  if (ukf) {
    /* Use alpha=1.0 for better numerical stability in single precision */
    srukf_set_scale(ukf, 1.0f, 2.0f, 0.0f);
  }

  return ukf;
}

/* ========================= Basic functionality ===================== */

static void test_type_size(void) {
  /* Verify we're actually using float */
  assert(sizeof(srukf_value) == sizeof(float));
  printf("  test_type_size (float) OK\n");
}

static void test_basic_create(void) {
  srukf *ukf = create_test_filter(3, 2, 0.1f, 0.1f);
  assert(ukf != NULL);
  assert(srukf_state_dim(ukf) == 3);
  assert(srukf_meas_dim(ukf) == 2);
  srukf_free(ukf);
  printf("  test_basic_create    OK\n");
}

static void test_basic_predict(void) {
  srukf *ukf = create_test_filter(2, 1, 0.1f, 0.1f);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 1.0f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 2.0f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 0.5f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 0.5f;

  srukf_return rc = srukf_predict(ukf, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should be near original for identity process */
  assert(fabsf(SRUKF_ENTRY(ukf->x, 0, 0) - 1.0f) < 0.1f);
  assert(fabsf(SRUKF_ENTRY(ukf->x, 1, 0) - 2.0f) < 0.1f);

  srukf_free(ukf);
  printf("  test_basic_predict   OK\n");
}

static void test_basic_correct(void) {
  srukf *ukf = create_test_filter(2, 2, 0.1f, 0.1f);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0f;

  srukf_mat *z = SRUKF_MAT_ALLOC(2, 1);
  SRUKF_ENTRY(z, 0, 0) = 5.0f;
  SRUKF_ENTRY(z, 1, 0) = 10.0f;

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should move toward measurement */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) > 0.0f);
  assert(SRUKF_ENTRY(ukf->x, 1, 0) > 0.0f);

  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_basic_correct   OK\n");
}

/* ========================= Numerical precision ===================== */

static void test_precision_accumulation(void) {
  /* Test that errors don't accumulate too badly over many cycles. */
  srukf *ukf = create_test_filter(2, 2, 0.1f, 0.1f);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 0.0f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1.0f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1.0f;

  srukf_mat *z = SRUKF_MAT_ALLOC(2, 1);
  SRUKF_ENTRY(z, 0, 0) = 1.0f;
  SRUKF_ENTRY(z, 1, 0) = 2.0f;

  /* Run 100 cycles */
  for (int k = 0; k < 100; ++k) {
    srukf_return rc = srukf_predict(ukf, process_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);

    rc = srukf_correct(ukf, z, meas_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);

    /* Verify values are finite */
    for (int i = 0; i < 2; ++i) {
      assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
      for (int j = 0; j < 2; ++j) {
        assert(isfinite(SRUKF_ENTRY(ukf->S, i, j)));
      }
    }
  }

  /* State should have converged */
  assert(fabsf(SRUKF_ENTRY(ukf->x, 0, 0) - 1.0f) < 0.5f);
  assert(fabsf(SRUKF_ENTRY(ukf->x, 1, 0) - 2.0f) < 0.5f);

  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_precision_accum OK\n");
}

static void test_small_values(void) {
  /* Test with small values that might lose precision in float */
  srukf *ukf = create_test_filter(2, 2, 1e-4f, 1e-4f);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 1e-5f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 1e-5f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1e-3f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1e-3f;

  srukf_return rc = srukf_predict(ukf, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Values should still be finite */
  assert(isfinite(SRUKF_ENTRY(ukf->x, 0, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 1, 0)));

  srukf_free(ukf);
  printf("  test_small_values    OK\n");
}

static void test_large_values(void) {
  /* Test with large values */
  srukf *ukf = create_test_filter(2, 2, 100.0f, 100.0f);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 1e5f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 1e5f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1e3f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 1e3f;

  srukf_return rc = srukf_predict(ukf, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  assert(isfinite(SRUKF_ENTRY(ukf->x, 0, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 1, 0)));

  srukf_free(ukf);
  printf("  test_large_values    OK\n");
}

/* ========================= Convergence ============================= */

static void test_convergence(void) {
  /* Test that filter converges to correct value */
  srukf *ukf = create_test_filter(2, 2, 0.1f, 0.1f);
  assert(ukf);

  /* Start far from true state */
  SRUKF_ENTRY(ukf->x, 0, 0) = 100.0f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 100.0f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 10.0f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 10.0f;

  srukf_mat *z = SRUKF_MAT_ALLOC(2, 1);
  SRUKF_ENTRY(z, 0, 0) = 5.0f;
  SRUKF_ENTRY(z, 1, 0) = 10.0f;

  /* Run until convergence */
  float prev_error = 1e10f;
  for (int k = 0; k < 50; ++k) {
    srukf_predict(ukf, process_identity, NULL);
    srukf_correct(ukf, z, meas_identity, NULL);

    float err0 = SRUKF_ENTRY(ukf->x, 0, 0) - 5.0f;
    float err1 = SRUKF_ENTRY(ukf->x, 1, 0) - 10.0f;
    float error = sqrtf(err0 * err0 + err1 * err1);

    /* Error should generally decrease */
    if (k > 5) {
      assert(error <= prev_error + 1.0f); /* Allow some fluctuation */
    }
    prev_error = error;
  }

  /* Should be close to true value */
  assert(fabsf(SRUKF_ENTRY(ukf->x, 0, 0) - 5.0f) < 1.0f);
  assert(fabsf(SRUKF_ENTRY(ukf->x, 1, 0) - 10.0f) < 1.0f);

  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_convergence     OK\n");
}

/* ========================= Velocity model ========================== */

static void test_velocity_model(void) {
  srukf *ukf = create_test_filter(2, 1, 0.01f, 0.1f);
  assert(ukf);

  /* State: [position, velocity] = [0, 1] */
  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 1.0f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 0.1f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 0.1f;

  srukf_value dt = 0.1f;

  /* Predict 10 steps */
  for (int k = 0; k < 10; ++k) {
    srukf_return rc = srukf_predict(ukf, process_const_vel, &dt);
    assert(rc == SRUKF_RETURN_OK);
  }

  /* After 10 steps with dt=0.1, vel=1, position should be ~1.0 */
  assert(fabsf(SRUKF_ENTRY(ukf->x, 0, 0) - 1.0f) < 0.5f);
  assert(fabsf(SRUKF_ENTRY(ukf->x, 1, 0) - 1.0f) < 0.5f);

  srukf_free(ukf);
  printf("  test_velocity_model  OK\n");
}

/* ========================= Higher dimensions ======================= */

static void test_higher_dims(void) {
  const int N = 10, M = 5;
  srukf *ukf = create_test_filter(N, M, 0.1f, 0.1f);
  assert(ukf);

  for (int i = 0; i < N; ++i) {
    SRUKF_ENTRY(ukf->x, i, 0) = (float)i;
    SRUKF_ENTRY(ukf->S, i, i) = 1.0f;
  }

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(z, i, 0) = 10.0f;

  srukf_return rc = srukf_predict(ukf, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* All values should be finite */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_higher_dims     OK\n");
}

/* ========================= Transactional API ======================= */

static void test_predict_to(void) {
  srukf *ukf = create_test_filter(2, 1, 0.1f, 0.1f);
  assert(ukf);

  SRUKF_ENTRY(ukf->x, 0, 0) = 1.0f;
  SRUKF_ENTRY(ukf->x, 1, 0) = 2.0f;
  SRUKF_ENTRY(ukf->S, 0, 0) = 0.5f;
  SRUKF_ENTRY(ukf->S, 1, 1) = 0.5f;

  float x0_orig = SRUKF_ENTRY(ukf->x, 0, 0);

  srukf_mat *x = SRUKF_MAT_ALLOC(2, 1);
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  memcpy(x->data, ukf->x->data, 2 * sizeof(srukf_value));
  memcpy(S->data, ukf->S->data, 4 * sizeof(srukf_value));

  srukf_return rc = srukf_predict_to(ukf, x, S, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Original unchanged */
  assert(SRUKF_ENTRY(ukf->x, 0, 0) == x0_orig);

  srukf_mat_free(x);
  srukf_mat_free(S);
  srukf_free(ukf);
  printf("  test_predict_to      OK\n");
}

int main(void) {
  printf("Running single-precision (float) tests...\n");

  test_type_size();
  test_basic_create();
  test_basic_predict();
  test_basic_correct();
  test_precision_accumulation();
  test_small_values();
  test_large_values();
  test_convergence();
  test_velocity_model();
  test_higher_dims();
  test_predict_to();

  printf("single-precision tests passed.\n");
  return 0;
}
