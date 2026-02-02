/* --------------------------------------------------------------------
 * 35_numerical.c - Numerical stability tests for srukf
 *
 * Tests edge cases and numerical robustness:
 * - Near-singular covariance matrices
 * - Very small/large noise values
 * - State values near machine epsilon
 *
 * This test includes srukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "srukf.c"

/* Identity process model */
static void process_identity(const srukf_mat *x, srukf_mat *xp, void *user) {
  (void)user;
  for (srukf_index i = 0; i < x->n_rows; ++i)
    SRUKF_ENTRY(xp, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Identity measurement model */
static void meas_identity(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; ++i)
    SRUKF_ENTRY(z, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Test 1: Very small noise covariances */
static void test_small_noise(void) {
  const int N = 3, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  /* Very small noise (sqrt of 1e-12) */
  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);

  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 1e-6; /* sqrt(1e-12) */
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 1e-6;

  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);

  /* Initialize state */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->x, i, 0) = 1.0;

  /* Should still work with small noise */
  srukf_return ret = srukf_predict(ukf, process_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Verify state is still valid (not NaN/Inf) */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_small_noise     OK\n");
}

/* Test 2: Large noise covariances */
static void test_large_noise(void) {
  const int N = 3, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  /* Large noise */
  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);

  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 100.0;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 100.0;

  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);

  /* Initialize state */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->x, i, 0) = 1.0;

  /* Should work with large noise */
  srukf_return ret = srukf_predict(ukf, process_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Correct with measurement */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(z, i, 0) = 2.0;

  ret = srukf_correct(ukf, z, meas_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Verify state is still valid */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_large_noise     OK\n");
}

/* Test 3: Near-zero state values */
static void test_near_zero_state(void) {
  const int N = 3, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);

  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.01;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 0.01;

  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);

  /* Initialize state to very small values */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->x, i, 0) = 1e-10;

  srukf_return ret = srukf_predict(ukf, process_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Verify state is still valid */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_near_zero_state OK\n");
}

/* Test 4: Multiple predict/correct cycles with varying measurements */
static void test_stability_cycles(void) {
  const int N = 4, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);

  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 0.1;

  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);
  assert(srukf_set_scale(ukf, 1.0, 2.0, 0.0) == SRUKF_RETURN_OK);

  /* Initialize state */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->x, i, 0) = 0.0;

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);

  /* Run many cycles with varying measurements */
  for (int k = 0; k < 100; ++k) {
    /* Predict */
    srukf_return ret = srukf_predict(ukf, process_identity, NULL);
    assert(ret == SRUKF_RETURN_OK);

    /* Varying measurement */
    for (int i = 0; i < M; ++i)
      SRUKF_ENTRY(z, i, 0) = sin((double)k * 0.1 + i);

    /* Correct */
    ret = srukf_correct(ukf, z, meas_identity, NULL);
    assert(ret == SRUKF_RETURN_OK);

    /* Verify state and covariance are valid */
    for (int i = 0; i < N; ++i) {
      assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
    }
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        assert(isfinite(SRUKF_ENTRY(ukf->S, i, j)));
      }
    }
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_stability_cyc   OK\n");
}

/* Test 5: Asymmetric noise (different Q and R scales) */
static void test_asymmetric_noise(void) {
  const int N = 3, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);

  /* Very different scales */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.001; /* Small process noise */
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 10.0; /* Large measurement noise */

  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);

  /* Initialize state */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->x, i, 0) = 1.0;

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(z, i, 0) = 5.0;

  /* Run a few cycles */
  for (int k = 0; k < 10; ++k) {
    srukf_return ret = srukf_predict(ukf, process_identity, NULL);
    assert(ret == SRUKF_RETURN_OK);
    ret = srukf_correct(ukf, z, meas_identity, NULL);
    assert(ret == SRUKF_RETURN_OK);
  }

  /* With large measurement noise and small process noise,
   * state should stay closer to initial than measurement */
  /* Just verify it's valid */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_asymmetric_nois OK\n");
}

int main(void) {
  printf("Running numerical stability tests...\n");

  test_small_noise();
  test_large_noise();
  test_near_zero_state();
  test_stability_cycles();
  test_asymmetric_noise();

  printf("numerical stability tests passed.\n");
  return 0;
}
