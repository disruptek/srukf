/* --------------------------------------------------------------------
 * 40_stress.c - Stress tests for srukf
 *
 * Tests with larger dimensions to verify scalability:
 * - N=50 states, M=25 measurements
 * - Memory allocation succeeds
 * - Predict/correct complete without error
 * - Results are numerically valid
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

/* Identity measurement model (first M states) */
static void meas_identity(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; ++i)
    SRUKF_ENTRY(z, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Test 1: Large dimensions (N=50, M=25) */
static void test_large_dimensions(void) {
  const int N = 50, M = 25;

  srukf *ukf = srukf_create(N, M);
  assert(ukf != NULL);
  assert(ukf->x != NULL);
  assert(ukf->S != NULL);

  /* Initialize noise matrices */
  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);

  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 0.1;

  srukf_return ret = srukf_set_noise(ukf, Q, R);
  assert(ret == SRUKF_RETURN_OK);

  ret = srukf_set_scale(ukf, 1.0, 2.0, 0.0);
  assert(ret == SRUKF_RETURN_OK);

  /* Initialize state */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->x, i, 0) = 0.1 * i;

  /* Predict */
  ret = srukf_predict(ukf, process_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Verify state is valid */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  /* Correct */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(z, i, 0) = 0.2 * i;

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

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_large_dim (50x25) OK\n");
}

/* Test 2: Multiple cycles with large dimensions */
static void test_large_cycles(void) {
  const int N = 20, M = 10;
  const int CYCLES = 50;

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

  /* Run many cycles */
  for (int k = 0; k < CYCLES; ++k) {
    srukf_return ret = srukf_predict(ukf, process_identity, NULL);
    assert(ret == SRUKF_RETURN_OK);

    for (int i = 0; i < M; ++i)
      SRUKF_ENTRY(z, i, 0) = 1.0;

    ret = srukf_correct(ukf, z, meas_identity, NULL);
    assert(ret == SRUKF_RETURN_OK);
  }

  /* Verify final state is valid and reasonable */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  /* State should have converged toward measurement */
  for (int i = 0; i < M; ++i) {
    double x_i = SRUKF_ENTRY(ukf->x, i, 0);
    assert(fabs(x_i - 1.0) < 0.5); /* Should be close to 1.0 */
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_large_cycles    OK\n");
}

/* Test 3: Workspace pre-allocation */
static void test_workspace_prealloc(void) {
  const int N = 30, M = 15;

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

  /* Explicitly allocate workspace */
  srukf_return ret = srukf_alloc_workspace(ukf);
  assert(ret == SRUKF_RETURN_OK);
  assert(ukf->ws != NULL);

  /* Run predict/correct - should use pre-allocated workspace */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->x, i, 0) = 1.0;

  ret = srukf_predict(ukf, process_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(z, i, 0) = 1.0;

  ret = srukf_correct(ukf, z, meas_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Free workspace explicitly */
  srukf_free_workspace(ukf);
  assert(ukf->ws == NULL);

  /* Should still be able to use filter (workspace will be reallocated) */
  ret = srukf_predict(ukf, process_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_workspace_preal OK\n");
}

/* Test 4: Transactional API with large dimensions */
static void test_transactional_large(void) {
  const int N = 25, M = 12;

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

  /* User-managed buffers */
  srukf_mat *x = SRUKF_MAT_ALLOC(N, 1);
  srukf_mat *S = SRUKF_MAT_ALLOC(N, N);
  assert(x && S);

  for (int i = 0; i < N; ++i) {
    SRUKF_ENTRY(x, i, 0) = 0.5;
    SRUKF_ENTRY(S, i, i) = 0.01;
  }

  /* Transactional predict */
  srukf_return ret = srukf_predict_to(ukf, x, S, process_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Transactional correct */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(z, i, 0) = 1.0;

  ret = srukf_correct_to(ukf, x, S, z, meas_identity, NULL);
  assert(ret == SRUKF_RETURN_OK);

  /* Verify results are valid */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(x, i, 0)));
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(x);
  srukf_mat_free(S);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_transact_large  OK\n");
}

int main(void) {
  printf("Running stress tests...\n");

  test_large_dimensions();
  test_large_cycles();
  test_workspace_prealloc();
  test_transactional_large();

  printf("stress tests passed.\n");
  return 0;
}
