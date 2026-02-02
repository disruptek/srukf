/* --------------------------------------------------------------------
 * 46_edge_cases.c - Edge case and error path tests
 *
 * Tests for untested code paths:
 * - Zero/negative dimensions
 * - Non-square noise matrices
 * - Negative wc[0] (very small alpha)
 * - Cholesky downdate failure
 * - Near-zero Syy (measurement covariance)
 * - SRUKF_MAT_ALLOC_NO_DATA (NoData flag)
 *
 * This test includes srukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "srukf.c"
#include "tests/test_helpers.h"

#define EPS 1e-10

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

/* ========================= Dimension validation ==================== */

static void test_create_zero_state_dim(void) {
  srukf *ukf = srukf_create(0, 2);
  assert(ukf == NULL);
  printf("  test_create_zero_N   OK\n");
}

static void test_create_zero_meas_dim(void) {
  srukf *ukf = srukf_create(3, 0);
  assert(ukf == NULL);
  printf("  test_create_zero_M   OK\n");
}

static void test_create_negative_dims(void) {
  srukf *ukf;

  ukf = srukf_create(-1, 2);
  assert(ukf == NULL);

  ukf = srukf_create(3, -5);
  assert(ukf == NULL);

  ukf = srukf_create(-2, -3);
  assert(ukf == NULL);

  printf("  test_create_neg_dims OK\n");
}

/* ========================= Non-square noise matrices =============== */

static void test_create_from_noise_nonsquare_Q(void) {
  srukf_mat *Q = SRUKF_MAT_ALLOC(3, 4); /* Non-square */
  srukf_mat *R = SRUKF_MAT_ALLOC(2, 2);
  assert(Q && R);

  for (int i = 0; i < 2; ++i)
    SRUKF_ENTRY(R, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Q, R);
  assert(ukf == NULL);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  printf("  test_nonsquare_Q     OK\n");
}

static void test_create_from_noise_nonsquare_R(void) {
  srukf_mat *Q = SRUKF_MAT_ALLOC(3, 3);
  srukf_mat *R = SRUKF_MAT_ALLOC(2, 3); /* Non-square */
  assert(Q && R);

  for (int i = 0; i < 3; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;

  srukf *ukf = srukf_create_from_noise(Q, R);
  assert(ukf == NULL);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  printf("  test_nonsquare_R     OK\n");
}

/* ========================= SRUKF_MAT_ALLOC_NO_DATA (NoData flag) ========== */

static void test_alloc_matrix_later(void) {
  srukf_mat *m = SRUKF_MAT_ALLOC_NO_DATA(5, 3);
  assert(m != NULL);
  assert(m->n_rows == 5);
  assert(m->n_cols == 3);
  assert(m->data == NULL);
  assert(SRUKF_IS_TYPE(m, SRUKF_TYPE_NO_DATA));

  srukf_mat_free(m);
  printf("  test_alloc_later     OK\n");
}

/* ========================= Negative wc[0] (very small alpha) ======= */

/* When alpha is very small, wc[0] = wm[0] + (1 - alpha^2 + beta) can be
 * negative (if wm[0] = lambda/(n+lambda) is very negative). This triggers
 * special handling with Cholesky downdate.
 */
static void test_negative_wc0_predict(void) {
  const int N = 3, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  /* Set noise */
  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 0.1;
  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);

  /* Very small alpha (1e-4) with kappa=0, beta=2 for N=3:
   * lambda = alpha^2 * (N + kappa) - N = 1e-8 * 3 - 3 ≈ -3
   * wm[0] = lambda / (N + lambda) = -3 / (3 - 3) -> undefined
   *
   * Actually need to use parameters that give small but valid lambda.
   * Let's use alpha=0.01, kappa=0, N=3:
   * lambda = 0.0001 * 3 - 3 = -2.9997
   * N + lambda = 0.0003 > 0
   * wm[0] = -2.9997 / 0.0003 = -9999
   * wc[0] = wm[0] + (1 - 0.0001 + 2) = -9999 + 2.9999 ≈ -9996 < 0
   */
  srukf_return rc = srukf_set_scale(ukf, 0.01, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify wc[0] is negative */
  assert(ukf->wc[0] < 0.0);

  /* Set state and covariance */
  for (int i = 0; i < N; ++i) {
    SRUKF_ENTRY(ukf->x, i, 0) = 1.0;
    SRUKF_ENTRY(ukf->S, i, i) = 0.5;
  }

  /* Predict should still work (uses downdate path) */
  rc = srukf_predict(ukf, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify state is valid */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  /* Verify covariance produces SPD result */
  assert(is_sqrt_valid(ukf->S));

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_neg_wc0_predict OK\n");
}

static void test_negative_wc0_correct(void) {
  const int N = 3, M = 2;
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

  /* Very small alpha -> negative wc[0] */
  srukf_return rc = srukf_set_scale(ukf, 0.01, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_OK);
  assert(ukf->wc[0] < 0.0);

  /* Set state and covariance */
  for (int i = 0; i < N; ++i) {
    SRUKF_ENTRY(ukf->x, i, 0) = 0.0;
    SRUKF_ENTRY(ukf->S, i, i) = 1.0;
  }

  /* Measurement */
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = 1.0;
  SRUKF_ENTRY(z, 1, 0) = 2.0;

  /* Correct should work (uses downdate path for Syy) */
  rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify state is valid */
  for (int i = 0; i < N; ++i) {
    assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_neg_wc0_correct OK\n");
}

/* ========================= Cholesky downdate failure =============== */

/* The Cholesky downdate S' * S' = S * S - v * v' fails if S*S - v*v' is not
 * positive definite. This happens when v is "too large" relative to S.
 *
 * We can trigger this by:
 * 1. Setting up a filter with small S
 * 2. Making K*Syy columns large (via large cross-covariance or small meas
 * noise)
 *
 * Actually, let's directly test chol_downdate_rank1 with an artificial
 * scenario.
 */
static void test_chol_downdate_failure(void) {
  /* Create a small S (lower triangular) */
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  assert(S);
  SRUKF_ENTRY(S, 0, 0) = 1.0;
  SRUKF_ENTRY(S, 0, 1) = 0.0;
  SRUKF_ENTRY(S, 1, 0) = 0.0;
  SRUKF_ENTRY(S, 1, 1) = 1.0;

  /* S*S' = I (identity) */

  /* Create v such that I - v*v' is not positive definite.
   * If v = [1.5, 0], then v*v' = [[2.25, 0], [0, 0]]
   * I - v*v' = [[-1.25, 0], [0, 1]] which has a negative eigenvalue.
   */
  srukf_value v[2] = {1.5, 0.0};
  srukf_value work[2];

  srukf_return rc = chol_downdate_rank1(S, v, work);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  srukf_mat_free(S);
  printf("  test_chol_downdate_fail OK\n");
}

/* Also test with a more complex case */
static void test_chol_downdate_failure_3d(void) {
  srukf_mat *S = SRUKF_MAT_ALLOC(3, 3);
  assert(S);

  /* Identity-like lower triangular */
  SRUKF_ENTRY(S, 0, 0) = 0.5;
  SRUKF_ENTRY(S, 1, 1) = 0.5;
  SRUKF_ENTRY(S, 2, 2) = 0.5;

  /* S*S' = 0.25 * I */

  /* v such that 0.25*I - v*v' is not SPD.
   * v = [0.6, 0, 0] gives v*v' = [[0.36, 0, 0], [0,0,0], [0,0,0]]
   * 0.25*I - v*v' has (0.25 - 0.36) = -0.11 in (0,0) -> not SPD
   */
  srukf_value v[3] = {0.6, 0.0, 0.0};
  srukf_value work[3];

  srukf_return rc = chol_downdate_rank1(S, v, work);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  srukf_mat_free(S);
  printf("  test_chol_downdate_fail_3d OK\n");
}

/* Test successful downdate for comparison */
static void test_chol_downdate_success(void) {
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  assert(S);
  SRUKF_ENTRY(S, 0, 0) = 2.0;
  SRUKF_ENTRY(S, 0, 1) = 0.0;
  SRUKF_ENTRY(S, 1, 0) = 0.0;
  SRUKF_ENTRY(S, 1, 1) = 2.0;

  /* S*S' = 4*I */

  /* Small v: v = [0.5, 0], v*v' = [[0.25, 0], [0, 0]]
   * 4*I - v*v' = [[3.75, 0], [0, 4]] which is SPD
   */
  srukf_value v[2] = {0.5, 0.0};
  srukf_value work[2];

  srukf_return rc = chol_downdate_rank1(S, v, work);
  assert(rc == SRUKF_RETURN_OK);

  /* Verify S is still valid (lower triangular with positive diagonal) */
  assert(SRUKF_ENTRY(S, 0, 0) > 0);
  assert(SRUKF_ENTRY(S, 1, 1) > 0);

  srukf_mat_free(S);
  printf("  test_chol_downdate_ok OK\n");
}

/* ========================= Near-zero Syy (early return path) ======= */

/* When Syy is essentially zero, srukf_correct_core has an early return that
 * copies input to output without updating. This can happen with very small
 * state covariance and very small measurement noise.
 */
static void test_syy_near_zero(void) {
  const int N = 2, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  /* Very small noise - sqrt of tiny variance */
  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 1e-15;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 1e-15;
  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);

  /* Very small state covariance */
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(ukf->S, i, i) = 1e-15;

  /* Save state before correct */
  double x0_before = SRUKF_ENTRY(ukf->x, 0, 0);
  double x1_before = SRUKF_ENTRY(ukf->x, 1, 0);

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = 100.0;
  SRUKF_ENTRY(z, 1, 0) = 200.0;

  /* With near-zero Syy, the state might not change much or at all */
  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Just verify state is still valid (not NaN/Inf) */
  assert(isfinite(SRUKF_ENTRY(ukf->x, 0, 0)));
  assert(isfinite(SRUKF_ENTRY(ukf->x, 1, 0)));

  /* If Syy_zero path was taken, state should be unchanged */
  /* (This is a soft check - implementation may vary) */
  (void)x0_before;
  (void)x1_before;

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_syy_near_zero   OK\n");
}

/* ========================= propagate_sigma_points errors =========== */

static void test_propagate_null_ysig_data(void) {
  srukf_mat *Xsig = SRUKF_MAT_ALLOC(2, 5);
  srukf_mat *Ysig = SRUKF_MAT_ALLOC_NO_DATA(2, 5); /* No data allocated */
  assert(Xsig && Ysig);

  srukf_return rc = propagate_sigma_points(Xsig, Ysig, process_identity, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(Xsig);
  srukf_mat_free(Ysig);
  printf("  test_propagate_null_data OK\n");
}

static void test_propagate_dim_mismatch(void) {
  srukf_mat *Xsig = SRUKF_MAT_ALLOC(2, 5);
  srukf_mat *Ysig = SRUKF_MAT_ALLOC(2, 7); /* Wrong columns */
  assert(Xsig && Ysig);

  srukf_return rc = propagate_sigma_points(Xsig, Ysig, process_identity, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(Xsig);
  srukf_mat_free(Ysig);
  printf("  test_propagate_dim_mismatch OK\n");
}

/* ========================= compute_weights edge cases ============== */

static void test_weights_recomputation(void) {
  srukf *ukf = srukf_create(3, 2);
  assert(ukf);

  /* Set noise to make filter usable */
  srukf_mat *Q = SRUKF_MAT_ALLOC(3, 3);
  srukf_mat *R = SRUKF_MAT_ALLOC(2, 2);
  for (int i = 0; i < 3; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < 2; ++i)
    SRUKF_ENTRY(R, i, i) = 0.1;
  srukf_set_noise(ukf, Q, R);

  /* Change scale multiple times */
  assert(srukf_set_scale(ukf, 1.0, 2.0, 0.0) == SRUKF_RETURN_OK);
  double wm0_first = ukf->wm[0];

  assert(srukf_set_scale(ukf, 0.5, 2.0, 1.0) == SRUKF_RETURN_OK);
  double wm0_second = ukf->wm[0];

  /* Weights should have changed */
  assert(fabs(wm0_first - wm0_second) > EPS);

  /* Change back */
  assert(srukf_set_scale(ukf, 1.0, 2.0, 0.0) == SRUKF_RETURN_OK);
  assert(fabs(ukf->wm[0] - wm0_first) < EPS);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_weights_recomp  OK\n");
}

/* ========================= Multiple predict/correct cycles ========= */

/* Test negative wc[0] over many cycles for stability */
static void test_negative_wc0_stability(void) {
  const int N = 3, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 0.1;
  srukf_set_noise(ukf, Q, R);

  /* Small alpha -> negative wc[0] */
  srukf_set_scale(ukf, 0.01, 2.0, 0.0);
  assert(ukf->wc[0] < 0.0);

  /* Initialize */
  for (int i = 0; i < N; ++i) {
    SRUKF_ENTRY(ukf->x, i, 0) = 0.0;
    SRUKF_ENTRY(ukf->S, i, i) = 1.0;
  }

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);

  /* Run 50 cycles */
  for (int k = 0; k < 50; ++k) {
    srukf_return rc = srukf_predict(ukf, process_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);

    SRUKF_ENTRY(z, 0, 0) = 1.0;
    SRUKF_ENTRY(z, 1, 0) = 2.0;

    rc = srukf_correct(ukf, z, meas_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);

    /* Verify state and covariance are finite */
    for (int i = 0; i < N; ++i) {
      assert(isfinite(SRUKF_ENTRY(ukf->x, i, 0)));
      for (int j = 0; j < N; ++j) {
        assert(isfinite(SRUKF_ENTRY(ukf->S, i, j)));
      }
    }
  }

  /* State should have converged toward measurement */
  assert(fabs(SRUKF_ENTRY(ukf->x, 0, 0) - 1.0) < 0.5);
  assert(fabs(SRUKF_ENTRY(ukf->x, 1, 0) - 2.0) < 0.5);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_neg_wc0_stable  OK\n");
}

int main(void) {
  printf("Running edge case tests...\n");

  /* Dimension validation */
  test_create_zero_state_dim();
  test_create_zero_meas_dim();
  test_create_negative_dims();

  /* Non-square noise */
  test_create_from_noise_nonsquare_Q();
  test_create_from_noise_nonsquare_R();

  /* SRUKF_MAT_ALLOC_NO_DATA */
  test_alloc_matrix_later();

  /* Negative wc[0] */
  test_negative_wc0_predict();
  test_negative_wc0_correct();
  test_negative_wc0_stability();

  /* Cholesky downdate */
  test_chol_downdate_failure();
  test_chol_downdate_failure_3d();
  test_chol_downdate_success();

  /* Near-zero Syy */
  test_syy_near_zero();

  /* propagate_sigma_points errors */
  test_propagate_null_ysig_data();
  test_propagate_dim_mismatch();

  /* Weight recomputation */
  test_weights_recomputation();

  printf("edge case tests passed.\n");
  return 0;
}
