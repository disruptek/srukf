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

/* ========================= srukf_mat_alloc hardening ================ */

static void test_mat_alloc_zero_dims(void) {
  assert(srukf_mat_alloc(0, 5, 1) == NULL);
  assert(srukf_mat_alloc(5, 0, 1) == NULL);
  assert(srukf_mat_alloc(0, 0, 1) == NULL);
  assert(srukf_mat_alloc(0, 5, 0) == NULL); /* no-data descriptors too */
  printf("  test_mat_alloc_zero  OK\n");
}

static void test_mat_alloc_overflow(void) {
  /* rows * cols wraps the size_t multiplication to a tiny value; the
   * result would be a descriptor claiming 2^61 rows over a (nearly)
   * empty buffer.  Must be rejected before calloc ever sees it. */
  srukf_index huge = ((srukf_index)1) << 61;
  assert(srukf_mat_alloc(huge, 8, 1) == NULL);
  assert(srukf_mat_alloc(8, huge, 1) == NULL);
  printf("  test_mat_alloc_ovf   OK\n");
}

/* ========================= set_noise in-place update ================ */

/* Once buffers exist, set_noise must update them in place: no
 * reallocation (the data pointers are stable) and exact value copy.
 * The first call after srukf_create() allocates the buffers. */
static void test_set_noise_in_place(void) {
  const int N = 2, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);
  assert(ukf->Qsqrt->data == NULL); /* created without noise buffers */

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.5;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 0.25;

  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);
  srukf_value *qdata = ukf->Qsqrt->data;
  srukf_value *rdata = ukf->Rsqrt->data;
  assert(qdata && rdata);
  assert(SRUKF_ENTRY(ukf->Qsqrt, 0, 0) == 0.5);
  assert(SRUKF_ENTRY(ukf->Rsqrt, 1, 1) == 0.25);

  /* Second call: same buffers, new values. */
  SRUKF_ENTRY(Q, 0, 0) = 0.7;
  SRUKF_ENTRY(Q, 1, 0) = 0.1;
  SRUKF_ENTRY(R, 1, 1) = 0.9;
  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);
  assert(ukf->Qsqrt->data == qdata);
  assert(ukf->Rsqrt->data == rdata);
  assert(SRUKF_ENTRY(ukf->Qsqrt, 0, 0) == 0.7);
  assert(SRUKF_ENTRY(ukf->Qsqrt, 1, 0) == 0.1);
  assert(SRUKF_ENTRY(ukf->Rsqrt, 1, 1) == 0.9);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_set_noise_inplace OK\n");
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

/* Exact cancellation: v removes ALL variance along the first axis while
 * correlation with the second axis remains.  P - v*v' is indefinite
 * (det < 0), so the downdate must fail; the historical bug was a
 * division by zero here (c = r/Sjj with r = 0) that wrote -inf into S
 * and returned OK. */
static void test_chol_downdate_exact_cancellation(void) {
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  assert(S);
  SRUKF_ENTRY(S, 0, 0) = 1.0;
  SRUKF_ENTRY(S, 1, 0) = 0.5;
  SRUKF_ENTRY(S, 1, 1) = 1.0;

  /* P = S*S' = [[1, 0.5], [0.5, 1.25]];  v*v' = [[1, 0.6], [0.6, 0.36]]
   * P - v*v' = [[0, -0.1], [-0.1, 0.89]] -> indefinite (det = -0.01) */
  srukf_value v[2] = {1.0, 0.6};
  srukf_value work[2];

  srukf_return rc = chol_downdate_rank1(S, v, work);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  srukf_mat_free(S);
  printf("  test_chol_downdate_exact_cancel OK\n");
}

/* Near cancellation: r2 = Sjj^2 - wj^2 is positive but far below the
 * working precision relative to Sjj^2.  Proceeding would amplify the
 * remaining column by 1/c ~ 1/sqrt(r2) -- silently garbage, so it must
 * be rejected just like the indefinite case. */
static void test_chol_downdate_near_cancellation(void) {
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  assert(S);
  SRUKF_ENTRY(S, 0, 0) = 1.0;
  SRUKF_ENTRY(S, 1, 0) = 0.5;
  SRUKF_ENTRY(S, 1, 1) = 1.0;

  /* r2 = 1 - (1 - 5e-14)^2 ~ 1e-13: positive, but c ~ 3e-7 would
   * scale S(1,0) from -0.1 to ~ -3e5. */
  srukf_value v[2] = {1.0 - 5e-14, 0.6};
  srukf_value work[2];

  /* (In single precision 1 - 5e-14 rounds to 1.0, collapsing onto the
   * exact-cancellation case -- the expectation is the same.) */
  srukf_return rc = chol_downdate_rank1(S, v, work);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  srukf_mat_free(S);
  printf("  test_chol_downdate_near_cancel OK\n");
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

/* ========================= Singular / tiny Syy ===================== */

/* A filter legitimately operating at tiny absolute scales must NOT have
 * its measurements discarded: the Kalman gain is scale-invariant in Syy,
 * so a well-conditioned Syy of magnitude 1e-15 is perfectly usable.
 * (Historically an ABSOLUTE Syy < eps test silently skipped the update
 * and still returned OK.) */
static void test_tiny_scale_update(void) {
  srukf *ukf = srukf_create(1, 1);
  assert(ukf);

  srukf_mat *Q = SRUKF_MAT_ALLOC(1, 1);
  srukf_mat *R = SRUKF_MAT_ALLOC(1, 1);
  assert(Q && R);
  SRUKF_ENTRY(Q, 0, 0) = 1e-16;
  SRUKF_ENTRY(R, 0, 0) = 1e-15;
  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);
  /* alpha = 1 for clean weights: gamma = 1, pyy = p + r^2 */
  assert(srukf_set_scale(ukf, 1.0, 2.0, 0.0) == SRUKF_RETURN_OK);

  SRUKF_ENTRY(ukf->x, 0, 0) = 0.0;
  SRUKF_ENTRY(ukf->S, 0, 0) = 1e-15;

  /* Innovation of 1e-13 at prior variance p = 1e-30, r^2 = 1e-30:
   * K = p / (p + r^2) = 0.5, so the update must move x to ~ 0.5e-13. */
  srukf_mat *z = SRUKF_MAT_ALLOC(1, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = 1e-13;

  srukf_return rc = srukf_correct(ukf, z, meas_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  double x_post = SRUKF_ENTRY(ukf->x, 0, 0);
  assert(x_post > 0.4e-13 && x_post < 0.6e-13);
  assert(isfinite(SRUKF_ENTRY(ukf->S, 0, 0)));

  /* The innovation must be available: the measurement was incorporated. */
  srukf_value nis = -1.0;
  assert(srukf_get_nis(ukf, &nis) == SRUKF_RETURN_OK);
  assert(isfinite(nis) && nis >= 0.0);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_tiny_scale_update OK\n");
}

/* Measurement model with no spread at all: h(x) = const.  Combined with
 * a zero Rsqrt, Syy is exactly singular -- the Kalman gain is undefined
 * and correct must FAIL (leaving the state untouched) rather than
 * silently ignore the measurement and report success. */
static void meas_constant(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)x;
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; ++i)
    SRUKF_ENTRY(z, i, 0) = 1.0 + (srukf_value)i;
}

static void test_syy_singular(void) {
  const int N = 2, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M); /* all zeros: no measurement noise */
  assert(Q && R);
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);
  /* Benign weights (alpha = 1): with the default alpha = 1e-3 the huge
   * wm[0] turns the exactly-zero deviation column into ~1e-6 roundoff,
   * blurring the singularity this test is constructing. */
  assert(srukf_set_scale(ukf, 1.0, 2.0, 0.0) == SRUKF_RETURN_OK);

  for (int i = 0; i < N; ++i) {
    SRUKF_ENTRY(ukf->x, i, 0) = 3.0;
    SRUKF_ENTRY(ukf->S, i, i) = 1.0;
  }

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = 100.0;
  SRUKF_ENTRY(z, 1, 0) = 200.0;

  srukf_return rc = srukf_correct(ukf, z, meas_constant, NULL);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  /* Transactional: the failed step must not have touched the state. */
  for (int i = 0; i < N; ++i)
    assert(SRUKF_ENTRY(ukf->x, i, 0) == 3.0);

  /* No innovation available from a failed step. */
  srukf_value nis;
  assert(srukf_get_nis(ukf, &nis) == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_syy_singular    OK\n");
}

/* One observable component, one degenerate component: Syy has one
 * healthy diagonal and one ~zero diagonal.  Relative to the healthy
 * one it is singular, so the step must fail. */
static void meas_first_only(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  SRUKF_ENTRY(z, 0, 0) = SRUKF_ENTRY(x, 0, 0);
  SRUKF_ENTRY(z, 1, 0) = 42.0; /* constant: no information, no noise */
}

static void test_syy_partially_singular(void) {
  const int N = 2, M = 2;
  srukf *ukf = srukf_create(N, M);
  assert(ukf);

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  assert(Q && R);
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  SRUKF_ENTRY(R, 0, 0) = 0.1;
  /* R(1,1) left at zero: second component has neither spread nor noise */
  assert(srukf_set_noise(ukf, Q, R) == SRUKF_RETURN_OK);
  /* Benign weights so the degenerate column stays exactly zero. */
  assert(srukf_set_scale(ukf, 1.0, 2.0, 0.0) == SRUKF_RETURN_OK);

  for (int i = 0; i < N; ++i) {
    SRUKF_ENTRY(ukf->x, i, 0) = 1.0;
    SRUKF_ENTRY(ukf->S, i, i) = 1.0;
  }

  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = 2.0;
  SRUKF_ENTRY(z, 1, 0) = 42.0;

  srukf_return rc = srukf_correct(ukf, z, meas_first_only, NULL);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  /* State untouched by the failed step. */
  for (int i = 0; i < N; ++i)
    assert(SRUKF_ENTRY(ukf->x, i, 0) == 1.0);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_syy_partial_singular OK\n");
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

  /* srukf_mat_alloc hardening */
  test_mat_alloc_zero_dims();
  test_mat_alloc_overflow();

  /* set_noise in-place update */
  test_set_noise_in_place();

  /* SRUKF_MAT_ALLOC_NO_DATA */
  test_alloc_matrix_later();

  /* Negative wc[0] */
  test_negative_wc0_predict();
  test_negative_wc0_correct();
  test_negative_wc0_stability();

  /* Cholesky downdate */
  test_chol_downdate_failure();
  test_chol_downdate_failure_3d();
  test_chol_downdate_exact_cancellation();
  test_chol_downdate_near_cancellation();
  test_chol_downdate_success();

  /* Singular / tiny Syy */
  test_tiny_scale_update();
  test_syy_singular();
  test_syy_partially_singular();

  /* propagate_sigma_points errors */
  test_propagate_null_ysig_data();
  test_propagate_dim_mismatch();

  /* Weight recomputation */
  test_weights_recomputation();

  printf("edge case tests passed.\n");
  return 0;
}
