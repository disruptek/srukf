/* --------------------------------------------------------------------
 * 04_mean_cov.c - Mean and covariance computation tests
 *
 * This test includes sr_ukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "sr_ukf.c"

/* small epsilon for floating point comparison */
#define EPS 1e-10

/* helper: compare two matrices element‑wise */
static void assert_matrix_equal(const lah_mat *A, const lah_mat *B) {
  assert(A && B);
  assert(A->nR == B->nR && A->nC == B->nC);
  for (lah_index i = 0; i < A->nR; ++i)
    for (lah_index j = 0; j < A->nC; ++j) {
      lah_value diff = fabs(LAH_ENTRY(A, i, j) - LAH_ENTRY(B, i, j));
      assert(diff <= EPS);
    }
}

/* helper: build expected mean and covariance from Ysig, wm, wc */
static void expected_mean_cov(const lah_mat *Ysig, const lah_value *wm,
                              const lah_value *wc, lah_mat *x_mean,
                              lah_mat *S_full) {
  lah_index M = Ysig->nR;
  lah_index n_sigma = Ysig->nC;

  /* mean */
  for (lah_index i = 0; i < M; ++i) {
    lah_value val = 0.0;
    for (lah_index k = 0; k < n_sigma; ++k)
      val += wm[k] * LAH_ENTRY(Ysig, i, k);
    LAH_ENTRY(x_mean, i, 0) = val;
  }

  /* full covariance */
  for (lah_index i = 0; i < M; ++i)
    for (lah_index j = 0; j < M; ++j)
      LAH_ENTRY(S_full, i, j) = 0.0;

  for (lah_index k = 0; k < n_sigma; ++k) {
    lah_value w = wc[k];
    for (lah_index i = 0; i < M; ++i) {
      lah_value di = LAH_ENTRY(Ysig, i, k) - LAH_ENTRY(x_mean, i, 0);
      for (lah_index j = 0; j < M; ++j) {
        lah_value dj = LAH_ENTRY(Ysig, j, k) - LAH_ENTRY(x_mean, j, 0);
        LAH_ENTRY(S_full, i, j) += w * di * dj;
      }
    }
  }
}

/* test: 1‑D case with simple numbers */
static void test_basic_1d(void) {
  lah_index M = 1;
  lah_index n = 1;
  lah_index n_sigma = 2 * n + 1; /* 3 */

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);
  /* sigma points: 0.0, 1.0, -1.0 */
  LAH_ENTRY(Ysig, 0, 0) = 0.0;
  LAH_ENTRY(Ysig, 0, 1) = 1.0;
  LAH_ENTRY(Ysig, 0, 2) = -1.0;

  /* ----- compute the correct weights ----- */
  sr_ukf *tmp = sr_ukf_create(1, 1); /* dummy filter to use the routine */
  lah_Return rc = sr_ukf_set_scale(tmp, 1.0, 2.0, 0.0);
  assert(rc == lahReturnOk);
  lah_value wm[3], wc[3];
  for (int i = 0; i < 3; ++i) {
    wm[i] = tmp->wm[i];
    wc[i] = tmp->wc[i];
  }
  sr_ukf_free(tmp);

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);
  assert(x_mean && S_sqrt);

  rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* -----------------------------------------------------------------
     1‑D case: weighted mean is zero, weighted covariance is 1.0
     (σ‑points 1 and –1 each contribute 0.5×1²).
     ----------------------------------------------------------------- */
  assert(fabs(LAH_ENTRY(x_mean, 0, 0)) <= EPS);

  /* full covariance matrix – the library keeps the raw covariance
     (not scaled by ½).  Therefore it must be 1.0. */
  lah_mat *S_full = allocMatrixNow(M, M);
  expected_mean_cov(Ysig, wm, wc, x_mean, S_full);
  printf("Expected full covariance:\n");
  lah_matPrint(S_full, 1);
  fflush(stdout);
  assert(fabs(LAH_ENTRY(S_full, 0, 0) - 1.0) <= EPS);

  /* square‑root of the covariance – Cholesky of 1.0 is 1.0. */
  assert(fabs(LAH_ENTRY(S_sqrt, 0, 0) - sqrt(1.0)) <= EPS);

  /* clean up */
  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  lah_matFree(S_full);
  printf("basic 1D test OK\n");
}

/* test: 2‑D case with random values, check SPD */
static void test_random_2d(void) {
  lah_index M = 2;
  lah_index n = 2;
  lah_index n_sigma = 2 * n + 1; /* 5 */

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);

  /* Sigma points that span 2D space (non-degenerate covariance).
   * Center point plus variations in both dimensions. */
  LAH_ENTRY(Ysig, 0, 0) = 0.0;
  LAH_ENTRY(Ysig, 1, 0) = 0.0; /* center */
  LAH_ENTRY(Ysig, 0, 1) = 1.0;
  LAH_ENTRY(Ysig, 1, 1) = 0.0; /* +x */
  LAH_ENTRY(Ysig, 0, 2) = -1.0;
  LAH_ENTRY(Ysig, 1, 2) = 0.0; /* -x */
  LAH_ENTRY(Ysig, 0, 3) = 0.0;
  LAH_ENTRY(Ysig, 1, 3) = 1.0; /* +y */
  LAH_ENTRY(Ysig, 0, 4) = 0.0;
  LAH_ENTRY(Ysig, 1, 4) = -1.0; /* -y */

  /* weights: standard unscented weights for λ=0 */
  lah_value wm[5] = {0.0, 0.25, 0.25, 0.25, 0.25};
  lah_value wc[5] = {0.0, 0.25, 0.25, 0.25, 0.25};

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);
  assert(x_mean && S_sqrt);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* full covariance for verification */
  lah_mat *S_full = allocMatrixNow(M, M);
  expected_mean_cov(Ysig, wm, wc, x_mean, S_full);

  /* compare full covariance with the result of S_sqrt * S_sqrtᵀ */
  lah_mat *P_from_sqrt = allocMatrixNow(M, M);
  sqrt_to_covariance(S_sqrt, P_from_sqrt);
  assert_matrix_equal(S_full, P_from_sqrt);

  /* SPD test */
  assert(is_spd(S_full));

  /* clean up */
  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  lah_matFree(S_full);
  lah_matFree(P_from_sqrt);
  printf("random 2D test OK\n");
}

/* test: 5D case with proper UKF weights */
static void test_5d(void) {
  lah_index M = 5;
  lah_index n = 5;
  lah_index n_sigma = 2 * n + 1; /* 11 */

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);

  /* Create sigma points centered at [1,2,3,4,5] with identity-like spread */
  lah_value center[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

  /* First sigma point is the mean */
  for (lah_index i = 0; i < M; i++)
    LAH_ENTRY(Ysig, i, 0) = center[i];

  /* Remaining sigma points: mean ± spread in each dimension */
  lah_value spread = 0.5;
  for (lah_index k = 0; k < n; k++) {
    for (lah_index i = 0; i < M; i++) {
      LAH_ENTRY(Ysig, i, k + 1) = center[i] + ((i == k) ? spread : 0.0);
      LAH_ENTRY(Ysig, i, k + 1 + n) = center[i] - ((i == k) ? spread : 0.0);
    }
  }

  /* Get weights from a real UKF */
  sr_ukf *tmp = sr_ukf_create((int)n, 1);
  sr_ukf_set_scale(tmp, 1.0, 2.0, 0.0);
  lah_value *wm = malloc(n_sigma * sizeof(lah_value));
  lah_value *wc = malloc(n_sigma * sizeof(lah_value));
  for (lah_index i = 0; i < n_sigma; i++) {
    wm[i] = tmp->wm[i];
    wc[i] = tmp->wc[i];
  }
  sr_ukf_free(tmp);

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);
  assert(x_mean && S_sqrt);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* Mean should be close to center */
  for (lah_index i = 0; i < M; i++)
    assert(fabs(LAH_ENTRY(x_mean, i, 0) - center[i]) < 0.1);

  /* Verify S*S' gives a valid covariance */
  lah_mat *P = allocMatrixNow(M, M);
  sqrt_to_covariance(S_sqrt, P);
  assert(is_spd(P));

  /* Diagonal should be positive */
  for (lah_index i = 0; i < M; i++)
    assert(LAH_ENTRY(P, i, i) > 0);

  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  lah_matFree(P);
  free(wm);
  free(wc);
  printf("  test_5d             OK\n");
}

/* test: 10D high dimension stress test */
static void test_10d(void) {
  lah_index M = 10;
  lah_index n = 10;
  lah_index n_sigma = 2 * n + 1; /* 21 */

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);

  /* Initialize sigma points as identity-spread around zero */
  for (lah_index k = 0; k < n_sigma; k++)
    for (lah_index i = 0; i < M; i++)
      LAH_ENTRY(Ysig, i, k) = 0.0;

  /* Add spread */
  lah_value spread = 1.0;
  for (lah_index k = 0; k < n; k++) {
    LAH_ENTRY(Ysig, k, k + 1) = spread;
    LAH_ENTRY(Ysig, k, k + 1 + n) = -spread;
  }

  /* Uniform weights */
  lah_value *wm = malloc(n_sigma * sizeof(lah_value));
  lah_value *wc = malloc(n_sigma * sizeof(lah_value));
  lah_value w = 1.0 / (lah_value)n_sigma;
  for (lah_index i = 0; i < n_sigma; i++) {
    wm[i] = w;
    wc[i] = w;
  }

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);
  assert(x_mean && S_sqrt);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* Mean should be zero (symmetric sigma points) */
  for (lah_index i = 0; i < M; i++)
    assert(fabs(LAH_ENTRY(x_mean, i, 0)) < EPS);

  /* Covariance should be SPD */
  lah_mat *P = allocMatrixNow(M, M);
  sqrt_to_covariance(S_sqrt, P);
  assert(is_spd(P));

  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  lah_matFree(P);
  free(wm);
  free(wc);
  printf("  test_10d            OK\n");
}

/* test: large magnitude values */
static void test_large_values(void) {
  lah_index M = 2;
  lah_index n_sigma = 5;

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);

  /* Large values */
  lah_value scale = 1e6;
  LAH_ENTRY(Ysig, 0, 0) = 0.0;
  LAH_ENTRY(Ysig, 1, 0) = 0.0;
  LAH_ENTRY(Ysig, 0, 1) = scale;
  LAH_ENTRY(Ysig, 1, 1) = 0.0;
  LAH_ENTRY(Ysig, 0, 2) = -scale;
  LAH_ENTRY(Ysig, 1, 2) = 0.0;
  LAH_ENTRY(Ysig, 0, 3) = 0.0;
  LAH_ENTRY(Ysig, 1, 3) = scale;
  LAH_ENTRY(Ysig, 0, 4) = 0.0;
  LAH_ENTRY(Ysig, 1, 4) = -scale;

  lah_value wm[5] = {0.0, 0.25, 0.25, 0.25, 0.25};
  lah_value wc[5] = {0.0, 0.25, 0.25, 0.25, 0.25};

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* Mean should be zero */
  assert(fabs(LAH_ENTRY(x_mean, 0, 0)) < 1.0);
  assert(fabs(LAH_ENTRY(x_mean, 1, 0)) < 1.0);

  /* S_sqrt should have finite values */
  for (lah_index i = 0; i < M; i++)
    for (lah_index j = 0; j < M; j++)
      assert(isfinite(LAH_ENTRY(S_sqrt, i, j)));

  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  printf("  test_large_values   OK\n");
}

/* test: small magnitude values */
static void test_small_values(void) {
  lah_index M = 2;
  lah_index n_sigma = 5;

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);

  /* Small values */
  lah_value scale = 1e-8;
  LAH_ENTRY(Ysig, 0, 0) = 0.0;
  LAH_ENTRY(Ysig, 1, 0) = 0.0;
  LAH_ENTRY(Ysig, 0, 1) = scale;
  LAH_ENTRY(Ysig, 1, 1) = 0.0;
  LAH_ENTRY(Ysig, 0, 2) = -scale;
  LAH_ENTRY(Ysig, 1, 2) = 0.0;
  LAH_ENTRY(Ysig, 0, 3) = 0.0;
  LAH_ENTRY(Ysig, 1, 3) = scale;
  LAH_ENTRY(Ysig, 0, 4) = 0.0;
  LAH_ENTRY(Ysig, 1, 4) = -scale;

  lah_value wm[5] = {0.0, 0.25, 0.25, 0.25, 0.25};
  lah_value wc[5] = {0.0, 0.25, 0.25, 0.25, 0.25};

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* All values should be finite */
  for (lah_index i = 0; i < M; i++)
    for (lah_index j = 0; j < M; j++)
      assert(isfinite(LAH_ENTRY(S_sqrt, i, j)));

  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  printf("  test_small_values   OK\n");
}

/* test: correlated sigma points */
static void test_correlated(void) {
  lah_index M = 2;
  lah_index n_sigma = 5;

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);

  /* Correlated sigma points - diagonal spread */
  LAH_ENTRY(Ysig, 0, 0) = 0.0;
  LAH_ENTRY(Ysig, 1, 0) = 0.0;
  LAH_ENTRY(Ysig, 0, 1) = 1.0;
  LAH_ENTRY(Ysig, 1, 1) = 1.0; /* +,+ */
  LAH_ENTRY(Ysig, 0, 2) = -1.0;
  LAH_ENTRY(Ysig, 1, 2) = -1.0; /* -,- */
  LAH_ENTRY(Ysig, 0, 3) = 1.0;
  LAH_ENTRY(Ysig, 1, 3) = -1.0; /* +,- */
  LAH_ENTRY(Ysig, 0, 4) = -1.0;
  LAH_ENTRY(Ysig, 1, 4) = 1.0; /* -,+ */

  lah_value wm[5] = {0.0, 0.25, 0.25, 0.25, 0.25};
  lah_value wc[5] = {0.0, 0.25, 0.25, 0.25, 0.25};

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* Mean should be zero */
  assert(fabs(LAH_ENTRY(x_mean, 0, 0)) < EPS);
  assert(fabs(LAH_ENTRY(x_mean, 1, 0)) < EPS);

  /* Covariance should be SPD */
  lah_mat *P = allocMatrixNow(M, M);
  sqrt_to_covariance(S_sqrt, P);
  assert(is_spd(P));

  /* Should have equal diagonal elements */
  assert(fabs(LAH_ENTRY(P, 0, 0) - LAH_ENTRY(P, 1, 1)) < EPS);

  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  lah_matFree(P);
  printf("  test_correlated     OK\n");
}

/* test: single sigma point (degenerate case) */
static void test_single_sigma(void) {
  lah_index M = 2;
  lah_index n_sigma = 1;

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);
  LAH_ENTRY(Ysig, 0, 0) = 5.0;
  LAH_ENTRY(Ysig, 1, 0) = 3.0;

  lah_value wm[1] = {1.0};
  lah_value wc[1] = {1.0};

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* Mean should equal the single sigma point */
  assert(fabs(LAH_ENTRY(x_mean, 0, 0) - 5.0) < EPS);
  assert(fabs(LAH_ENTRY(x_mean, 1, 0) - 3.0) < EPS);

  /* Covariance should be zero (or regularized) */
  lah_mat *P = allocMatrixNow(M, M);
  sqrt_to_covariance(S_sqrt, P);
  /* Diagonal should be very small (regularization only) */
  assert(LAH_ENTRY(P, 0, 0) < 1e-5);
  assert(LAH_ENTRY(P, 1, 1) < 1e-5);

  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  lah_matFree(P);
  printf("  test_single_sigma   OK\n");
}

/* test: verify sqrt recovery P = S*S' */
static void test_sqrt_recovery(void) {
  lah_index M = 3;
  lah_index n_sigma = 7;

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  assert(Ysig);

  /* Standard sigma points */
  for (lah_index k = 0; k < n_sigma; k++)
    for (lah_index i = 0; i < M; i++)
      LAH_ENTRY(Ysig, i, k) = 0.0;

  /* Add spread */
  for (lah_index k = 0; k < M; k++) {
    LAH_ENTRY(Ysig, k, k + 1) = 1.0;
    LAH_ENTRY(Ysig, k, k + 1 + M) = -1.0;
  }

  /* Get proper UKF weights */
  sr_ukf *tmp = sr_ukf_create((int)M, 1);
  sr_ukf_set_scale(tmp, 1.0, 2.0, 0.0);
  lah_value *wm = malloc(n_sigma * sizeof(lah_value));
  lah_value *wc = malloc(n_sigma * sizeof(lah_value));
  for (lah_index i = 0; i < n_sigma; i++) {
    wm[i] = tmp->wm[i];
    wc[i] = tmp->wc[i];
  }
  sr_ukf_free(tmp);

  lah_mat *x_mean = allocMatrixNow(M, 1);
  lah_mat *S_sqrt = allocMatrixNow(M, M);
  assert(x_mean && S_sqrt);

  lah_Return rc = compute_mean_cov(Ysig, wm, wc, x_mean, S_sqrt);
  assert(rc == lahReturnOk);

  /* Compute expected covariance */
  lah_mat *P_expected = allocMatrixNow(M, M);
  expected_mean_cov(Ysig, wm, wc, x_mean, P_expected);

  /* Compute P from S_sqrt */
  lah_mat *P_from_sqrt = allocMatrixNow(M, M);
  sqrt_to_covariance(S_sqrt, P_from_sqrt);

  /* They should match */
  assert_matrix_equal(P_expected, P_from_sqrt);

  lah_matFree(Ysig);
  lah_matFree(x_mean);
  lah_matFree(S_sqrt);
  lah_matFree(P_expected);
  lah_matFree(P_from_sqrt);
  free(wm);
  free(wc);
  printf("  test_sqrt_recovery  OK\n");
}

/* test: negative and zero weights */
static void test_weights_variations(void) {
  lah_index M = 1;
  lah_index n_sigma = 3;

  lah_mat *Ysig = allocMatrixNow(M, n_sigma);
  LAH_ENTRY(Ysig, 0, 0) = 0.0;
  LAH_ENTRY(Ysig, 0, 1) = 2.0;
  LAH_ENTRY(Ysig, 0, 2) = -2.0;

  /* all zero weights: result should be mean zero.
   * NOTE: Zero covariance isn't SPD, so Cholesky will fail and regularization
   * is applied. The sqrt will be tiny but non-zero (sqrt of regularization). */
  lah_value wm0[3] = {0.0, 0.0, 0.0};
  lah_value wc0[3] = {0.0, 0.0, 0.0};

  lah_mat *x_mean0 = allocMatrixNow(M, 1);
  lah_mat *S_sqrt0 = allocMatrixNow(M, M);
  compute_mean_cov(Ysig, wm0, wc0, x_mean0, S_sqrt0);
  assert(fabs(LAH_ENTRY(x_mean0, 0, 0)) <= EPS);
  /* With regularization enabled, S_sqrt will be small but not exactly zero */
  assert(fabs(LAH_ENTRY(S_sqrt0, 0, 0)) < 1e-5);

  /* negative weights: function still returns a result, but covariance may be
   * negative */
  lah_value wm1[3] = {0.5, 0.25, 0.25};
  lah_value wc1[3] = {0.5, -0.25, -0.25};

  lah_mat *x_mean1 = allocMatrixNow(M, 1);
  lah_mat *S_sqrt1 = allocMatrixNow(M, M);
  compute_mean_cov(Ysig, wm1, wc1, x_mean1, S_sqrt1);

  /* check that the function returned Ok (no hard error) */
  assert(LAH_ENTRY(x_mean1, 0, 0) == (0.5 * 0.0 + 0.25 * 2.0 + 0.25 * -2.0));
  /* covariance may be non‑SPD, so we don't check SPD here */

  /* clean up */
  lah_matFree(Ysig);
  lah_matFree(x_mean0);
  lah_matFree(S_sqrt0);
  lah_matFree(x_mean1);
  lah_matFree(S_sqrt1);
  printf("weights variations test OK\n");
}

/* main: run all tests */
int main(void) {
  printf("Running compute_mean_cov tests...\n");

  test_basic_1d();
  test_random_2d();
  test_5d();
  test_10d();
  test_large_values();
  test_small_values();
  test_correlated();
  test_single_sigma();
  test_sqrt_recovery();
  test_weights_variations();

  printf("mean_cov tests passed.\n");
  return 0;
}
