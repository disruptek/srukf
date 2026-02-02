/* --------------------------------------------------------------------
 * 00_sigma.c - Sigma point generation tests
 *
 * Tests generate_sigma_points_from with various dimensions and
 * parameter configurations.
 *
 * This test includes srukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "srukf.c"

#define EPS 1e-12

/* Helper: verify sigma points structure */
static void verify_sigma_points(srukf *f, srukf_mat *Xsig) {
  srukf_index N = f->x->n_rows;
  srukf_value gamma = sqrt((srukf_value)N + f->lambda);

  /* First column must equal state mean */
  for (srukf_index i = 0; i < N; i++)
    assert(fabs(SRUKF_ENTRY(Xsig, i, 0) - SRUKF_ENTRY(f->x, i, 0)) < EPS);

  /* Columns 1..N: x + gamma * S(:,k) */
  /* Columns N+1..2N: x - gamma * S(:,k) */
  for (srukf_index k = 0; k < N; k++) {
    for (srukf_index i = 0; i < N; i++) {
      srukf_value plus = SRUKF_ENTRY(f->x, i, 0) + gamma * SRUKF_ENTRY(f->S, i, k);
      srukf_value minus = SRUKF_ENTRY(f->x, i, 0) - gamma * SRUKF_ENTRY(f->S, i, k);
      assert(fabs(SRUKF_ENTRY(Xsig, i, k + 1) - plus) < EPS);
      assert(fabs(SRUKF_ENTRY(Xsig, i, k + 1 + N) - minus) < EPS);
    }
  }

  /* Verify symmetry: sigma points should be symmetric around mean */
  for (srukf_index k = 0; k < N; k++) {
    for (srukf_index i = 0; i < N; i++) {
      srukf_value diff_plus = SRUKF_ENTRY(Xsig, i, k + 1) - SRUKF_ENTRY(f->x, i, 0);
      srukf_value diff_minus =
          SRUKF_ENTRY(Xsig, i, k + 1 + N) - SRUKF_ENTRY(f->x, i, 0);
      assert(fabs(diff_plus + diff_minus) < EPS); /* should sum to zero */
    }
  }
}

/* Test 1: Basic 3D case (original test) */
static void test_basic_3d(void) {
  srukf *f = srukf_create(3, 2);
  assert(f && f->x && f->S);

  /* Set state */
  SRUKF_ENTRY(f->x, 0, 0) = 1.0;
  SRUKF_ENTRY(f->x, 1, 0) = 2.0;
  SRUKF_ENTRY(f->x, 2, 0) = 3.0;

  /* Set diagonal covariance sqrt */
  for (srukf_index i = 0; i < 3; i++)
    for (srukf_index j = 0; j < 3; j++)
      SRUKF_ENTRY(f->S, i, j) = (i == j) ? 0.1 : 0.0;

  srukf_set_scale(f, 1e-3, 2.0, 0.0);

  srukf_mat *Xsig = SRUKF_MAT_ALLOC(3, 7); /* N=3, 2N+1=7 */
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);

  verify_sigma_points(f, Xsig);

  srukf_mat_free(Xsig);
  srukf_free(f);
  printf("  test_basic_3d      OK\n");
}

/* Test 2: Minimal 1D case */
static void test_1d(void) {
  srukf *f = srukf_create(1, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 5.0;
  SRUKF_ENTRY(f->S, 0, 0) = 0.5;

  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_mat *Xsig = SRUKF_MAT_ALLOC(1, 3); /* N=1, 2N+1=3 */
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);

  verify_sigma_points(f, Xsig);

  /* For 1D: sigma points should be [mean, mean+gamma*s, mean-gamma*s] */
  srukf_value gamma = sqrt(1.0 + f->lambda);
  assert(fabs(SRUKF_ENTRY(Xsig, 0, 0) - 5.0) < EPS);
  assert(fabs(SRUKF_ENTRY(Xsig, 0, 1) - (5.0 + gamma * 0.5)) < EPS);
  assert(fabs(SRUKF_ENTRY(Xsig, 0, 2) - (5.0 - gamma * 0.5)) < EPS);

  srukf_mat_free(Xsig);
  srukf_free(f);
  printf("  test_1d            OK\n");
}

/* Test 3: Higher dimension (10D) */
static void test_10d(void) {
  srukf_index N = 10;
  srukf *f = srukf_create((int)N, 5);
  assert(f);

  /* Set state to [1, 2, 3, ..., 10] */
  for (srukf_index i = 0; i < N; i++)
    SRUKF_ENTRY(f->x, i, 0) = (srukf_value)(i + 1);

  /* Set diagonal covariance sqrt with varying values */
  for (srukf_index i = 0; i < N; i++)
    for (srukf_index j = 0; j < N; j++)
      SRUKF_ENTRY(f->S, i, j) = (i == j) ? 0.1 * (i + 1) : 0.0;

  srukf_set_scale(f, 0.5, 2.0, 1.0);

  srukf_mat *Xsig = SRUKF_MAT_ALLOC(N, 2 * N + 1);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);

  verify_sigma_points(f, Xsig);

  /* Verify dimensions */
  assert(Xsig->n_rows == N);
  assert(Xsig->n_cols == 2 * N + 1);

  srukf_mat_free(Xsig);
  srukf_free(f);
  printf("  test_10d           OK\n");
}

/* Test 4: Zero state */
static void test_zero_state(void) {
  srukf *f = srukf_create(3, 1);
  assert(f);

  /* State is zero */
  for (srukf_index i = 0; i < 3; i++)
    SRUKF_ENTRY(f->x, i, 0) = 0.0;

  /* Identity covariance sqrt */
  for (srukf_index i = 0; i < 3; i++)
    for (srukf_index j = 0; j < 3; j++)
      SRUKF_ENTRY(f->S, i, j) = (i == j) ? 1.0 : 0.0;

  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_mat *Xsig = SRUKF_MAT_ALLOC(3, 7);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);

  verify_sigma_points(f, Xsig);

  /* First column should be zeros */
  for (srukf_index i = 0; i < 3; i++)
    assert(fabs(SRUKF_ENTRY(Xsig, i, 0)) < EPS);

  srukf_mat_free(Xsig);
  srukf_free(f);
  printf("  test_zero_state    OK\n");
}

/* Test 5: Different alpha values */
static void test_alpha_variations(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 1.0;
  SRUKF_ENTRY(f->x, 1, 0) = 2.0;
  SRUKF_ENTRY(f->S, 0, 0) = 0.5;
  SRUKF_ENTRY(f->S, 1, 1) = 0.5;
  SRUKF_ENTRY(f->S, 0, 1) = 0.0;
  SRUKF_ENTRY(f->S, 1, 0) = 0.0;

  srukf_mat *Xsig = SRUKF_MAT_ALLOC(2, 5);
  assert(Xsig);

  /* Test with small alpha (tight spread) */
  srukf_set_scale(f, 1e-3, 2.0, 0.0);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);
  verify_sigma_points(f, Xsig);

  /* Test with alpha = 1 (standard spread) */
  srukf_set_scale(f, 1.0, 2.0, 0.0);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);
  verify_sigma_points(f, Xsig);

  /* Test with larger alpha */
  srukf_set_scale(f, 2.0, 2.0, 0.0);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);
  verify_sigma_points(f, Xsig);

  srukf_mat_free(Xsig);
  srukf_free(f);
  printf("  test_alpha_var     OK\n");
}

/* Test 6: Non-diagonal S (correlated states) */
static void test_correlated_states(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 0.0;
  SRUKF_ENTRY(f->x, 1, 0) = 0.0;

  /* Non-diagonal lower triangular S (Cholesky factor) */
  SRUKF_ENTRY(f->S, 0, 0) = 1.0;
  SRUKF_ENTRY(f->S, 0, 1) = 0.0;
  SRUKF_ENTRY(f->S, 1, 0) = 0.5;   /* correlation */
  SRUKF_ENTRY(f->S, 1, 1) = 0.866; /* sqrt(1 - 0.5^2) */

  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_mat *Xsig = SRUKF_MAT_ALLOC(2, 5);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);

  verify_sigma_points(f, Xsig);

  srukf_mat_free(Xsig);
  srukf_free(f);
  printf("  test_correlated    OK\n");
}

/* Test 7: Large scale differences */
static void test_scale_differences(void) {
  srukf *f = srukf_create(3, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 1e6;  /* large */
  SRUKF_ENTRY(f->x, 1, 0) = 1.0;  /* normal */
  SRUKF_ENTRY(f->x, 2, 0) = 1e-6; /* small */

  /* Diagonal S with matching scales */
  SRUKF_ENTRY(f->S, 0, 0) = 1e3;
  SRUKF_ENTRY(f->S, 1, 1) = 0.1;
  SRUKF_ENTRY(f->S, 2, 2) = 1e-9;

  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_mat *Xsig = SRUKF_MAT_ALLOC(3, 7);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         SRUKF_RETURN_OK);

  verify_sigma_points(f, Xsig);

  /* Verify all values are finite */
  for (srukf_index i = 0; i < 3; i++)
    for (srukf_index j = 0; j < 7; j++)
      assert(isfinite(SRUKF_ENTRY(Xsig, i, j)));

  srukf_mat_free(Xsig);
  srukf_free(f);
  printf("  test_scale_diff    OK\n");
}

/* Test 8: Error cases */
static void test_errors(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  /* NULL Xsig */
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, NULL) ==
         SRUKF_RETURN_PARAMETER_ERROR);

  /* Wrong dimensions */
  srukf_mat *Xsig_wrong = SRUKF_MAT_ALLOC(3, 5); /* wrong rows */
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig_wrong) ==
         SRUKF_RETURN_PARAMETER_ERROR);
  srukf_mat_free(Xsig_wrong);

  Xsig_wrong = SRUKF_MAT_ALLOC(2, 4); /* wrong columns */
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig_wrong) ==
         SRUKF_RETURN_PARAMETER_ERROR);
  srukf_mat_free(Xsig_wrong);

  srukf_free(f);
  printf("  test_errors        OK\n");
}

int main(void) {
  printf("Running sigma point generation tests...\n");

  test_basic_3d();
  test_1d();
  test_10d();
  test_zero_state();
  test_alpha_variations();
  test_correlated_states();
  test_scale_differences();
  test_errors();

  printf("sigma‑point generation test passed\n");
  return 0;
}
