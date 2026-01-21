/* --------------------------------------------------------------------
 * 00_sigma.c - Sigma point generation tests
 *
 * Tests generate_sigma_points_from with various dimensions and
 * parameter configurations.
 *
 * This test includes sr_ukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "sr_ukf.c"

#define EPS 1e-12

/* Helper: verify sigma points structure */
static void verify_sigma_points(sr_ukf *f, lah_mat *Xsig) {
  lah_index N = f->x->nR;
  lah_value gamma = sqrt((lah_value)N + f->lambda);

  /* First column must equal state mean */
  for (lah_index i = 0; i < N; i++)
    assert(fabs(LAH_ENTRY(Xsig, i, 0) - LAH_ENTRY(f->x, i, 0)) < EPS);

  /* Columns 1..N: x + gamma * S(:,k) */
  /* Columns N+1..2N: x - gamma * S(:,k) */
  for (lah_index k = 0; k < N; k++) {
    for (lah_index i = 0; i < N; i++) {
      lah_value plus = LAH_ENTRY(f->x, i, 0) + gamma * LAH_ENTRY(f->S, i, k);
      lah_value minus = LAH_ENTRY(f->x, i, 0) - gamma * LAH_ENTRY(f->S, i, k);
      assert(fabs(LAH_ENTRY(Xsig, i, k + 1) - plus) < EPS);
      assert(fabs(LAH_ENTRY(Xsig, i, k + 1 + N) - minus) < EPS);
    }
  }

  /* Verify symmetry: sigma points should be symmetric around mean */
  for (lah_index k = 0; k < N; k++) {
    for (lah_index i = 0; i < N; i++) {
      lah_value diff_plus = LAH_ENTRY(Xsig, i, k + 1) - LAH_ENTRY(f->x, i, 0);
      lah_value diff_minus =
          LAH_ENTRY(Xsig, i, k + 1 + N) - LAH_ENTRY(f->x, i, 0);
      assert(fabs(diff_plus + diff_minus) < EPS); /* should sum to zero */
    }
  }
}

/* Test 1: Basic 3D case (original test) */
static void test_basic_3d(void) {
  sr_ukf *f = sr_ukf_create(3, 2);
  assert(f && f->x && f->S);

  /* Set state */
  LAH_ENTRY(f->x, 0, 0) = 1.0;
  LAH_ENTRY(f->x, 1, 0) = 2.0;
  LAH_ENTRY(f->x, 2, 0) = 3.0;

  /* Set diagonal covariance sqrt */
  for (lah_index i = 0; i < 3; i++)
    for (lah_index j = 0; j < 3; j++)
      LAH_ENTRY(f->S, i, j) = (i == j) ? 0.1 : 0.0;

  sr_ukf_set_scale(f, 1e-3, 2.0, 0.0);

  lah_mat *Xsig = allocMatrixNow(3, 7); /* N=3, 2N+1=7 */
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);

  verify_sigma_points(f, Xsig);

  lah_matFree(Xsig);
  sr_ukf_free(f);
  printf("  test_basic_3d      OK\n");
}

/* Test 2: Minimal 1D case */
static void test_1d(void) {
  sr_ukf *f = sr_ukf_create(1, 1);
  assert(f);

  LAH_ENTRY(f->x, 0, 0) = 5.0;
  LAH_ENTRY(f->S, 0, 0) = 0.5;

  sr_ukf_set_scale(f, 1.0, 2.0, 0.0);

  lah_mat *Xsig = allocMatrixNow(1, 3); /* N=1, 2N+1=3 */
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);

  verify_sigma_points(f, Xsig);

  /* For 1D: sigma points should be [mean, mean+gamma*s, mean-gamma*s] */
  lah_value gamma = sqrt(1.0 + f->lambda);
  assert(fabs(LAH_ENTRY(Xsig, 0, 0) - 5.0) < EPS);
  assert(fabs(LAH_ENTRY(Xsig, 0, 1) - (5.0 + gamma * 0.5)) < EPS);
  assert(fabs(LAH_ENTRY(Xsig, 0, 2) - (5.0 - gamma * 0.5)) < EPS);

  lah_matFree(Xsig);
  sr_ukf_free(f);
  printf("  test_1d            OK\n");
}

/* Test 3: Higher dimension (10D) */
static void test_10d(void) {
  lah_index N = 10;
  sr_ukf *f = sr_ukf_create((int)N, 5);
  assert(f);

  /* Set state to [1, 2, 3, ..., 10] */
  for (lah_index i = 0; i < N; i++)
    LAH_ENTRY(f->x, i, 0) = (lah_value)(i + 1);

  /* Set diagonal covariance sqrt with varying values */
  for (lah_index i = 0; i < N; i++)
    for (lah_index j = 0; j < N; j++)
      LAH_ENTRY(f->S, i, j) = (i == j) ? 0.1 * (i + 1) : 0.0;

  sr_ukf_set_scale(f, 0.5, 2.0, 1.0);

  lah_mat *Xsig = allocMatrixNow(N, 2 * N + 1);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);

  verify_sigma_points(f, Xsig);

  /* Verify dimensions */
  assert(Xsig->nR == N);
  assert(Xsig->nC == 2 * N + 1);

  lah_matFree(Xsig);
  sr_ukf_free(f);
  printf("  test_10d           OK\n");
}

/* Test 4: Zero state */
static void test_zero_state(void) {
  sr_ukf *f = sr_ukf_create(3, 1);
  assert(f);

  /* State is zero */
  for (lah_index i = 0; i < 3; i++)
    LAH_ENTRY(f->x, i, 0) = 0.0;

  /* Identity covariance sqrt */
  for (lah_index i = 0; i < 3; i++)
    for (lah_index j = 0; j < 3; j++)
      LAH_ENTRY(f->S, i, j) = (i == j) ? 1.0 : 0.0;

  sr_ukf_set_scale(f, 1.0, 2.0, 0.0);

  lah_mat *Xsig = allocMatrixNow(3, 7);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);

  verify_sigma_points(f, Xsig);

  /* First column should be zeros */
  for (lah_index i = 0; i < 3; i++)
    assert(fabs(LAH_ENTRY(Xsig, i, 0)) < EPS);

  lah_matFree(Xsig);
  sr_ukf_free(f);
  printf("  test_zero_state    OK\n");
}

/* Test 5: Different alpha values */
static void test_alpha_variations(void) {
  sr_ukf *f = sr_ukf_create(2, 1);
  assert(f);

  LAH_ENTRY(f->x, 0, 0) = 1.0;
  LAH_ENTRY(f->x, 1, 0) = 2.0;
  LAH_ENTRY(f->S, 0, 0) = 0.5;
  LAH_ENTRY(f->S, 1, 1) = 0.5;
  LAH_ENTRY(f->S, 0, 1) = 0.0;
  LAH_ENTRY(f->S, 1, 0) = 0.0;

  lah_mat *Xsig = allocMatrixNow(2, 5);
  assert(Xsig);

  /* Test with small alpha (tight spread) */
  sr_ukf_set_scale(f, 1e-3, 2.0, 0.0);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);
  verify_sigma_points(f, Xsig);

  /* Test with alpha = 1 (standard spread) */
  sr_ukf_set_scale(f, 1.0, 2.0, 0.0);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);
  verify_sigma_points(f, Xsig);

  /* Test with larger alpha */
  sr_ukf_set_scale(f, 2.0, 2.0, 0.0);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);
  verify_sigma_points(f, Xsig);

  lah_matFree(Xsig);
  sr_ukf_free(f);
  printf("  test_alpha_var     OK\n");
}

/* Test 6: Non-diagonal S (correlated states) */
static void test_correlated_states(void) {
  sr_ukf *f = sr_ukf_create(2, 1);
  assert(f);

  LAH_ENTRY(f->x, 0, 0) = 0.0;
  LAH_ENTRY(f->x, 1, 0) = 0.0;

  /* Non-diagonal lower triangular S (Cholesky factor) */
  LAH_ENTRY(f->S, 0, 0) = 1.0;
  LAH_ENTRY(f->S, 0, 1) = 0.0;
  LAH_ENTRY(f->S, 1, 0) = 0.5;   /* correlation */
  LAH_ENTRY(f->S, 1, 1) = 0.866; /* sqrt(1 - 0.5^2) */

  sr_ukf_set_scale(f, 1.0, 2.0, 0.0);

  lah_mat *Xsig = allocMatrixNow(2, 5);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);

  verify_sigma_points(f, Xsig);

  lah_matFree(Xsig);
  sr_ukf_free(f);
  printf("  test_correlated    OK\n");
}

/* Test 7: Large scale differences */
static void test_scale_differences(void) {
  sr_ukf *f = sr_ukf_create(3, 1);
  assert(f);

  LAH_ENTRY(f->x, 0, 0) = 1e6;  /* large */
  LAH_ENTRY(f->x, 1, 0) = 1.0;  /* normal */
  LAH_ENTRY(f->x, 2, 0) = 1e-6; /* small */

  /* Diagonal S with matching scales */
  LAH_ENTRY(f->S, 0, 0) = 1e3;
  LAH_ENTRY(f->S, 1, 1) = 0.1;
  LAH_ENTRY(f->S, 2, 2) = 1e-9;

  sr_ukf_set_scale(f, 1.0, 2.0, 0.0);

  lah_mat *Xsig = allocMatrixNow(3, 7);
  assert(Xsig);
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig) ==
         lahReturnOk);

  verify_sigma_points(f, Xsig);

  /* Verify all values are finite */
  for (lah_index i = 0; i < 3; i++)
    for (lah_index j = 0; j < 7; j++)
      assert(isfinite(LAH_ENTRY(Xsig, i, j)));

  lah_matFree(Xsig);
  sr_ukf_free(f);
  printf("  test_scale_diff    OK\n");
}

/* Test 8: Error cases */
static void test_errors(void) {
  sr_ukf *f = sr_ukf_create(2, 1);
  assert(f);

  /* NULL Xsig */
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, NULL) ==
         lahReturnParameterError);

  /* Wrong dimensions */
  lah_mat *Xsig_wrong = allocMatrixNow(3, 5); /* wrong rows */
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig_wrong) ==
         lahReturnParameterError);
  lah_matFree(Xsig_wrong);

  Xsig_wrong = allocMatrixNow(2, 4); /* wrong columns */
  assert(generate_sigma_points_from(f->x, f->S, f->lambda, Xsig_wrong) ==
         lahReturnParameterError);
  lah_matFree(Xsig_wrong);

  sr_ukf_free(f);
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
