/* --------------------------------------------------------------------
 * 30_errors.c - Error handling tests for sr_ukf
 *
 * Tests parameter validation and error returns for all public functions.
 *
 * This test includes sr_ukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "sr_ukf.c"

/* simple helper to allocate a square matrix and set diagonal to 1.0 */
static lah_mat *alloc_unit_square(lah_index n) {
  lah_mat *m = allocMatrixNow(n, n);
  if (!m)
    return NULL;
  for (lah_index i = 0; i < n; ++i)
    for (lah_index j = 0; j < n; ++j)
      LAH_ENTRY(m, i, j) = (i == j) ? 1.0 : 0.0;
  return m;
}

/* simple identity measurement model */
static void meas(const lah_mat *x, lah_mat *z, void *user) {
  (void)user;
  for (lah_index i = 0; i < z->nR; ++i)
    LAH_ENTRY(z, i, 0) = LAH_ENTRY(x, i, 0);
}

/* simple identity process model */
static void process(const lah_mat *x, lah_mat *x_out, void *user) {
  (void)user;
  for (lah_index i = 0; i < x->nR; ++i)
    LAH_ENTRY(x_out, i, 0) = LAH_ENTRY(x, i, 0);
}

/* Test 1: sr_ukf_create with invalid dimensions */
static void test_create_invalid(void) {
  sr_ukf *ukf;

  /* Zero state dimension - should fail or return NULL */
  ukf = sr_ukf_create(0, 1);
  /* Implementation may return NULL or valid - just don't crash */
  if (ukf)
    sr_ukf_free(ukf);

  /* Zero measurement dimension */
  ukf = sr_ukf_create(2, 0);
  if (ukf)
    sr_ukf_free(ukf);

  /* Valid creation */
  ukf = sr_ukf_create(2, 1);
  assert(ukf != NULL);
  sr_ukf_free(ukf);

  printf("  test_create_invalid  OK\n");
}

/* Test 2: sr_ukf_free with NULL */
static void test_free_null(void) {
  /* Should not crash */
  sr_ukf_free(NULL);
  printf("  test_free_null       OK\n");
}

/* Test 3: sr_ukf_set_noise dimension mismatches */
static void test_set_noise_errors(void) {
  sr_ukf *ukf = sr_ukf_create(2, 1);
  assert(ukf);

  lah_mat *Qgood = alloc_unit_square(2);
  lah_mat *Rgood = alloc_unit_square(1);
  assert(Qgood && Rgood);

  /* Valid case */
  lah_Return rc = sr_ukf_set_noise(ukf, Qgood, Rgood);
  assert(rc == lahReturnOk);

  /* NULL ukf */
  rc = sr_ukf_set_noise(NULL, Qgood, Rgood);
  assert(rc == lahReturnParameterError);

  /* NULL Q */
  rc = sr_ukf_set_noise(ukf, NULL, Rgood);
  assert(rc == lahReturnParameterError);

  /* NULL R */
  rc = sr_ukf_set_noise(ukf, Qgood, NULL);
  assert(rc == lahReturnParameterError);

  /* Wrong Q dimension */
  lah_mat *Qwrong = alloc_unit_square(3);
  rc = sr_ukf_set_noise(ukf, Qwrong, Rgood);
  assert(rc == lahReturnParameterError);
  lah_matFree(Qwrong);

  /* Wrong R dimension */
  lah_mat *Rwrong = alloc_unit_square(2);
  rc = sr_ukf_set_noise(ukf, Qgood, Rwrong);
  assert(rc == lahReturnParameterError);
  lah_matFree(Rwrong);

  /* Non-square Q */
  lah_mat *Qrect = allocMatrixNow(2, 3);
  if (Qrect) {
    rc = sr_ukf_set_noise(ukf, Qrect, Rgood);
    assert(rc == lahReturnParameterError);
    lah_matFree(Qrect);
  }

  /* Non-square R */
  lah_mat *Rrect = allocMatrixNow(1, 2);
  if (Rrect) {
    rc = sr_ukf_set_noise(ukf, Qgood, Rrect);
    assert(rc == lahReturnParameterError);
    lah_matFree(Rrect);
  }

  lah_matFree(Qgood);
  lah_matFree(Rgood);
  sr_ukf_free(ukf);
  printf("  test_set_noise_err   OK\n");
}

/* Test 4: sr_ukf_set_scale errors */
static void test_set_scale_errors(void) {
  sr_ukf *ukf = sr_ukf_create(2, 1);
  assert(ukf);

  lah_Return rc;

  /* NULL ukf */
  rc = sr_ukf_set_scale(NULL, 1.0, 2.0, 0.0);
  assert(rc == lahReturnParameterError);

  /* alpha = 0 */
  rc = sr_ukf_set_scale(ukf, 0.0, 2.0, 0.0);
  assert(rc == lahReturnParameterError);

  /* alpha < 0 */
  rc = sr_ukf_set_scale(ukf, -1.0, 2.0, 0.0);
  assert(rc == lahReturnParameterError);

  /* Valid alpha */
  rc = sr_ukf_set_scale(ukf, 1e-3, 2.0, 0.0);
  assert(rc == lahReturnOk);

  /* Valid alpha = 1 */
  rc = sr_ukf_set_scale(ukf, 1.0, 2.0, 0.0);
  assert(rc == lahReturnOk);

  /* Large alpha */
  rc = sr_ukf_set_scale(ukf, 10.0, 2.0, 0.0);
  assert(rc == lahReturnOk);

  /* Negative kappa is allowed */
  rc = sr_ukf_set_scale(ukf, 1.0, 2.0, -1.0);
  assert(rc == lahReturnOk);

  /* kappa = -n causes division by zero in lambda clamping path */
  /* For N=2, kappa=-2 triggers this edge case with small alpha */
  rc = sr_ukf_set_scale(ukf, 1e-6, 2.0, -2.0);
  assert(rc == lahReturnParameterError);

  sr_ukf_free(ukf);
  printf("  test_set_scale_err   OK\n");
}

/* Test 5: generate_sigma_points_from errors */
static void test_sigma_points_errors(void) {
  sr_ukf *ukf = sr_ukf_create(2, 1);
  assert(ukf);

  lah_Return rc;

  /* NULL x */
  lah_mat *Xsig = allocMatrixNow(2, 5);
  assert(Xsig);
  rc = generate_sigma_points_from(NULL, ukf->S, ukf->lambda, Xsig);
  assert(rc == lahReturnParameterError);

  /* NULL Xsig */
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, NULL);
  assert(rc == lahReturnParameterError);

  /* Wrong Xsig rows */
  lah_mat *Xsig_wrong_rows = allocMatrixNow(3, 5);
  assert(Xsig_wrong_rows);
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, Xsig_wrong_rows);
  assert(rc == lahReturnParameterError);
  lah_matFree(Xsig_wrong_rows);

  /* Wrong Xsig columns */
  lah_mat *Xsig_wrong_cols = allocMatrixNow(2, 4);
  assert(Xsig_wrong_cols);
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, Xsig_wrong_cols);
  assert(rc == lahReturnParameterError);
  lah_matFree(Xsig_wrong_cols);

  /* Valid case */
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, Xsig);
  assert(rc == lahReturnOk);

  lah_matFree(Xsig);
  sr_ukf_free(ukf);
  printf("  test_sigma_pts_err   OK\n");
}

/* Test 6: sr_ukf_predict errors */
static void test_predict_errors(void) {
  sr_ukf *ukf = sr_ukf_create(2, 1);
  assert(ukf);

  /* Set noise (required for predict) */
  lah_mat *Q = alloc_unit_square(2);
  lah_mat *R = alloc_unit_square(1);
  assert(Q && R);
  lah_Return rc = sr_ukf_set_noise(ukf, Q, R);
  assert(rc == lahReturnOk);

  /* NULL ukf */
  rc = sr_ukf_predict(NULL, process, NULL);
  assert(rc == lahReturnParameterError);

  /* NULL process function */
  rc = sr_ukf_predict(ukf, NULL, NULL);
  assert(rc == lahReturnParameterError);

  /* Valid case */
  rc = sr_ukf_predict(ukf, process, NULL);
  assert(rc == lahReturnOk);

  lah_matFree(Q);
  lah_matFree(R);
  sr_ukf_free(ukf);
  printf("  test_predict_err     OK\n");
}

/* Test 7: sr_ukf_correct errors */
static void test_correct_errors(void) {
  sr_ukf *ukf = sr_ukf_create(2, 1);
  assert(ukf);

  /* Set noise (required for correct) */
  lah_mat *Q = alloc_unit_square(2);
  lah_mat *R = alloc_unit_square(1);
  assert(Q && R);
  lah_Return rc = sr_ukf_set_noise(ukf, Q, R);
  assert(rc == lahReturnOk);

  lah_mat *z = allocMatrixNow(1, 1);
  assert(z);
  LAH_ENTRY(z, 0, 0) = 0.0;

  /* NULL ukf */
  rc = sr_ukf_correct(NULL, z, meas, NULL);
  assert(rc == lahReturnParameterError);

  /* NULL measurement */
  rc = sr_ukf_correct(ukf, NULL, meas, NULL);
  assert(rc == lahReturnParameterError);

  /* NULL measurement function */
  rc = sr_ukf_correct(ukf, z, NULL, NULL);
  assert(rc == lahReturnParameterError);

  /* Wrong measurement dimension */
  lah_mat *z_wrong = allocMatrixNow(2, 1);
  assert(z_wrong);
  rc = sr_ukf_correct(ukf, z_wrong, meas, NULL);
  assert(rc == lahReturnParameterError);
  lah_matFree(z_wrong);

  /* Valid case */
  rc = sr_ukf_correct(ukf, z, meas, NULL);
  assert(rc == lahReturnOk);

  lah_matFree(z);
  lah_matFree(Q);
  lah_matFree(R);
  sr_ukf_free(ukf);
  printf("  test_correct_err     OK\n");
}

/* Test 8: sqrt_to_covariance errors (used as test helper) */
static void test_sqrt_to_cov_errors(void) {
  const lah_index N = 2;

  lah_mat *S = allocMatrixNow(N, N);
  lah_mat *P = allocMatrixNow(N, N);
  assert(S && P);

  /* Set up valid S (identity) */
  for (lah_index i = 0; i < N; i++)
    for (lah_index j = 0; j < N; j++)
      LAH_ENTRY(S, i, j) = (i == j) ? 1.0 : 0.0;

  lah_Return rc;

  /* NULL S */
  rc = sqrt_to_covariance(NULL, P);
  assert(rc == lahReturnParameterError);

  /* NULL P */
  rc = sqrt_to_covariance(S, NULL);
  assert(rc == lahReturnParameterError);

  /* Valid case */
  rc = sqrt_to_covariance(S, P);
  assert(rc == lahReturnOk);

  lah_matFree(S);
  lah_matFree(P);
  printf("  test_sqrt_to_cov     OK\n");
}

/* Test 9: is_spd edge cases (used as test helper) */
static void test_is_spd(void) {
  const lah_index N = 2;

  /* NULL matrix */
  assert(is_spd(NULL) == false);

  /* SPD matrix (identity) */
  lah_mat *identity = alloc_unit_square(N);
  assert(identity);
  assert(is_spd(identity) == true);
  lah_matFree(identity);

  /* Non-square matrix */
  lah_mat *rect = allocMatrixNow(2, 3);
  if (rect) {
    assert(is_spd(rect) == false);
    lah_matFree(rect);
  }

  /* Zero matrix - is_spd adds epsilon to diagonal for numerical
   * stability, so a zero matrix becomes SPD after the epsilon is added */
  lah_mat *zero = allocMatrixNow(N, N);
  if (zero) {
    for (lah_index i = 0; i < N; i++)
      for (lah_index j = 0; j < N; j++)
        LAH_ENTRY(zero, i, j) = 0.0;
    assert(is_spd(zero) == true); /* passes due to epsilon jitter */
    lah_matFree(zero);
  }

  /* Negative definite matrix */
  lah_mat *neg = allocMatrixNow(N, N);
  if (neg) {
    LAH_ENTRY(neg, 0, 0) = -1.0;
    LAH_ENTRY(neg, 0, 1) = 0.0;
    LAH_ENTRY(neg, 1, 0) = 0.0;
    LAH_ENTRY(neg, 1, 1) = -1.0;
    assert(is_spd(neg) == false);
    lah_matFree(neg);
  }

  printf("  test_is_spd          OK\n");
}

/* callback that produces NaN */
static void process_nan(const lah_mat *x, lah_mat *x_out, void *user) {
  (void)user;
  (void)x;
  for (lah_index i = 0; i < x_out->nR; ++i)
    LAH_ENTRY(x_out, i, 0) = NAN;
}

/* callback that produces Inf */
static void meas_inf(const lah_mat *x, lah_mat *z, void *user) {
  (void)user;
  (void)x;
  for (lah_index i = 0; i < z->nR; ++i)
    LAH_ENTRY(z, i, 0) = INFINITY;
}

/* counter for diagnostic callback invocations */
static int g_diag_count = 0;

/* diagnostic callback that counts invocations and prints */
static void test_diag_callback(const char *msg) {
  g_diag_count++;
  printf("%s\n", msg);
}

/* Test 10: callback validation - NaN/Inf detection */
static void test_callback_validation(void) {
  sr_ukf *ukf = sr_ukf_create(2, 1);
  assert(ukf);

  /* Set noise (required for predict/correct) */
  lah_mat *Q = alloc_unit_square(2);
  lah_mat *R = alloc_unit_square(1);
  assert(Q && R);
  lah_Return rc = sr_ukf_set_noise(ukf, Q, R);
  assert(rc == lahReturnOk);

  /* Enable diagnostic callback */
  g_diag_count = 0;
  sr_ukf_set_diag_callback(test_diag_callback);

  /* predict with callback that returns NaN should fail */
  rc = sr_ukf_predict(ukf, process_nan, NULL);
  assert(rc == lahReturnMathError);
  assert(g_diag_count == 1); /* diagnostic should have been called */

  /* correct with callback that returns Inf should fail */
  lah_mat *z = allocMatrixNow(1, 1);
  assert(z);
  LAH_ENTRY(z, 0, 0) = 0.0;
  rc = sr_ukf_correct(ukf, z, meas_inf, NULL);
  assert(rc == lahReturnMathError);
  assert(g_diag_count == 2); /* diagnostic should have been called again */

  /* Disable callback */
  sr_ukf_set_diag_callback(NULL);

  lah_matFree(z);
  lah_matFree(Q);
  lah_matFree(R);
  sr_ukf_free(ukf);
  printf("  test_callback_valid  OK\n");
}

/* Test 11: sr_ukf_create_from_noise errors */
static void test_create_from_noise_errors(void) {
  lah_mat *Q = alloc_unit_square(2);
  lah_mat *R = alloc_unit_square(1);
  assert(Q && R);

  sr_ukf *ukf;

  /* NULL Q */
  ukf = sr_ukf_create_from_noise(NULL, R);
  assert(ukf == NULL);

  /* NULL R */
  ukf = sr_ukf_create_from_noise(Q, NULL);
  assert(ukf == NULL);

  /* Valid case */
  ukf = sr_ukf_create_from_noise(Q, R);
  assert(ukf != NULL);
  sr_ukf_free(ukf);

  lah_matFree(Q);
  lah_matFree(R);
  printf("  test_create_noise    OK\n");
}

int main(void) {
  printf("Running error handling tests...\n");

  test_create_invalid();
  test_free_null();
  test_set_noise_errors();
  test_set_scale_errors();
  test_sigma_points_errors();
  test_predict_errors();
  test_correct_errors();
  test_sqrt_to_cov_errors();
  test_is_spd();
  test_callback_validation();
  test_create_from_noise_errors();

  printf("error handling tests passed.\n");
  return 0;
}
