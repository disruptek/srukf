/* --------------------------------------------------------------------
 * 30_errors.c - Error handling tests for srukf
 *
 * Tests parameter validation and error returns for all public functions.
 *
 * This test includes srukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "srukf.c"
#include "tests/test_helpers.h"

/* simple helper to allocate a square matrix and set diagonal to 1.0 */
static srukf_mat *alloc_unit_square(srukf_index n) {
  srukf_mat *m = SRUKF_MAT_ALLOC(n, n);
  if (!m)
    return NULL;
  for (srukf_index i = 0; i < n; ++i)
    for (srukf_index j = 0; j < n; ++j)
      SRUKF_ENTRY(m, i, j) = (i == j) ? 1.0 : 0.0;
  return m;
}

/* simple identity measurement model */
static void meas(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; ++i)
    SRUKF_ENTRY(z, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* simple identity process model */
static void process(const srukf_mat *x, srukf_mat *x_out, void *user) {
  (void)user;
  for (srukf_index i = 0; i < x->n_rows; ++i)
    SRUKF_ENTRY(x_out, i, 0) = SRUKF_ENTRY(x, i, 0);
}

/* Test 1: srukf_create with invalid dimensions */
static void test_create_invalid(void) {
  srukf *ukf;

  /* Zero state dimension - should fail or return NULL */
  ukf = srukf_create(0, 1);
  /* Implementation may return NULL or valid - just don't crash */
  if (ukf)
    srukf_free(ukf);

  /* Zero measurement dimension */
  ukf = srukf_create(2, 0);
  if (ukf)
    srukf_free(ukf);

  /* Valid creation */
  ukf = srukf_create(2, 1);
  assert(ukf != NULL);
  srukf_free(ukf);

  printf("  test_create_invalid  OK\n");
}

/* Test 2: srukf_free with NULL */
static void test_free_null(void) {
  /* Should not crash */
  srukf_free(NULL);
  printf("  test_free_null       OK\n");
}

/* Test 3: srukf_set_noise dimension mismatches */
static void test_set_noise_errors(void) {
  srukf *ukf = srukf_create(2, 1);
  assert(ukf);

  srukf_mat *Qgood = alloc_unit_square(2);
  srukf_mat *Rgood = alloc_unit_square(1);
  assert(Qgood && Rgood);

  /* Valid case */
  srukf_return rc = srukf_set_noise(ukf, Qgood, Rgood);
  assert(rc == SRUKF_RETURN_OK);

  /* NULL ukf */
  rc = srukf_set_noise(NULL, Qgood, Rgood);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* NULL Q */
  rc = srukf_set_noise(ukf, NULL, Rgood);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* NULL R */
  rc = srukf_set_noise(ukf, Qgood, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* Wrong Q dimension */
  srukf_mat *Qwrong = alloc_unit_square(3);
  rc = srukf_set_noise(ukf, Qwrong, Rgood);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
  srukf_mat_free(Qwrong);

  /* Wrong R dimension */
  srukf_mat *Rwrong = alloc_unit_square(2);
  rc = srukf_set_noise(ukf, Qgood, Rwrong);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
  srukf_mat_free(Rwrong);

  /* Non-square Q */
  srukf_mat *Qrect = SRUKF_MAT_ALLOC(2, 3);
  if (Qrect) {
    rc = srukf_set_noise(ukf, Qrect, Rgood);
    assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
    srukf_mat_free(Qrect);
  }

  /* Non-square R */
  srukf_mat *Rrect = SRUKF_MAT_ALLOC(1, 2);
  if (Rrect) {
    rc = srukf_set_noise(ukf, Qgood, Rrect);
    assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
    srukf_mat_free(Rrect);
  }

  srukf_mat_free(Qgood);
  srukf_mat_free(Rgood);
  srukf_free(ukf);
  printf("  test_set_noise_err   OK\n");
}

/* Test 4: srukf_set_scale errors */
static void test_set_scale_errors(void) {
  srukf *ukf = srukf_create(2, 1);
  assert(ukf);

  srukf_return rc;

  /* NULL ukf */
  rc = srukf_set_scale(NULL, 1.0, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* alpha = 0 */
  rc = srukf_set_scale(ukf, 0.0, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* alpha < 0 */
  rc = srukf_set_scale(ukf, -1.0, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* Valid alpha */
  rc = srukf_set_scale(ukf, 1e-3, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_OK);

  /* Valid alpha = 1 */
  rc = srukf_set_scale(ukf, 1.0, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_OK);

  /* Large alpha */
  rc = srukf_set_scale(ukf, 10.0, 2.0, 0.0);
  assert(rc == SRUKF_RETURN_OK);

  /* Negative kappa is allowed */
  rc = srukf_set_scale(ukf, 1.0, 2.0, -1.0);
  assert(rc == SRUKF_RETURN_OK);

  /* kappa = -n causes division by zero in lambda clamping path */
  /* For N=2, kappa=-2 triggers this edge case with small alpha */
  rc = srukf_set_scale(ukf, 1e-6, 2.0, -2.0);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(ukf);
  printf("  test_set_scale_err   OK\n");
}

/* Test 5: generate_sigma_points_from errors */
static void test_sigma_points_errors(void) {
  srukf *ukf = srukf_create(2, 1);
  assert(ukf);

  srukf_return rc;

  /* NULL x */
  srukf_mat *Xsig = SRUKF_MAT_ALLOC(2, 5);
  assert(Xsig);
  rc = generate_sigma_points_from(NULL, ukf->S, ukf->lambda, Xsig);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* NULL Xsig */
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* Wrong Xsig rows */
  srukf_mat *Xsig_wrong_rows = SRUKF_MAT_ALLOC(3, 5);
  assert(Xsig_wrong_rows);
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, Xsig_wrong_rows);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
  srukf_mat_free(Xsig_wrong_rows);

  /* Wrong Xsig columns */
  srukf_mat *Xsig_wrong_cols = SRUKF_MAT_ALLOC(2, 4);
  assert(Xsig_wrong_cols);
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, Xsig_wrong_cols);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
  srukf_mat_free(Xsig_wrong_cols);

  /* Valid case */
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, Xsig);
  assert(rc == SRUKF_RETURN_OK);

  srukf_mat_free(Xsig);
  srukf_free(ukf);
  printf("  test_sigma_pts_err   OK\n");
}

/* Test 6: srukf_predict errors */
static void test_predict_errors(void) {
  srukf *ukf = srukf_create(2, 1);
  assert(ukf);

  /* Set noise (required for predict) */
  srukf_mat *Q = alloc_unit_square(2);
  srukf_mat *R = alloc_unit_square(1);
  assert(Q && R);
  srukf_return rc = srukf_set_noise(ukf, Q, R);
  assert(rc == SRUKF_RETURN_OK);

  /* NULL ukf */
  rc = srukf_predict(NULL, process, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* NULL process function */
  rc = srukf_predict(ukf, NULL, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* Valid case */
  rc = srukf_predict(ukf, process, NULL);
  assert(rc == SRUKF_RETURN_OK);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_predict_err     OK\n");
}

/* Test 7: srukf_correct errors */
static void test_correct_errors(void) {
  srukf *ukf = srukf_create(2, 1);
  assert(ukf);

  /* Set noise (required for correct) */
  srukf_mat *Q = alloc_unit_square(2);
  srukf_mat *R = alloc_unit_square(1);
  assert(Q && R);
  srukf_return rc = srukf_set_noise(ukf, Q, R);
  assert(rc == SRUKF_RETURN_OK);

  srukf_mat *z = SRUKF_MAT_ALLOC(1, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = 0.0;

  /* NULL ukf */
  rc = srukf_correct(NULL, z, meas, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* NULL measurement */
  rc = srukf_correct(ukf, NULL, meas, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* NULL measurement function */
  rc = srukf_correct(ukf, z, NULL, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* Wrong measurement dimension */
  srukf_mat *z_wrong = SRUKF_MAT_ALLOC(2, 1);
  assert(z_wrong);
  rc = srukf_correct(ukf, z_wrong, meas, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
  srukf_mat_free(z_wrong);

  /* Valid case */
  rc = srukf_correct(ukf, z, meas, NULL);
  assert(rc == SRUKF_RETURN_OK);

  srukf_mat_free(z);
  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_correct_err     OK\n");
}

/* Test 8: sqrt_to_covariance errors (used as test helper) */
static void test_sqrt_to_cov_errors(void) {
  const srukf_index N = 2;

  srukf_mat *S = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *P = SRUKF_MAT_ALLOC(N, N);
  assert(S && P);

  /* Set up valid S (identity) */
  for (srukf_index i = 0; i < N; i++)
    for (srukf_index j = 0; j < N; j++)
      SRUKF_ENTRY(S, i, j) = (i == j) ? 1.0 : 0.0;

  srukf_return rc;

  /* NULL S */
  rc = sqrt_to_covariance(NULL, P);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* NULL P */
  rc = sqrt_to_covariance(S, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  /* Valid case */
  rc = sqrt_to_covariance(S, P);
  assert(rc == SRUKF_RETURN_OK);

  srukf_mat_free(S);
  srukf_mat_free(P);
  printf("  test_sqrt_to_cov     OK\n");
}

/* Test 9: is_spd edge cases (used as test helper) */
static void test_is_spd(void) {
  const srukf_index N = 2;

  /* NULL matrix */
  assert(is_spd(NULL) == false);

  /* SPD matrix (identity) */
  srukf_mat *identity = alloc_unit_square(N);
  assert(identity);
  assert(is_spd(identity) == true);
  srukf_mat_free(identity);

  /* Non-square matrix */
  srukf_mat *rect = SRUKF_MAT_ALLOC(2, 3);
  if (rect) {
    assert(is_spd(rect) == false);
    srukf_mat_free(rect);
  }

  /* Zero matrix - is_spd adds epsilon to diagonal for numerical
   * stability, so a zero matrix becomes SPD after the epsilon is added */
  srukf_mat *zero = SRUKF_MAT_ALLOC(N, N);
  if (zero) {
    for (srukf_index i = 0; i < N; i++)
      for (srukf_index j = 0; j < N; j++)
        SRUKF_ENTRY(zero, i, j) = 0.0;
    assert(is_spd(zero) == true); /* passes due to epsilon jitter */
    srukf_mat_free(zero);
  }

  /* Negative definite matrix */
  srukf_mat *neg = SRUKF_MAT_ALLOC(N, N);
  if (neg) {
    SRUKF_ENTRY(neg, 0, 0) = -1.0;
    SRUKF_ENTRY(neg, 0, 1) = 0.0;
    SRUKF_ENTRY(neg, 1, 0) = 0.0;
    SRUKF_ENTRY(neg, 1, 1) = -1.0;
    assert(is_spd(neg) == false);
    srukf_mat_free(neg);
  }

  printf("  test_is_spd          OK\n");
}

/* callback that produces NaN */
static void process_nan(const srukf_mat *x, srukf_mat *x_out, void *user) {
  (void)user;
  (void)x;
  for (srukf_index i = 0; i < x_out->n_rows; ++i)
    SRUKF_ENTRY(x_out, i, 0) = NAN;
}

/* callback that produces Inf */
static void meas_inf(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  (void)x;
  for (srukf_index i = 0; i < z->n_rows; ++i)
    SRUKF_ENTRY(z, i, 0) = INFINITY;
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
  srukf *ukf = srukf_create(2, 1);
  assert(ukf);

  /* Set noise (required for predict/correct) */
  srukf_mat *Q = alloc_unit_square(2);
  srukf_mat *R = alloc_unit_square(1);
  assert(Q && R);
  srukf_return rc = srukf_set_noise(ukf, Q, R);
  assert(rc == SRUKF_RETURN_OK);

  /* Enable diagnostic callback */
  g_diag_count = 0;
  srukf_set_diag_callback(test_diag_callback);

  /* predict with callback that returns NaN should fail */
  rc = srukf_predict(ukf, process_nan, NULL);
  assert(rc == SRUKF_RETURN_MATH_ERROR);
  assert(g_diag_count == 1); /* diagnostic should have been called */

  /* correct with callback that returns Inf should fail */
  srukf_mat *z = SRUKF_MAT_ALLOC(1, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = 0.0;
  rc = srukf_correct(ukf, z, meas_inf, NULL);
  assert(rc == SRUKF_RETURN_MATH_ERROR);
  assert(g_diag_count == 2); /* diagnostic should have been called again */

  /* Disable callback */
  srukf_set_diag_callback(NULL);

  srukf_mat_free(z);
  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  printf("  test_callback_valid  OK\n");
}

/* Test 11: srukf_create_from_noise errors */
static void test_create_from_noise_errors(void) {
  srukf_mat *Q = alloc_unit_square(2);
  srukf_mat *R = alloc_unit_square(1);
  assert(Q && R);

  srukf *ukf;

  /* NULL Q */
  ukf = srukf_create_from_noise(NULL, R);
  assert(ukf == NULL);

  /* NULL R */
  ukf = srukf_create_from_noise(Q, NULL);
  assert(ukf == NULL);

  /* Valid case */
  ukf = srukf_create_from_noise(Q, R);
  assert(ukf != NULL);
  srukf_free(ukf);

  srukf_mat_free(Q);
  srukf_mat_free(R);
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
