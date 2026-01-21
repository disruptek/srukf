#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sr_ukf.h"

/* ---------- helper: compute λ from α, κ, N --------------------- */
static lah_value lambda_from(lah_value alpha, lah_value kappa, lah_index N) {
  return alpha * alpha * ((lah_value)N + kappa) - (lah_value)N;
}

/* ---------- helper: compute the expected weight vectors ------------- */
static void expected_weights(lah_value lambda, lah_value alpha, lah_value beta,
                             lah_index n, lah_value *wm, lah_value *wc) {
  lah_index n_sigma = 2 * n + 1;
  lah_value denom = (lah_value)n + lambda;

  /* mean weights ------------------------------------------------- */
  wm[0] = lambda / denom;
  for (lah_index i = 1; i < n_sigma; ++i)
    wm[i] = 1.0 / (2.0 * denom);

  /* covariance weights --------------------------------------------- */
  wc[0] = wm[0] + (1.0 - alpha * alpha + beta);
  for (lah_index i = 1; i < n_sigma; ++i)
    wc[i] = wm[i];
}

/* ---------- test 1 – standard parameters (N = 1) ------------------- */
static void test_standard(void) {
  sr_ukf *ukf = sr_ukf_create(1, 1);
  assert(ukf && ukf->x);

  int rc = sr_ukf_set_scale(ukf, 1e-3, 2.0, 0.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  lah_index n = ukf->x->nR; /* = 1 */
  lah_value lam = lambda_from(1e-3, 0.0, n);

  lah_value *wm_exp = calloc(2 * n + 1, sizeof(lah_value));
  lah_value *wc_exp = calloc(2 * n + 1, sizeof(lah_value));
  expected_weights(lam, 1e-3, 2.0, n, wm_exp, wc_exp);

  for (lah_index i = 0; i < 2 * n + 1; ++i) {
    assert(fabs(ukf->wm[i] - wm_exp[i]) < 1e-12);
    assert(fabs(ukf->wc[i] - wc_exp[i]) < 1e-12);
  }

  free(wm_exp);
  free(wc_exp);
  sr_ukf_free(ukf);
}

/* ---------- test 2 – larger dimension (N = 5) --------------------- */
static void test_large(void) {
  sr_ukf *ukf = sr_ukf_create(5, 1);
  assert(ukf && ukf->x);

  int rc = sr_ukf_set_scale(ukf, 1.0, 0.0, 1.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  lah_index n = ukf->x->nR; /* = 5 */
  lah_value lam = lambda_from(1.0, 1.0, n);

  lah_value *wm_exp = calloc(2 * n + 1, sizeof(lah_value));
  lah_value *wc_exp = calloc(2 * n + 1, sizeof(lah_value));
  expected_weights(lam, 1.0, 0.0, n, wm_exp, wc_exp);

  for (lah_index i = 0; i < 2 * n + 1; ++i) {
    assert(fabs(ukf->wm[i] - wm_exp[i]) < 1e-12);
    assert(fabs(ukf->wc[i] - wc_exp[i]) < 1e-12);
  }

  free(wm_exp);
  free(wc_exp);
  sr_ukf_free(ukf);
}

/* ---------- test 3 – re‑use of weight vectors --------------------- */
static void test_reuse(void) {
  sr_ukf *ukf = sr_ukf_create(3, 1);
  assert(ukf && ukf->x);

  /* first call – allocates the vectors */
  int rc = sr_ukf_set_scale(ukf, 0.5, 1.0, 0.0);
  assert(rc == lahReturnOk);
  void *wm1 = ukf->wm;
  void *wc1 = ukf->wc;

  /* second call – should NOT allocate new memory */
  rc = sr_ukf_set_scale(ukf, 0.3, 2.0, 1.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm == wm1);
  assert(ukf->wc == wc1);

  sr_ukf_free(ukf);
}

/* ---------- test 4 – negative λ (but ≠ –N) ----------------------- */
static void test_negative_lambda(void) {
  sr_ukf *ukf = sr_ukf_create(4, 1);
  assert(ukf && ukf->x);

  /* choose α and κ so that λ is negative but not –N */
  int rc = sr_ukf_set_scale(ukf, 0.1, 2.0, -0.5); /* λ ≈ –4.95  */
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  lah_index n = ukf->x->nR;
  lah_value lam = lambda_from(0.1, -0.5, n);

  lah_value *wm_exp = calloc(2 * n + 1, sizeof(lah_value));
  lah_value *wc_exp = calloc(2 * n + 1, sizeof(lah_value));
  expected_weights(lam, 0.1, 2.0, n, wm_exp, wc_exp);

  for (lah_index i = 0; i < 2 * n + 1; ++i) {
    assert(fabs(ukf->wm[i] - wm_exp[i]) < 1e-12);
    assert(fabs(ukf->wc[i] - wc_exp[i]) < 1e-12);
  }

  free(wm_exp);
  free(wc_exp);
  sr_ukf_free(ukf);
}

/* ---------- test 5 – very small α (α → 0) ------------ */
static void test_small_alpha(void) {
  sr_ukf *ukf = sr_ukf_create(6, 1);
  assert(ukf && ukf->x);

  /* Set the filter with an extremely small α. */
  int rc = sr_ukf_set_scale(ukf, 1e-12, 2.0, 0.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  lah_index n = ukf->x->nR;

  /* Compute λ the same way as sr_ukf_set_scale() does: */
  lah_value lambda = 1e-12 * 1e-12 * ((lah_value)n + 0.0) - (lah_value)n;
  const lah_value eps = 1e-12;
  if (fabs((double)(n + lambda)) < eps)
    lambda = -(lah_value)n + eps; /* same clamp as in sr_ukf_set_scale() */

  /* Allocate expected weight vectors. */
  lah_value *wm_exp = calloc(2 * n + 1, sizeof(lah_value));
  lah_value *wc_exp = calloc(2 * n + 1, sizeof(lah_value));
  expected_weights(lambda, 1e-12, 2.0, n, wm_exp, wc_exp);

  /* Compare the filter's weights against the expected values. */
  for (lah_index i = 0; i < 2 * n + 1; ++i) {
    assert(fabs(ukf->wm[i] - wm_exp[i]) < 1e-12);
    assert(fabs(ukf->wc[i] - wc_exp[i]) < 1e-12);
  }

  free(wm_exp);
  free(wc_exp);
  sr_ukf_free(ukf);
}

/* ---------- main – run all tests -------------------------------- */
int main(void) {
  test_standard();
  printf("  test_standard      OK\n");

  test_large();
  printf("  test_large        OK\n");

  test_reuse();
  printf("  test_reuse        OK\n");

  test_negative_lambda();
  printf("  test_negative_lambda OK\n");

  test_small_alpha();
  printf("  test_small_alpha  OK\n");

  printf("compute_weights tests passed.\n");
  return 0;
}
