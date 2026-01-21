#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "sr_ukf.h"

/* ---------- helper: compute λ from α, κ, N ------------------------ */
static lah_value expected_lambda(lah_value alpha, lah_value kappa,
                                 lah_index N) {
  return alpha * alpha * ((lah_value)N + kappa) - (lah_value)N;
}

/* ---------- helper: compute expected weight vectors --------------- */
static void compute_expected_weights(lah_value lambda, lah_value alpha,
                                     lah_value beta, lah_index n, lah_value *wm,
                                     lah_value *wc) {
  lah_index n_sigma = 2 * n + 1;
  lah_value denom = (lah_value)n + lambda;

  /* mean weights ------------------------------------------------- */
  wm[0] = lambda / denom;
  for (lah_index i = 1; i < n_sigma; ++i)
    wm[i] = 1.0 / (2.0 * denom);

  /* covariance weights ------------------------------------------- */
  wc[0] = wm[0] + (1.0 - alpha * alpha + beta);
  for (lah_index i = 1; i < n_sigma; ++i)
    wc[i] = wm[i];
}

/* ---------- test 1 – basic weight calculation -------------------- */
static void test_basic(void) {
  sr_ukf *ukf = sr_ukf_create(1, 1);
  assert(ukf && ukf->x);

  int rc = sr_ukf_set_scale(ukf, 1e-3, 2.0, 0.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  lah_index n = ukf->x->nR;
  lah_value lambda = expected_lambda(1e-3, 0.0, n);

  lah_value *wm_exp = (lah_value *)calloc(2 * n + 1, sizeof(lah_value));
  lah_value *wc_exp = (lah_value *)calloc(2 * n + 1, sizeof(lah_value));
  compute_expected_weights(lambda, 1e-3, 2.0, n, wm_exp, wc_exp);

  for (lah_index i = 0; i < 2 * n + 1; ++i) {
    assert(fabs(ukf->wm[i] - wm_exp[i]) < 1e-12);
    assert(fabs(ukf->wc[i] - wc_exp[i]) < 1e-12);
  }

  free(wm_exp);
  free(wc_exp);
  sr_ukf_free(ukf);
}

/* ---------- test 2 – larger dimension (N = 5) ------------------- */
static void test_large_dim(void) {
  sr_ukf *ukf = sr_ukf_create(5, 1);
  assert(ukf && ukf->x);

  int rc = sr_ukf_set_scale(ukf, 1.0, 0.0, 1.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  lah_index n = ukf->x->nR;
  lah_value lambda = expected_lambda(1.0, 1.0, n);

  lah_value *wm_exp = (lah_value *)calloc(2 * n + 1, sizeof(lah_value));
  lah_value *wc_exp = (lah_value *)calloc(2 * n + 1, sizeof(lah_value));
  compute_expected_weights(lambda, 1.0, 0.0, n, wm_exp, wc_exp);

  for (lah_index i = 0; i < 2 * n + 1; ++i) {
    assert(fabs(ukf->wm[i] - wm_exp[i]) < 1e-12);
    assert(fabs(ukf->wc[i] - wc_exp[i]) < 1e-12);
  }

  free(wm_exp);
  free(wc_exp);
  sr_ukf_free(ukf);
}

/* ---------- test 3 – reuse of weight vectors ------------------- */
static void test_reuse(void) {
  sr_ukf *ukf = sr_ukf_create(3, 1);
  assert(ukf && ukf->x);

  /* first scale – allocate weights */
  int rc = sr_ukf_set_scale(ukf, 0.5, 1.0, 0.0);
  assert(rc == lahReturnOk);
  void *wm1 = ukf->wm;
  void *wc1 = ukf->wc;

  /* second scale – should not re‑allocate */
  rc = sr_ukf_set_scale(ukf, 0.3, 2.0, 1.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm == wm1);
  assert(ukf->wc == wc1);

  sr_ukf_free(ukf);
}

/* ---------- test 4 – negative λ (λ ≠ –N) --------------------- */
static void test_negative_lambda(void) {
  sr_ukf *ukf = sr_ukf_create(4, 1);
  assert(ukf && ukf->x);

  /* α and κ chosen so that λ is negative but not –N */
  int rc = sr_ukf_set_scale(ukf, 0.1, 2.0, -0.5); /* λ ≈ –4.95  */
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  lah_index n = ukf->x->nR;
  lah_value lambda = expected_lambda(0.1, -0.5, n);

  lah_value *wm_exp = (lah_value *)calloc(2 * n + 1, sizeof(lah_value));
  lah_value *wc_exp = (lah_value *)calloc(2 * n + 1, sizeof(lah_value));
  compute_expected_weights(lambda, 0.1, 2.0, n, wm_exp, wc_exp);

  for (lah_index i = 0; i < 2 * n + 1; ++i) {
    assert(fabs(ukf->wm[i] - wm_exp[i]) < 1e-12);
    assert(fabs(ukf->wc[i] - wc_exp[i]) < 1e-12);
  }

  free(wm_exp);
  free(wc_exp);
  sr_ukf_free(ukf);
}

/* ---------- test 5 – *very* small α --------- */
static void test_small_alpha(void) {
  sr_ukf *ukf = sr_ukf_create(6, 1);
  assert(ukf && ukf->x);

  /* α is chosen so that λ ≈ –N  (n + λ ≈ 0) */
  /* With the new compute_weights this produces NaN/∞ in the weights. */
  /* The old routine keeps λ = eps, so the weights remain finite.   */
  int rc = sr_ukf_set_scale(ukf, 1e-6, 2.0, 0.0);
  assert(rc == lahReturnOk);
  assert(ukf->wm && ukf->wc);

  /* All weights must be *finite* – the new implementation will produce
NaNs. */
  for (lah_index i = 0; i < 2 * ukf->x->nR + 1; ++i) {
    assert(isfinite(ukf->wm[i]));
    assert(isfinite(ukf->wc[i]));
  }

  sr_ukf_free(ukf);
}

/* ---------- main – run all tests --------------------------------- */
int main(void) {
  test_basic();
  printf("  test_basic        OK\n");

  test_large_dim();
  printf("  test_large_dim    OK\n");

  test_reuse();
  printf("  test_reuse        OK\n");

  test_negative_lambda();
  printf("  test_negative_lambda OK\n");

  test_small_alpha();
  printf("  test_small_alpha OK\n");

  printf("compute_weights tests passed.\n");
  return 0;
}
