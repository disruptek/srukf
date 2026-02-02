/* ------------------------------------------------------------------
 *  test_simple.c
 *  Simple unit test for the Square‑Root Unscented Kalman Filter
 *  (srukf).  The test uses a 2‑state linear system
 *  with an identity process model and a scalar measurement.
 *
 *  The filter is exercised through:
 *     * creation from scratch
 *     * setting of noise covariances
 *     * setting of scaling parameters
 *     * generation of sigma points
 *     * one predict / correct cycle
 *
 *  All operations must return lahReturnOk and the state/covariance
 *  matrices must stay internally consistent (SPD, correct
 *  dimensions, etc.).
 *
 *  This test includes srukf.c directly to access internal functions.
 * ------------------------------------------------------------------ */

#include "srukf.c"

#define DEBUG_PRINT 0
static void print_vec(const char *name, const lah_mat *v) {
#if DEBUG_PRINT
  if (!v) {
    printf("%s: NULL\n", name);
    return;
  }
  printf("%s (%zu×%zu):\n", name, v->nR, v->nC);
  for (lah_index i = 0; i < v->nR; ++i)
    printf("  [%zu] = % .6g\n", i, LAH_ENTRY(v, i, 0));
#endif
}

/* Identity process model:  x_k+1 = x_k */
static void id_process(const lah_mat *x, lah_mat *xp, void *user) {
  (void)user; /* unused */
  assert(x->nR == xp->nR && x->nC == xp->nC);
  memcpy(xp->data, x->data, sizeof(lah_value) * x->nR * x->nC);
}

/* Measurement model:  z_k = [x_0, x_1, ..., x_{M-1}]ᵀ + noise */
static void meas_model(const lah_mat *x, lah_mat *z, void *user) {
  (void)user; /* unused */
  /* z is M×1 */
  for (lah_index i = 0; i < z->nR; ++i)
    LAH_ENTRY(z, i, 0) = LAH_ENTRY(x, i, 0);
}

int main(void) {
  /* ---- 1. Build square‑root noise matrices */
  const lah_index N = 7; /* state dimension */
  const lah_index M = 5; /* measurement dimension */

  /* Qsqrt = sqrt( diag(0.01, ..., 0.01) )  -> 0.1 * I */
  lah_mat *Qsqrt = allocMatrixNow(N, N);
  for (lah_index i = 0; i < N; ++i)
    LAH_ENTRY(Qsqrt, i, i) = 0.1;

  /* Rsqrt = sqrt( 0.04 )  -> 0.2 */
  lah_mat *Rsqrt = allocMatrixNow(M, M);
  for (lah_index i = 0; i < M; ++i)
    LAH_ENTRY(Rsqrt, i, i) = 0.2;

  /* ---- 2. Create filter */
  srukf *ukf = srukf_create(N, M);
  if (!ukf) {
    fprintf(stderr, "Filter creation failed.\n");
    return 1;
  }
  lah_Return rc = srukf_set_noise(ukf, Qsqrt, Rsqrt);
  if (rc != lahReturnOk) {
    fprintf(stderr, "Setting noise failed: %d\n", rc);
    srukf_free(ukf);
    return 1;
  }

  /* ---- 3. Set scaling parameters (α, β, κ) */
  rc = srukf_set_scale(ukf, 1e-3, 2.0, 0.0);
  if (rc != lahReturnOk) {
    fprintf(stderr, "Setting scale failed: %d\n", rc);
    srukf_free(ukf);
    return 1;
  }

  /* ---- 4. Verify sigma‑point generation */
  lah_index n_sigma = 2 * N + 1;              /* 15 */
  lah_mat *Xsig = allocMatrixNow(N, n_sigma); /* N rows, 2N+1 columns */
  rc = generate_sigma_points_from(ukf->x, ukf->S, ukf->lambda, Xsig);
  if (rc != lahReturnOk) {
    fprintf(stderr, "Sigma‑point generation failed: %d\n", rc);
    srukf_free(ukf);
    lah_matFree(Xsig);
    return 1;
  }

  /* First column must equal the current state (initially zero). */
  for (lah_index i = 0; i < N; ++i)
    if (fabs(LAH_ENTRY(Xsig, i, 0) - 0.0) > 1e-12) {
      fprintf(stderr, "Sigma point 0 wrong.\n");
      srukf_free(ukf);
      lah_matFree(Xsig);
      return 1;
    }

  /* Gamma = sqrt( N + λ ) should be positive. */
  lah_value gamma = sqrt((lah_value)N + ukf->lambda);
  if (gamma <= 0.0) {
    fprintf(stderr, "Gamma <= 0.\n");
    srukf_free(ukf);
    lah_matFree(Xsig);
    return 1;
  }

  /* ---- 5. Run one predict step */
  rc = srukf_predict(ukf, id_process, NULL);
  if (rc != lahReturnOk) {
    fprintf(stderr, "Predict failed: %d\n", rc);
    srukf_free(ukf);
    lah_matFree(Xsig);
    return 1;
  }

  /* State should still be zero. */
  for (lah_index i = 0; i < N; ++i)
    if (fabs(LAH_ENTRY(ukf->x, i, 0) - 0.0) > 1e-12) {
      fprintf(stderr, "State after predict non‑zero.\n");
      srukf_free(ukf);
      lah_matFree(Xsig);
      return 1;
    }

  /* Covariance should have increased (roughly).  Check SPD: */
  if (!is_spd(ukf->S)) {
    fprintf(stderr, "Covariance not SPD after predict.\n");
    srukf_free(ukf);
    lah_matFree(Xsig);
    return 1;
  }

  /* ---- 6. Run one correct step */
  /* measurement z = 0.1 (slightly off from true state 0). */
  lah_mat *z = allocMatrixNow(M, 1);
  for (lah_index i = 0; i < M; ++i)
    LAH_ENTRY(z, i, 0) = 0.1;

  rc = srukf_correct(ukf, z, meas_model, NULL);
  if (rc != lahReturnOk) {
    fprintf(stderr, "Correct failed: %d\n", rc);
    srukf_free(ukf);
    lah_matFree(Xsig);
    lah_matFree(z);
    return 1;
  }

  /* After correction, state[0] should move toward the measurement (~0.02).  */
  if (fabs(LAH_ENTRY(ukf->x, 0, 0) - 0.02) > 1e-2) {
    fprintf(stderr, "State after correct not close to expected.\n");
    print_vec("x", ukf->x);
    srukf_free(ukf);
    lah_matFree(Xsig);
    lah_matFree(z);
    return 1;
  }

  /* Covariance must remain SPD. */
  if (!is_spd(ukf->S)) {
    fprintf(stderr, "Covariance not SPD after correct.\n");
    srukf_free(ukf);
    lah_matFree(Xsig);
    lah_matFree(z);
    return 1;
  }

  /* ---- 7. Clean up */
  srukf_free(ukf);
  lah_matFree(Xsig);
  lah_matFree(z);

  printf("all checks passed.\n");
  return 0;
}
