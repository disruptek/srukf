/* --------------------------------------------------------------------
 * 48_atomicity.c - Transactional semantics of srukf_predict_to and
 * srukf_correct_to: on ANY error the user's x and S buffers must be
 * left exactly as they were.
 *
 * The interesting case is a failure that occurs late in the correct
 * step, after a naive in-place implementation has already begun
 * mutating S: a Cholesky downdate of the state covariance that goes
 * non-SPD. Such a failure cannot be produced by a well-behaved
 * measurement model (the sigma-point joint covariance is consistent by
 * construction), so we engineer one with a measurement "model" that
 * ignores x and returns scripted values per sigma-point index, combined
 * with a negative zeroth covariance weight (beta = 0, kappa < 0).
 *
 * Derivation of the constants (N = 2, M = 1, S = I, x = 0):
 *   alpha = 0.5, beta = 0, kappa = -1.5
 *     => t = n + lambda = alpha^2 (n + kappa) = 0.125
 *        gamma = sqrt(t), wm0 = 1 - n/t = -15, wm_i = 1/(2t) = 4
 *        wc0 = wm0 + 1 - alpha^2 + beta = -14.25, wc_i = 4
 *   scripted measurements v = (D, c1, c2, -c1, -c2)
 *     => y_mean = wm0 * D = -15 D
 *        Syy^2 = 8 c1^2 + 8 c2^2 + R^2 - 48 D^2
 *        Pxz   = 8 gamma (c1, c2)'
 *   with c1 = 0.2, c2 = 1, R = 0.1 and Syy^2 targeted at 1.0:
 *        D = sqrt((8.32 + 0.01 - 1.0) / 48)
 *        U = K * Syy = Pxz / Syy = (0.5657, 2.8284)
 *   Downdating S = I by U succeeds for column 0 (u0^2 = 0.32 < 1),
 *   mutating it, then fails at column 1 ((1 - u0^2) u1^2 = 5.44 > 1).
 *   A non-transactional implementation leaves S corrupted; a
 *   transactional one must return SRUKF_RETURN_MATH_ERROR and leave
 *   x and S untouched.
 *
 * This test links against the public API.
 * -------------------------------------------------------------------- */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "srukf.h"

#define EPS 1e-12

/* Scripted measurement model: returns v[k] for the k-th call,
 * regardless of the input state. */
typedef struct {
  const srukf_value *v;
  int k;
} script_ctx;

static void h_scripted(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)x;
  script_ctx *ctx = (script_ctx *)user;
  SRUKF_ENTRY(z, 0, 0) = ctx->v[ctx->k++];
}

/* Process model that always emits NaN, to fail predict early. */
static void f_nan(const srukf_mat *x, srukf_mat *x_out, void *user) {
  (void)x;
  (void)user;
  for (srukf_index i = 0; i < x_out->n_rows; ++i)
    SRUKF_ENTRY(x_out, i, 0) = NAN;
}

static srukf *make_filter(void) {
  srukf_mat *Q = SRUKF_MAT_ALLOC(2, 2);
  srukf_mat *R = SRUKF_MAT_ALLOC(1, 1);
  assert(Q && R);
  SRUKF_ENTRY(Q, 0, 0) = 0.1;
  SRUKF_ENTRY(Q, 1, 1) = 0.1;
  SRUKF_ENTRY(R, 0, 0) = 0.1;
  srukf *ukf = srukf_create_from_noise(Q, R);
  srukf_mat_free(Q);
  srukf_mat_free(R);
  assert(ukf);
  assert(srukf_set_scale(ukf, 0.5, 0.0, -1.5) == SRUKF_RETURN_OK);
  return ukf;
}

/* correct_to fails in the state-covariance downdate, after the point
 * where a non-transactional implementation has begun mutating S. */
static void test_correct_to_atomic_on_downdate_failure(void) {
  srukf *ukf = make_filter();

  srukf_mat *x = SRUKF_MAT_ALLOC(2, 1);
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  srukf_mat *z = SRUKF_MAT_ALLOC(1, 1);
  assert(x && S && z);
  SRUKF_ENTRY(S, 0, 0) = 1.0;
  SRUKF_ENTRY(S, 1, 1) = 1.0;
  SRUKF_ENTRY(z, 0, 0) = 0.0;

  const srukf_value D = sqrt((8.32 + 0.01 - 1.0) / 48.0);
  const srukf_value v[5] = {D, 0.2, 1.0, -0.2, -1.0};
  script_ctx ctx = {v, 0};

  srukf_return rc = srukf_correct_to(ukf, x, S, z, h_scripted, &ctx);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  /* the user's buffers must be exactly as before the failed call */
  assert(fabs(SRUKF_ENTRY(x, 0, 0)) < EPS);
  assert(fabs(SRUKF_ENTRY(x, 1, 0)) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 0, 0) - 1.0) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 1, 0)) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 0, 1)) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 1, 1) - 1.0) < EPS);

  srukf_mat_free(x);
  srukf_mat_free(S);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_correct_to_atomic OK\n");
}

/* The same scripted failure through srukf_correct must leave the
 * filter's internal state untouched. */
static void test_correct_internal_atomic(void) {
  srukf *ukf = make_filter();
  assert(srukf_reset(ukf, 1.0) == SRUKF_RETURN_OK);

  srukf_mat *z = SRUKF_MAT_ALLOC(1, 1);
  srukf_mat *S_after = SRUKF_MAT_ALLOC(2, 2);
  srukf_mat *x_after = SRUKF_MAT_ALLOC(2, 1);
  assert(z && S_after && x_after);

  const srukf_value D = sqrt((8.32 + 0.01 - 1.0) / 48.0);
  const srukf_value v[5] = {D, 0.2, 1.0, -0.2, -1.0};
  script_ctx ctx = {v, 0};

  srukf_return rc = srukf_correct(ukf, z, h_scripted, &ctx);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  assert(srukf_get_state(ukf, x_after) == SRUKF_RETURN_OK);
  assert(srukf_get_sqrt_cov(ukf, S_after) == SRUKF_RETURN_OK);
  assert(fabs(SRUKF_ENTRY(x_after, 0, 0)) < EPS);
  assert(fabs(SRUKF_ENTRY(x_after, 1, 0)) < EPS);
  assert(fabs(SRUKF_ENTRY(S_after, 0, 0) - 1.0) < EPS);
  assert(fabs(SRUKF_ENTRY(S_after, 1, 0)) < EPS);
  assert(fabs(SRUKF_ENTRY(S_after, 1, 1) - 1.0) < EPS);

  srukf_mat_free(z);
  srukf_mat_free(S_after);
  srukf_mat_free(x_after);
  srukf_free(ukf);
  printf("  test_correct_atomic    OK\n");
}

/* predict_to failing on a NaN-producing model must leave the buffers
 * unchanged. (The NaN check runs before any mutation even in a naive
 * implementation; this is a regression guard for the staged commit.) */
static void test_predict_to_atomic_on_nan(void) {
  srukf *ukf = make_filter();

  srukf_mat *x = SRUKF_MAT_ALLOC(2, 1);
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  assert(x && S);
  SRUKF_ENTRY(x, 0, 0) = 3.0;
  SRUKF_ENTRY(x, 1, 0) = -4.0;
  SRUKF_ENTRY(S, 0, 0) = 2.0;
  SRUKF_ENTRY(S, 1, 0) = 0.5;
  SRUKF_ENTRY(S, 1, 1) = 1.5;

  srukf_return rc = srukf_predict_to(ukf, x, S, f_nan, NULL);
  assert(rc == SRUKF_RETURN_MATH_ERROR);

  assert(fabs(SRUKF_ENTRY(x, 0, 0) - 3.0) < EPS);
  assert(fabs(SRUKF_ENTRY(x, 1, 0) + 4.0) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 0, 0) - 2.0) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 1, 0) - 0.5) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 0, 1)) < EPS);
  assert(fabs(SRUKF_ENTRY(S, 1, 1) - 1.5) < EPS);

  srukf_mat_free(x);
  srukf_mat_free(S);
  srukf_free(ukf);
  printf("  test_predict_to_atomic OK\n");
}

int main(void) {
  printf("Running transactional atomicity tests...\n");

  test_correct_to_atomic_on_downdate_failure();
  test_correct_internal_atomic();
  test_predict_to_atomic_on_nan();

  printf("atomicity tests passed.\n");
  return 0;
}
