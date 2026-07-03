/* --------------------------------------------------------------------
 * 50_hooks.c - Custom mean/residual hooks for non-Euclidean spaces
 *
 * Tracks a 1-D angle state across the +/-pi wrap.  Euclidean weighted
 * means are garbage there (the mean of {+3.1, -3.1} is 0, not +/-pi);
 * the circular-mean and wrapped-residual hooks must produce estimates
 * that stay on the circle.
 *
 * This test links against the public API.
 * -------------------------------------------------------------------- */

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "srukf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Wrap an angle to (-pi, pi]. */
static double wrap_angle(double a) {
  return atan2(sin(a), cos(a));
}

/* Wrapped distance |a - b| on the circle. */
static double angle_dist(double a, double b) {
  return fabs(wrap_angle(a - b));
}

/* ------------------- hooks for a 1-D angle space ------------------- */

static void angle_mean(const srukf_mat *sigma, const srukf_value *wm,
                       srukf_mat *mean, void *user) {
  (void)user;
  double s = 0.0, c = 0.0;
  for (srukf_index k = 0; k < sigma->n_cols; ++k) {
    s += wm[k] * sin(SRUKF_ENTRY(sigma, 0, k));
    c += wm[k] * cos(SRUKF_ENTRY(sigma, 0, k));
  }
  SRUKF_ENTRY(mean, 0, 0) = atan2(s, c);
}

static void angle_residual(const srukf_mat *a, const srukf_mat *b,
                           srukf_mat *out, void *user) {
  (void)user;
  double d = SRUKF_ENTRY(a, 0, 0) - SRUKF_ENTRY(b, 0, 0);
  SRUKF_ENTRY(out, 0, 0) = wrap_angle(d);
}

/* ------------------- models ---------------------------------------- */

/* Rotate by a fixed step, wrapped: the boundary crossing generator. */
static void process_rotate(const srukf_mat *x, srukf_mat *x_out, void *user) {
  double step = *(double *)user;
  SRUKF_ENTRY(x_out, 0, 0) = wrap_angle(SRUKF_ENTRY(x, 0, 0) + step);
}

/* Observe the (wrapped) angle directly. */
static void meas_wrapped(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  SRUKF_ENTRY(z, 0, 0) = wrap_angle(SRUKF_ENTRY(x, 0, 0));
}

/* ------------------- filter factory -------------------------------- */

/* Angle filter poised just below the +pi boundary: x = pi - 0.05,
 * S = 0.2, benign weights (alpha = 1 so gamma = 1). */
static srukf *make_angle_filter(void) {
  srukf_mat *Q = SRUKF_MAT_ALLOC(1, 1);
  srukf_mat *R = SRUKF_MAT_ALLOC(1, 1);
  assert(Q && R);
  SRUKF_ENTRY(Q, 0, 0) = 0.01;
  SRUKF_ENTRY(R, 0, 0) = 0.1;

  srukf *ukf = srukf_create_from_noise(Q, R);
  assert(ukf);
  srukf_mat_free(Q);
  srukf_mat_free(R);

  assert(srukf_set_scale(ukf, 1.0, 2.0, 0.0) == SRUKF_RETURN_OK);

  srukf_mat *x0 = SRUKF_MAT_ALLOC(1, 1);
  srukf_mat *S0 = SRUKF_MAT_ALLOC(1, 1);
  assert(x0 && S0);
  SRUKF_ENTRY(x0, 0, 0) = M_PI - 0.05;
  SRUKF_ENTRY(S0, 0, 0) = 0.2;
  assert(srukf_set_state(ukf, x0) == SRUKF_RETURN_OK);
  assert(srukf_set_sqrt_cov(ukf, S0) == SRUKF_RETURN_OK);
  srukf_mat_free(x0);
  srukf_mat_free(S0);

  return ukf;
}

/* ------------------- tests ----------------------------------------- */

/* Predict across the wrap WITH hooks: the sigma points straddle the
 * boundary after the rotation, and the circular mean must land at
 * wrap(x + step) instead of in the middle of the circle. */
static void test_predict_across_wrap(void) {
  srukf *ukf = make_angle_filter();
  assert(srukf_set_state_ops(ukf, angle_mean, angle_residual, NULL) ==
         SRUKF_RETURN_OK);

  double step = 0.2;
  assert(srukf_predict(ukf, process_rotate, &step) == SRUKF_RETURN_OK);

  double expected = wrap_angle(M_PI - 0.05 + step); /* ~ -2.9916 */
  srukf_mat *x = SRUKF_MAT_ALLOC(1, 1);
  srukf_mat *S = SRUKF_MAT_ALLOC(1, 1);
  assert(x && S);
  assert(srukf_get_state(ukf, x) == SRUKF_RETURN_OK);
  assert(srukf_get_sqrt_cov(ukf, S) == SRUKF_RETURN_OK);

  /* The symmetric sigma pair makes the circular mean exact. */
  assert(angle_dist(SRUKF_ENTRY(x, 0, 0), expected) < 1e-9);

  /* Deviations were computed with the wrapped residual: S stays at the
   * genuine local spread, not inflated by 2*pi jumps. */
  double s = SRUKF_ENTRY(S, 0, 0);
  assert(isfinite(s) && fabs(s) > 0.1 && fabs(s) < 0.3);

  srukf_mat_free(x);
  srukf_mat_free(S);
  srukf_free(ukf);
  printf("  test_predict_wrap    OK\n");
}

/* The same predict WITHOUT hooks lands near the circle's midpoint,
 * roughly pi away from the truth: the failure mode the hooks fix. */
static void test_predict_across_wrap_euclidean(void) {
  srukf *ukf = make_angle_filter();

  double step = 0.2;
  assert(srukf_predict(ukf, process_rotate, &step) == SRUKF_RETURN_OK);

  double expected = wrap_angle(M_PI - 0.05 + step);
  srukf_mat *x = SRUKF_MAT_ALLOC(1, 1);
  assert(x);
  assert(srukf_get_state(ukf, x) == SRUKF_RETURN_OK);

  assert(angle_dist(SRUKF_ENTRY(x, 0, 0), expected) > 1.0);

  srukf_mat_free(x);
  srukf_free(ukf);
  printf("  test_predict_wrap_euclid OK (fails as expected without hooks)\n");
}

/* Correct across the wrap WITH measurement hooks: the measurement sits
 * just across the boundary from the prior; the wrapped innovation is
 * 0.07 (not ~ -2*pi + 0.07), so the posterior stays near the boundary
 * and close to the measurement. */
static void test_correct_across_wrap(void) {
  srukf *ukf = make_angle_filter();
  assert(srukf_set_state_ops(ukf, angle_mean, angle_residual, NULL) ==
         SRUKF_RETURN_OK);
  assert(srukf_set_meas_ops(ukf, angle_mean, angle_residual, NULL) ==
         SRUKF_RETURN_OK);

  srukf_mat *z = SRUKF_MAT_ALLOC(1, 1);
  assert(z);
  SRUKF_ENTRY(z, 0, 0) = -M_PI + 0.02; /* 0.07 rad past the prior, wrapped */

  assert(srukf_correct(ukf, z, meas_wrapped, NULL) == SRUKF_RETURN_OK);

  srukf_mat *x = SRUKF_MAT_ALLOC(1, 1);
  assert(x);
  assert(srukf_get_state(ukf, x) == SRUKF_RETURN_OK);

  /* Posterior between prior and measurement, i.e. within 0.07 of the
   * measurement on the circle -- and nowhere near the Euclidean
   * midpoint of the two branch values. */
  assert(angle_dist(SRUKF_ENTRY(x, 0, 0), SRUKF_ENTRY(z, 0, 0)) < 0.05);

  /* The innovation must be the wrapped residual. */
  srukf_mat *innov = SRUKF_MAT_ALLOC(1, 1);
  assert(innov);
  assert(srukf_get_innovation(ukf, innov) == SRUKF_RETURN_OK);
  assert(fabs(SRUKF_ENTRY(innov, 0, 0) - 0.07) < 1e-6);

  srukf_value nis = -1.0;
  assert(srukf_get_nis(ukf, &nis) == SRUKF_RETURN_OK);
  assert(isfinite(nis) && nis >= 0.0);

  srukf_mat_free(innov);
  srukf_mat_free(x);
  srukf_mat_free(z);
  srukf_free(ukf);
  printf("  test_correct_wrap    OK\n");
}

/* Hooks can be cleared back to Euclidean behavior. */
static void test_hooks_cleared(void) {
  srukf *ukf = make_angle_filter();
  assert(srukf_set_state_ops(ukf, angle_mean, angle_residual, NULL) ==
         SRUKF_RETURN_OK);
  assert(srukf_set_state_ops(ukf, NULL, NULL, NULL) == SRUKF_RETURN_OK);

  double step = 0.2;
  assert(srukf_predict(ukf, process_rotate, &step) == SRUKF_RETURN_OK);

  srukf_mat *x = SRUKF_MAT_ALLOC(1, 1);
  assert(x);
  assert(srukf_get_state(ukf, x) == SRUKF_RETURN_OK);
  /* Euclidean garbage again: far from the wrapped truth. */
  assert(angle_dist(SRUKF_ENTRY(x, 0, 0), wrap_angle(M_PI - 0.05 + step)) >
         1.0);

  /* NULL filter is rejected. */
  assert(srukf_set_state_ops(NULL, angle_mean, angle_residual, NULL) ==
         SRUKF_RETURN_PARAMETER_ERROR);
  assert(srukf_set_meas_ops(NULL, NULL, NULL, NULL) ==
         SRUKF_RETURN_PARAMETER_ERROR);

  srukf_mat_free(x);
  srukf_free(ukf);
  printf("  test_hooks_cleared   OK\n");
}

int main(void) {
  printf("Running custom-space hook tests...\n");

  test_predict_across_wrap();
  test_predict_across_wrap_euclidean();
  test_correct_across_wrap();
  test_hooks_cleared();

  printf("custom-space hook tests passed.\n");
  return 0;
}
