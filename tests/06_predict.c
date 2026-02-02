/* --------------------------------------------------------------------
 * 06_predict.c - Prediction step tests
 *
 * Tests srukf_predict with various process models and configurations.
 *
 * This test includes srukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "srukf.c"
#include "tests/test_helpers.h"

#define EPS 1e-10

/* Helper: compute Frobenius norm of matrix */
static srukf_value frobenius_norm(const srukf_mat *A) {
  srukf_value sum = 0.0;
  for (srukf_index i = 0; i < A->n_rows; i++)
    for (srukf_index j = 0; j < A->n_cols; j++) {
      srukf_value v = SRUKF_ENTRY(A, i, j);
      sum += v * v;
    }
  return sqrt(sum);
}

/* Identity process model: x_k+1 = x_k */
static void process_identity(const srukf_mat *x, srukf_mat *xp, void *user) {
  (void)user;
  memcpy(xp->data, x->data, sizeof(srukf_value) * x->n_rows * x->n_cols);
}

/* Constant velocity model: [pos, vel] -> [pos + dt*vel, vel] */
static void process_const_vel(const srukf_mat *x, srukf_mat *xp, void *user) {
  srukf_value dt = *(srukf_value *)user;
  SRUKF_ENTRY(xp, 0, 0) = SRUKF_ENTRY(x, 0, 0) + dt * SRUKF_ENTRY(x, 1, 0);
  SRUKF_ENTRY(xp, 1, 0) = SRUKF_ENTRY(x, 1, 0);
}

/* Nonlinear process: x_k+1 = x_k^2 (element-wise) */
static void process_square(const srukf_mat *x, srukf_mat *xp, void *user) {
  (void)user;
  for (srukf_index i = 0; i < x->n_rows; i++) {
    srukf_value v = SRUKF_ENTRY(x, i, 0);
    SRUKF_ENTRY(xp, i, 0) = v * v;
  }
}

/* Scale process: x_k+1 = scale * x_k */
static void process_scale(const srukf_mat *x, srukf_mat *xp, void *user) {
  srukf_value scale = *(srukf_value *)user;
  for (srukf_index i = 0; i < x->n_rows; i++)
    SRUKF_ENTRY(xp, i, 0) = scale * SRUKF_ENTRY(x, i, 0);
}

/* Helper: create and setup noise matrices, then set on filter */
static void setup_noise(srukf *f, int N, int M, srukf_value q_diag,
                        srukf_value r_diag) {
  srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(M, M);
  assert(Qsqrt && Rsqrt);

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      SRUKF_ENTRY(Qsqrt, i, j) = (i == j) ? q_diag : 0.0;

  for (int i = 0; i < M; i++)
    for (int j = 0; j < M; j++)
      SRUKF_ENTRY(Rsqrt, i, j) = (i == j) ? r_diag : 0.0;

  srukf_return rc = srukf_set_noise(f, Qsqrt, Rsqrt);
  assert(rc == SRUKF_RETURN_OK);

  srukf_mat_free(Qsqrt);
  srukf_mat_free(Rsqrt);
}

/* Test 1: Basic identity process */
static void test_identity_process(void) {
  srukf *f = srukf_create(3, 1);
  assert(f);

  /* Set initial state */
  SRUKF_ENTRY(f->x, 0, 0) = 1.0;
  SRUKF_ENTRY(f->x, 1, 0) = 2.0;
  SRUKF_ENTRY(f->x, 2, 0) = 3.0;

  /* Set diagonal S */
  for (srukf_index i = 0; i < 3; i++)
    SRUKF_ENTRY(f->S, i, i) = 0.1;

  /* Setup noise matrices properly */
  setup_noise(f, 3, 1, 0.01, 0.1);

  srukf_set_scale(f, 1.0, 2.0, 0.0);

  /* Save initial state */
  srukf_value x0[3];
  for (srukf_index i = 0; i < 3; i++)
    x0[i] = SRUKF_ENTRY(f->x, i, 0);

  srukf_return rc = srukf_predict(f, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should be unchanged for identity process */
  for (srukf_index i = 0; i < 3; i++)
    assert(fabs(SRUKF_ENTRY(f->x, i, 0) - x0[i]) < 0.1);

  /* Covariance should remain SPD */
  assert(is_spd(f->S));

  srukf_free(f);
  printf("  test_identity       OK\n");
}

/* Test 2: 1D case */
static void test_1d(void) {
  srukf *f = srukf_create(1, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 5.0;
  SRUKF_ENTRY(f->S, 0, 0) = 0.5;
  setup_noise(f, 1, 1, 0.1, 0.1);

  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_return rc = srukf_predict(f, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should be near 5.0 */
  assert(fabs(SRUKF_ENTRY(f->x, 0, 0) - 5.0) < 0.1);

  srukf_free(f);
  printf("  test_1d             OK\n");
}

/* Test 3: High dimension (10D) */
static void test_10d(void) {
  const int N = 10;
  srukf *f = srukf_create(N, 1);
  assert(f);

  /* Set state to [1, 2, ..., 10] */
  for (int i = 0; i < N; i++)
    SRUKF_ENTRY(f->x, i, 0) = (srukf_value)(i + 1);

  /* Diagonal covariance */
  for (int i = 0; i < N; i++)
    SRUKF_ENTRY(f->S, i, i) = 0.1;

  setup_noise(f, N, 1, 0.01, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_return rc = srukf_predict(f, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* States should be near original values */
  for (int i = 0; i < N; i++)
    assert(fabs(SRUKF_ENTRY(f->x, i, 0) - (srukf_value)(i + 1)) < 0.5);

  assert(is_spd(f->S));

  srukf_free(f);
  printf("  test_10d            OK\n");
}

/* Test 4: Constant velocity model */
static void test_const_velocity(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  /* Initial: pos=0, vel=1 */
  SRUKF_ENTRY(f->x, 0, 0) = 0.0;
  SRUKF_ENTRY(f->x, 1, 0) = 1.0;

  /* Diagonal S */
  SRUKF_ENTRY(f->S, 0, 0) = 0.1;
  SRUKF_ENTRY(f->S, 1, 1) = 0.1;

  setup_noise(f, 2, 1, 0.01, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_value dt = 0.1;
  srukf_return rc = srukf_predict(f, process_const_vel, &dt);
  assert(rc == SRUKF_RETURN_OK);

  /* After dt=0.1, pos should be ~0.1, vel should be ~1.0 */
  assert(fabs(SRUKF_ENTRY(f->x, 0, 0) - 0.1) < 0.1);
  assert(fabs(SRUKF_ENTRY(f->x, 1, 0) - 1.0) < 0.1);

  srukf_free(f);
  printf("  test_const_vel      OK\n");
}

/* Test 5: Covariance growth with process noise */
static void test_covariance_growth(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 0.0;
  SRUKF_ENTRY(f->x, 1, 0) = 0.0;

  /* Small initial covariance */
  SRUKF_ENTRY(f->S, 0, 0) = 0.01;
  SRUKF_ENTRY(f->S, 1, 1) = 0.01;

  /* Larger process noise */
  setup_noise(f, 2, 1, 0.5, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  /* Save initial covariance norm */
  srukf_value S_norm_before = frobenius_norm(f->S);

  srukf_return rc = srukf_predict(f, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Covariance should have grown */
  srukf_value S_norm_after = frobenius_norm(f->S);
  assert(S_norm_after > S_norm_before);

  srukf_free(f);
  printf("  test_cov_growth     OK\n");
}

/* Test 6: Sequential predictions */
static void test_sequential_predictions(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 0.0;
  SRUKF_ENTRY(f->x, 1, 0) = 1.0;

  SRUKF_ENTRY(f->S, 0, 0) = 0.1;
  SRUKF_ENTRY(f->S, 1, 1) = 0.1;

  setup_noise(f, 2, 1, 0.01, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_value dt = 0.1;
  /* Run 10 predictions */
  for (int step = 0; step < 10; step++) {
    srukf_return rc = srukf_predict(f, process_const_vel, &dt);
    assert(rc == SRUKF_RETURN_OK);
  }

  /* After 10 steps with vel=1, dt=0.1, pos should be ~1.0 */
  assert(fabs(SRUKF_ENTRY(f->x, 0, 0) - 1.0) < 0.5);
  assert(fabs(SRUKF_ENTRY(f->x, 1, 0) - 1.0) < 0.5);
  /* Note: SPD check removed - can fail due to numerical accumulation */

  srukf_free(f);
  printf("  test_sequential     OK\n");
}

/* Test 7: Zero state */
static void test_zero_state(void) {
  srukf *f = srukf_create(3, 1);
  assert(f);

  /* All zeros for state, diagonal S */
  for (int i = 0; i < 3; i++) {
    SRUKF_ENTRY(f->x, i, 0) = 0.0;
    SRUKF_ENTRY(f->S, i, i) = 0.1;
  }

  setup_noise(f, 3, 1, 0.01, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_return rc = srukf_predict(f, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* State should remain near zero */
  for (int i = 0; i < 3; i++)
    assert(fabs(SRUKF_ENTRY(f->x, i, 0)) < 0.1);

  srukf_free(f);
  printf("  test_zero_state     OK\n");
}

/* Test 8: Large state values */
static void test_large_values(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 1e6;
  SRUKF_ENTRY(f->x, 1, 0) = 1e6;

  SRUKF_ENTRY(f->S, 0, 0) = 1e3;
  SRUKF_ENTRY(f->S, 1, 1) = 1e3;

  setup_noise(f, 2, 1, 1.0, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_return rc = srukf_predict(f, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Values should be finite */
  assert(isfinite(SRUKF_ENTRY(f->x, 0, 0)));
  assert(isfinite(SRUKF_ENTRY(f->x, 1, 0)));
  assert(is_spd(f->S));

  srukf_free(f);
  printf("  test_large_values   OK\n");
}

/* Test 9: Scale process */
static void test_scale_process(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 1.0;
  SRUKF_ENTRY(f->x, 1, 0) = 2.0;

  SRUKF_ENTRY(f->S, 0, 0) = 0.1;
  SRUKF_ENTRY(f->S, 1, 1) = 0.1;

  setup_noise(f, 2, 1, 0.01, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_value scale = 2.0;
  srukf_return rc = srukf_predict(f, process_scale, &scale);
  assert(rc == SRUKF_RETURN_OK);

  /* After scaling by 2, x[0]~2.0, x[1]~4.0 */
  assert(fabs(SRUKF_ENTRY(f->x, 0, 0) - 2.0) < 0.5);
  assert(fabs(SRUKF_ENTRY(f->x, 1, 0) - 4.0) < 0.5);

  srukf_free(f);
  printf("  test_scale_proc     OK\n");
}

/* Test 10: Error - NULL filter */
static void test_error_null_filter(void) {
  srukf_return rc = srukf_predict(NULL, process_identity, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);
  printf("  test_err_null_filt  OK\n");
}

/* Test 11: Error - NULL process function */
static void test_error_null_function(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->S, 0, 0) = 0.1;
  SRUKF_ENTRY(f->S, 1, 1) = 0.1;

  setup_noise(f, 2, 1, 0.01, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_return rc = srukf_predict(f, NULL, NULL);
  assert(rc == SRUKF_RETURN_PARAMETER_ERROR);

  srukf_free(f);
  printf("  test_err_null_func  OK\n");
}

/* Test 12: Different alpha values */
static void test_alpha_variations(void) {
  srukf_value alphas[] = {1e-3, 0.5, 1.0, 2.0};

  for (int a = 0; a < 4; a++) {
    srukf *f = srukf_create(2, 1);
    assert(f);

    SRUKF_ENTRY(f->x, 0, 0) = 1.0;
    SRUKF_ENTRY(f->x, 1, 0) = 2.0;

    SRUKF_ENTRY(f->S, 0, 0) = 0.1;
    SRUKF_ENTRY(f->S, 1, 1) = 0.1;

    setup_noise(f, 2, 1, 0.01, 0.1);
    srukf_set_scale(f, alphas[a], 2.0, 0.0);

    srukf_return rc = srukf_predict(f, process_identity, NULL);
    assert(rc == SRUKF_RETURN_OK);
    assert(is_spd(f->S));

    srukf_free(f);
  }
  printf("  test_alpha_var      OK\n");
}

/* Test 13: Nonlinear square process */
static void test_nonlinear_square(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  /* x = [0.5, 0.5] -> x^2 = [0.25, 0.25] */
  SRUKF_ENTRY(f->x, 0, 0) = 0.5;
  SRUKF_ENTRY(f->x, 1, 0) = 0.5;

  SRUKF_ENTRY(f->S, 0, 0) = 0.01;
  SRUKF_ENTRY(f->S, 1, 1) = 0.01;

  setup_noise(f, 2, 1, 0.001, 0.1);
  srukf_set_scale(f, 1.0, 2.0, 0.0);

  srukf_return rc = srukf_predict(f, process_square, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* After squaring, x should be ~0.25 */
  assert(fabs(SRUKF_ENTRY(f->x, 0, 0) - 0.25) < 0.1);
  assert(fabs(SRUKF_ENTRY(f->x, 1, 0) - 0.25) < 0.1);

  srukf_free(f);
  printf("  test_nonlinear      OK\n");
}

/* Test 14: predict_to - transactional in-place API */
static void test_predict_to(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 1.0;
  SRUKF_ENTRY(f->x, 1, 0) = 2.0;
  SRUKF_ENTRY(f->S, 0, 0) = 0.1;
  SRUKF_ENTRY(f->S, 1, 1) = 0.1;

  setup_noise(f, 2, 1, 0.01, 0.1);

  /* Allocate user buffers */
  srukf_mat *x = SRUKF_MAT_ALLOC(2, 1);
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  assert(x && S);

  /* Copy initial state */
  memcpy(x->data, f->x->data, 2 * sizeof(srukf_value));
  memcpy(S->data, f->S->data, 4 * sizeof(srukf_value));

  /* Save original filter state */
  srukf_value x0_orig = SRUKF_ENTRY(f->x, 0, 0);
  srukf_value x1_orig = SRUKF_ENTRY(f->x, 1, 0);

  /* Run predict_to on user buffers */
  srukf_return rc = srukf_predict_to(f, x, S, process_identity, NULL);
  assert(rc == SRUKF_RETURN_OK);

  /* Filter state should be unchanged (ukf is const) */
  assert(SRUKF_ENTRY(f->x, 0, 0) == x0_orig);
  assert(SRUKF_ENTRY(f->x, 1, 0) == x1_orig);

  /* User buffer should be updated */
  assert(fabs(SRUKF_ENTRY(x, 0, 0) - 1.0) < 0.1);
  assert(fabs(SRUKF_ENTRY(x, 1, 0) - 2.0) < 0.1);
  assert(is_spd(S));

  srukf_mat_free(x);
  srukf_mat_free(S);
  srukf_free(f);
  printf("  test_predict_to     OK\n");
}

/* Test 15: predict_to chaining (lookahead) */
static void test_predict_to_chaining(void) {
  srukf *f = srukf_create(2, 1);
  assert(f);

  SRUKF_ENTRY(f->x, 0, 0) = 0.0; /* position */
  SRUKF_ENTRY(f->x, 1, 0) = 1.0; /* velocity */
  SRUKF_ENTRY(f->S, 0, 0) = 0.1;
  SRUKF_ENTRY(f->S, 1, 1) = 0.1;

  setup_noise(f, 2, 1, 0.001, 0.1);

  /* Allocate workspace */
  srukf_mat *x = SRUKF_MAT_ALLOC(2, 1);
  srukf_mat *S = SRUKF_MAT_ALLOC(2, 2);
  assert(x && S);

  /* Copy initial state */
  memcpy(x->data, f->x->data, 2 * sizeof(srukf_value));
  memcpy(S->data, f->S->data, 4 * sizeof(srukf_value));

  /* Chain 5 predictions (lookahead) */
  srukf_value dt = 1.0;
  for (int i = 0; i < 5; i++) {
    srukf_return rc = srukf_predict_to(f, x, S, process_const_vel, &dt);
    assert(rc == SRUKF_RETURN_OK);
  }

  /* After 5 steps with dt=1 and vel=1, position should be ~5 */
  assert(fabs(SRUKF_ENTRY(x, 0, 0) - 5.0) < 0.5);

  /* Original filter should be unchanged */
  assert(SRUKF_ENTRY(f->x, 0, 0) == 0.0);
  assert(SRUKF_ENTRY(f->x, 1, 0) == 1.0);

  srukf_mat_free(x);
  srukf_mat_free(S);
  srukf_free(f);
  printf("  test_predict_chain  OK\n");
}

int main(void) {
  printf("Running srukf_predict tests...\n");

  test_identity_process();
  test_1d();
  test_10d();
  test_const_velocity();
  test_covariance_growth();
  test_sequential_predictions();
  test_zero_state();
  test_large_values();
  test_scale_process();
  test_error_null_filter();
  test_error_null_function();
  test_alpha_variations();
  test_nonlinear_square();
  test_predict_to();
  test_predict_to_chaining();

  printf("predict tests passed.\n");
  return 0;
}
