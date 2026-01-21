/* --------------------------------------------------------------------
 * 03_gain.c - Kalman gain computation tests
 *
 * Tests compute_kalman_gain with various dimensions and
 * numerical conditions.
 *
 * This test includes sr_ukf.c directly to access internal functions.
 * -------------------------------------------------------------------- */

#include "sr_ukf.c"

#define EPS 1e-10

/* helper: create a column‑major matrix with given values */
static lah_mat *make_matrix(lah_index nR, lah_index nC, const lah_value *data) {
  lah_mat *m = allocMatrixNow(nR, nC);
  assert(m);
  for (lah_index i = 0; i < nR; ++i)
    for (lah_index j = 0; j < nC; ++j)
      LAH_ENTRY(m, i, j) = data[i + j * nR];
  return m;
}

/* helper: create zero matrix */
static lah_mat *make_zero_matrix(lah_index nR, lah_index nC) {
  lah_mat *m = allocMatrixNow(nR, nC);
  assert(m);
  for (lah_index i = 0; i < nR; ++i)
    for (lah_index j = 0; j < nC; ++j)
      LAH_ENTRY(m, i, j) = 0.0;
  return m;
}

/* helper: create identity matrix */
static lah_mat *make_identity(lah_index n) {
  lah_mat *m = allocMatrixNow(n, n);
  assert(m);
  for (lah_index i = 0; i < n; ++i)
    for (lah_index j = 0; j < n; ++j)
      LAH_ENTRY(m, i, j) = (i == j) ? 1.0 : 0.0;
  return m;
}

/* helper: create diagonal matrix */
static lah_mat *make_diagonal(lah_index n, const lah_value *diag) {
  lah_mat *m = allocMatrixNow(n, n);
  assert(m);
  for (lah_index i = 0; i < n; ++i)
    for (lah_index j = 0; j < n; ++j)
      LAH_ENTRY(m, i, j) = (i == j) ? diag[i] : 0.0;
  return m;
}

/* helper: compare two matrices with a tolerance */
static void assert_mat_eq(const lah_mat *A, const lah_mat *B, lah_value eps) {
  assert(A && B);
  assert(A->nR == B->nR && A->nC == B->nC);
  for (lah_index i = 0; i < A->nR; ++i)
    for (lah_index j = 0; j < A->nC; ++j) {
      lah_value diff = fabs(LAH_ENTRY(A, i, j) - LAH_ENTRY(B, i, j));
      if (diff > eps) {
        fprintf(stderr, "Mismatch at (%zu,%zu): %g vs %g (diff %g)\n",
                (size_t)i, (size_t)j, (double)LAH_ENTRY(A, i, j),
                (double)LAH_ENTRY(B, i, j), (double)diff);
        assert(0);
      }
    }
}

/* -------- 1×1 case --------------------------------------------------- */
static void test_1x1(void) {
  lah_value pxz_data[1] = {3.0};
  lah_value pyy_data[1] = {2.0};
  lah_value k_expected[1] = {1.5}; /* 3/2 */

  lah_mat *Pxz = make_matrix(1, 1, pxz_data);
  lah_mat *Pyy = make_matrix(1, 1, pyy_data);
  lah_mat *K = make_zero_matrix(1, 1);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);
  assert_mat_eq(K, make_matrix(1, 1, k_expected), EPS);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  1×1 case           OK\n");
}

/* -------- 2×2 case --------------------------------------------------- */
static void test_2x2(void) {
  lah_value pxz_data[4] = {1.0, 2.0, 3.0, 4.0};
  lah_value pyy_data[4] = {4.0, 1.0, 1.0, 2.0}; /* det = 7 */

  /* K = Pxz * inv(Pyy), inv = [[2/7, -1/7], [-1/7, 4/7]] */
  lah_value k_expected[4] = {
      1.0 * (2.0 / 7.0) + 3.0 * (-1.0 / 7.0), /* -1/7 */
      2.0 * (2.0 / 7.0) + 4.0 * (-1.0 / 7.0), /* 0 */
      1.0 * (-1.0 / 7.0) + 3.0 * (4.0 / 7.0), /* 11/7 */
      2.0 * (-1.0 / 7.0) + 4.0 * (4.0 / 7.0)  /* 2 */
  };

  lah_mat *Pxz = make_matrix(2, 2, pxz_data);
  lah_mat *Pyy = make_matrix(2, 2, pyy_data);
  lah_mat *K = make_zero_matrix(2, 2);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);
  assert_mat_eq(K, make_matrix(2, 2, k_expected), EPS);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  2×2 case           OK\n");
}

/* -------- 3×3 diagonal case ------------------------------------------- */
static void test_3x3_diagonal(void) {
  lah_value pxz_data[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};
  lah_value pyy_diag[3] = {2.0, 3.0, 4.0};

  /* K = Pxz * diag(1/2, 1/3, 1/4) */
  lah_value k_expected[9] = {1.0 / 2.0, 4.0 / 2.0, 7.0 / 2.0,
                             2.0 / 3.0, 5.0 / 3.0, 8.0 / 3.0,
                             3.0 / 4.0, 6.0 / 4.0, 9.0 / 4.0};

  lah_mat *Pxz = make_matrix(3, 3, pxz_data);
  lah_mat *Pyy = make_diagonal(3, pyy_diag);
  lah_mat *K = make_zero_matrix(3, 3);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);
  assert_mat_eq(K, make_matrix(3, 3, k_expected), EPS);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  3×3 diagonal       OK\n");
}

/* -------- Rectangular: N=3, M=2 (more states than measurements) ------- */
static void test_rectangular_N_gt_M(void) {
  /* Pxz: 3×2, Pyy: 2×2, K: 3×2 */
  lah_value pxz_data[6] = {1, 2, 3, 4, 5, 6}; /* column-major */
  lah_value pyy_diag[2] = {2.0, 4.0};

  /* K = Pxz * diag(0.5, 0.25) */
  lah_value k_expected[6] = {
      0.5, 1.0,  1.5, /* first column scaled by 0.5 */
      1.0, 1.25, 1.5  /* second column scaled by 0.25 */
  };

  lah_mat *Pxz = make_matrix(3, 2, pxz_data);
  lah_mat *Pyy = make_diagonal(2, pyy_diag);
  lah_mat *K = make_zero_matrix(3, 2);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);
  assert_mat_eq(K, make_matrix(3, 2, k_expected), EPS);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  rectangular N>M    OK\n");
}

/* -------- Rectangular: N=2, M=4 (more measurements than states) ------- */
static void test_rectangular_N_lt_M(void) {
  /* Pxz: 2×4, Pyy: 4×4, K: 2×4 */
  lah_value pxz_data[8] = {1, 2, 3, 4,
                           5, 6, 7, 8}; /* column-major: 2 rows, 4 cols */
  lah_value pyy_diag[4] = {1.0, 2.0, 4.0, 8.0};

  /* K = Pxz * diag(1, 0.5, 0.25, 0.125) */
  lah_value k_expected[8] = {
      1.0,   2.0, /* col 1: ×1 */
      1.5,   2.0, /* col 2: ×0.5 */
      1.25,  1.5, /* col 3: ×0.25 */
      0.875, 1.0  /* col 4: ×0.125 */
  };

  lah_mat *Pxz = make_matrix(2, 4, pxz_data);
  lah_mat *Pyy = make_diagonal(4, pyy_diag);
  lah_mat *K = make_zero_matrix(2, 4);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);
  assert_mat_eq(K, make_matrix(2, 4, k_expected), EPS);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  rectangular N<M    OK\n");
}

/* -------- Identity Pyy: K = Pxz ---------------------------------------- */
static void test_identity_pyy(void) {
  lah_value pxz_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  lah_mat *Pxz = make_matrix(3, 3, pxz_data);
  lah_mat *Pyy = make_identity(3);
  lah_mat *K = make_zero_matrix(3, 3);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);
  assert_mat_eq(K, Pxz, EPS); /* K should equal Pxz */

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  identity Pyy       OK\n");
}

/* -------- Well-conditioned 5×5 case ----------------------------------- */
static void test_5x5(void) {
  /* Use diagonal for easy verification */
  lah_value pyy_diag[5] = {1.0, 2.0, 3.0, 4.0, 5.0};

  lah_mat *Pxz = make_zero_matrix(5, 5);
  for (lah_index i = 0; i < 5; i++)
    for (lah_index j = 0; j < 5; j++)
      LAH_ENTRY(Pxz, i, j) = (lah_value)((i + 1) * (j + 1));

  lah_mat *Pyy = make_diagonal(5, pyy_diag);
  lah_mat *K = make_zero_matrix(5, 5);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);

  /* Verify K = Pxz * inv(Pyy) = Pxz * diag(1, 0.5, 0.333, 0.25, 0.2) */
  for (lah_index i = 0; i < 5; i++) {
    for (lah_index j = 0; j < 5; j++) {
      lah_value expected = (lah_value)((i + 1) * (j + 1)) / pyy_diag[j];
      assert(fabs(LAH_ENTRY(K, i, j) - expected) < EPS);
    }
  }

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  5×5 case           OK\n");
}

/* -------- Near-singular but invertible -------------------------------- */
static void test_near_singular(void) {
  /* Pyy with small but non-zero determinant */
  lah_value pyy_diag[2] = {1.0, 1e-8}; /* condition number ~1e8 */

  lah_value pxz_data[4] = {1.0, 0.0, 0.0, 1.0};

  lah_mat *Pxz = make_matrix(2, 2, pxz_data);
  lah_mat *Pyy = make_diagonal(2, pyy_diag);
  lah_mat *K = make_zero_matrix(2, 2);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);

  /* K = diag(1, 1e8) */
  assert(fabs(LAH_ENTRY(K, 0, 0) - 1.0) < EPS);
  assert(fabs(LAH_ENTRY(K, 1, 1) - 1e8) <
         1.0); /* relaxed tolerance for large value */

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  near-singular      OK\n");
}

/* -------- Small magnitude matrices ------------------------------------ */
static void test_small_magnitude(void) {
  lah_value pxz_data[4] = {1e-12, 2e-12, 3e-12, 4e-12};
  lah_value pyy_data[4] = {1e-10, 0, 0, 1e-10};

  lah_mat *Pxz = make_matrix(2, 2, pxz_data);
  lah_mat *Pyy = make_matrix(2, 2, pyy_data);
  lah_mat *K = make_zero_matrix(2, 2);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnOk);

  /* K = Pxz * inv(diag(1e-10, 1e-10)) = Pxz * 1e10 */
  assert(fabs(LAH_ENTRY(K, 0, 0) - 1e-2) < 1e-14);
  assert(fabs(LAH_ENTRY(K, 1, 0) - 2e-2) < 1e-14);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  small magnitude    OK\n");
}

/* -------- Singular Pyy – should return error -------------------------- */
static void test_singular_pyy(void) {
  lah_value pxz_data[4] = {1, 0, 0, 1};
  lah_value pyy_data[4] = {1, 2, 2, 4}; /* rank-1, det=0 */

  lah_mat *Pxz = make_matrix(2, 2, pxz_data);
  lah_mat *Pyy = make_matrix(2, 2, pyy_data);
  lah_mat *K = make_zero_matrix(2, 2);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnMathError); /* POTRF fails: not positive definite */

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  singular Pyy       OK\n");
}

/* -------- Wrong K dimensions ------------------------------------------ */
static void test_wrong_K_dimensions(void) {
  lah_mat *Pxz = make_identity(2);
  lah_mat *Pyy = make_identity(2);
  lah_mat *K_wrong;
  lah_Return rc;

  /* K wrong rows */
  K_wrong = make_zero_matrix(1, 2);
  rc = compute_kalman_gain(Pxz, Pyy, K_wrong);
  assert(rc == lahReturnParameterError);
  lah_matFree(K_wrong);

  /* K wrong columns */
  K_wrong = make_zero_matrix(2, 1);
  rc = compute_kalman_gain(Pxz, Pyy, K_wrong);
  assert(rc == lahReturnParameterError);
  lah_matFree(K_wrong);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  printf("  wrong K dims       OK\n");
}

/* -------- NULL pointer tests ------------------------------------------ */
static void test_null_pointers(void) {
  lah_mat *Pxz = make_identity(2);
  lah_mat *Pyy = make_identity(2);
  lah_mat *K = make_zero_matrix(2, 2);

  assert(compute_kalman_gain(NULL, Pyy, K) == lahReturnParameterError);
  assert(compute_kalman_gain(Pxz, NULL, K) == lahReturnParameterError);
  assert(compute_kalman_gain(Pxz, Pyy, NULL) == lahReturnParameterError);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  NULL pointers      OK\n");
}

/* -------- Dimension mismatch between Pxz and Pyy ---------------------- */
static void test_dimension_mismatch(void) {
  /* Pxz is 2×3 but Pyy is 2×2 (should be 3×3) */
  lah_mat *Pxz = make_zero_matrix(2, 3);
  lah_mat *Pyy = make_identity(2);
  lah_mat *K = make_zero_matrix(2, 3);

  lah_Return rc = compute_kalman_gain(Pxz, Pyy, K);
  assert(rc == lahReturnParameterError);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  printf("  dim mismatch       OK\n");
}

/* -------- Input immutability ------------------------------------------ */
static void test_input_immutability(void) {
  lah_value pxz_data[4] = {1.0, 2.0, 3.0, 4.0};
  lah_value pyy_data[4] = {2.0, 0.0, 0.0, 2.0};

  lah_mat *Pxz = make_matrix(2, 2, pxz_data);
  lah_mat *Pyy = make_matrix(2, 2, pyy_data);
  lah_mat *K = make_zero_matrix(2, 2);

  /* Save copies */
  lah_mat *Pxz_copy = make_matrix(2, 2, pxz_data);
  lah_mat *Pyy_copy = make_matrix(2, 2, pyy_data);

  compute_kalman_gain(Pxz, Pyy, K);

  /* Verify inputs unchanged */
  assert_mat_eq(Pxz, Pxz_copy, EPS);
  assert_mat_eq(Pyy, Pyy_copy, EPS);

  lah_matFree(Pxz);
  lah_matFree(Pyy);
  lah_matFree(K);
  lah_matFree(Pxz_copy);
  lah_matFree(Pyy_copy);
  printf("  input immutable    OK\n");
}

/* -------- main ------------------------------------------------------- */
int main(void) {
  printf("Running compute_kalman_gain tests...\n");

  test_1x1();
  test_2x2();
  test_3x3_diagonal();
  test_rectangular_N_gt_M();
  test_rectangular_N_lt_M();
  test_identity_pyy();
  test_5x5();
  test_near_singular();
  test_small_magnitude();
  test_singular_pyy();
  test_wrong_K_dimensions();
  test_null_pointers();
  test_dimension_mismatch();
  test_input_immutability();

  printf("gain tests passed.\n");
  return 0;
}
