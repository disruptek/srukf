#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

/*
 * Test helper functions for srukf tests.
 * Include this header after srukf.h in test files.
 */

#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "srukf.h"

#define TEST_EPS 1e-12

/* Check if matrix A is symmetric (to given precision in significant digits). */
static inline bool is_symmetric(const srukf_mat *A, int precision) {
  if (!A || A->n_rows != A->n_cols)
    return false;

  srukf_value epsilon = pow(10.0, -precision);
  for (srukf_index j = 0; j < A->n_cols; ++j) {
    for (srukf_index i = j + 1; i < A->n_rows; ++i) {
      if (fabs(SRUKF_ENTRY(A, i, j) - SRUKF_ENTRY(A, j, i)) > epsilon)
        return false;
    }
  }
  return true;
}

/* Cholesky factorization: P = L * L' where L is lower triangular.
 * Overwrites P with L. Returns true on success, false if not SPD.
 * Uses LAPACK POTRF. */
static inline bool cholesky(srukf_mat *P) {
  if (!P || P->n_rows != P->n_cols)
    return false;

  int info = SRUKF_POTRF(SRUKF_LAPACK_LAYOUT, 'L', (int)P->n_rows, P->data,
                         (int)SRUKF_LEADING_DIM(P));
  return (info == 0);
}

/* Check if matrix A is symmetric positive definite (SPD). */
static inline bool is_spd(const srukf_mat *A) {
  /* 1. Basic sanity checks */
  if (!A)
    return false;
  if (A->n_rows != A->n_cols)
    return false; /* must be square */

  /* 2. Symmetry test - 4 significant digits is usually sufficient. */
  if (!is_symmetric(A, 4))
    return false;

  /* 3. Copy the matrix (cholesky destroys its argument). */
  srukf_mat *tmp = SRUKF_MAT_ALLOC(A->n_rows, A->n_cols);
  if (!tmp)
    return false; /* out-of-memory */
  memcpy(tmp->data, A->data, (size_t)(A->n_cols * A->n_rows) * sizeof(srukf_value));

  /* tiny jitter to guard against round-off that might make a
   * positive-definite matrix look singular. */
  for (srukf_index i = 0; i < tmp->n_rows; ++i)
    SRUKF_ENTRY(tmp, i, i) += TEST_EPS;

  /* 4. Cholesky - success => SPD, failure => not SPD. */
  bool ok = cholesky(tmp);

  srukf_mat_free(tmp); /* release copy */
  return ok;
}

/* Check if a square-root factor S produces a valid SPD covariance P = S*S'.
 * This is the correct check for triangular Cholesky factors. */
static inline bool is_sqrt_valid(const srukf_mat *S) {
  if (!S || S->n_rows != S->n_cols)
    return false;

  srukf_index n = S->n_rows;
  srukf_mat *P = SRUKF_MAT_ALLOC(n, n);
  if (!P)
    return false;

  /* Compute P = S * S' */
  SRUKF_GEMM(SRUKF_CBLAS_LAYOUT, CblasNoTrans, CblasTrans, (int)n, (int)n, (int)n,
             (srukf_value)1.0, S->data, (int)SRUKF_LEADING_DIM(S), S->data,
             (int)SRUKF_LEADING_DIM(S), (srukf_value)0.0, P->data,
             (int)SRUKF_LEADING_DIM(P));

  bool ok = is_spd(P);
  srukf_mat_free(P);
  return ok;
}

/* Compute square-root covariance to full covariance matrix: P = S * S' */
static inline srukf_return sqrt_to_covariance(const srukf_mat *S, srukf_mat *P) {
  if (!S || !P)
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (S->n_rows != S->n_cols || P->n_rows != P->n_cols)
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (S->n_rows != P->n_rows)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Compute P = S * S' */
  int n = (int)S->n_rows;
  SRUKF_GEMM(SRUKF_CBLAS_LAYOUT, CblasNoTrans, CblasTrans, n, n, n,
             (srukf_value)1.0, S->data, (int)SRUKF_LEADING_DIM(S), S->data,
             (int)SRUKF_LEADING_DIM(S), (srukf_value)0.0, P->data,
             (int)SRUKF_LEADING_DIM(P));

  return SRUKF_RETURN_OK;
}

#endif /* TEST_HELPERS_H */
