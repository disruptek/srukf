/* prelude.c -- includes, precision macros, version, filter struct
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

/**
 * @file srukf.c
 * @brief Square-Root Unscented Kalman Filter Implementation
 *
 * This file contains the complete implementation of the SR-UKF algorithm.
 * The code is organized into several sections:
 *
 * 1. **Matrix utilities** - Basic allocation and helper functions
 * 2. **Workspace management** - Pre-allocated temporaries for efficiency
 * 3. **Weight computation** - UKF sigma point weights
 * 4. **Sigma point generation** - Creating the 2N+1 sample points
 * 5. **SR-UKF core operations** - QR-based covariance updates, Cholesky
 * downdates
 * 6. **Predict/Correct** - The main filter operations
 *
 * @section impl_numerical Numerical Considerations
 *
 * The SR-UKF differs from the standard UKF in how covariance is maintained:
 *
 * **Standard UKF:**
 * @code
 * P = sum(wc[i] * (X[i] - x_mean) * (X[i] - x_mean)') + Q
 * @endcode
 * This requires ensuring P remains positive-definite, which can fail
 * due to numerical errors.
 *
 * **Square-Root UKF:**
 * @code
 * S = qr([sqrt(wc) * (X - x_mean)' ; Qsqrt'])'
 * @endcode
 * We never form P explicitly. Instead, we maintain S where P = S*S'.
 * The QR decomposition guarantees S is a valid Cholesky factor.
 *
 * @section impl_negative_w Handling Negative Weights
 *
 * For small alpha (< 1), the zeroth covariance weight wc[0] can be negative:
 * @code
 * wc[0] = lambda/(N+lambda) + (1 - alpha^2 + beta)
 * @endcode
 *
 * If lambda ≈ -N (which happens for small alpha), wc[0] becomes negative.
 * We handle this by:
 * 1. Excluding the zeroth deviation from the QR (which computes S² = sum of
 * squares)
 * 2. Applying a Cholesky rank-1 downdate: S² → S² - dev0 * dev0'
 *
 * The downdate uses Givens rotations for numerical stability.
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

#include "srukf.h"

/** @brief BLAS layout constant */
#define SRUKF_CBLAS_LAYOUT CblasColMajor
/** @brief LAPACK layout constant */
#define SRUKF_LAPACK_LAYOUT LAPACK_COL_MAJOR

/* BLAS/LAPACK routine and math-function selection based on precision.
 * The _work QR variant takes a caller-owned scratch buffer, keeping the
 * predict/correct hot path free of heap allocation. */
#ifdef SRUKF_SINGLE
#define SRUKF_GEMM  cblas_sgemm         /**< Matrix multiply (single) */
#define SRUKF_GEMV  cblas_sgemv         /**< Matrix-vector multiply (single) */
#define SRUKF_TRSM  cblas_strsm         /**< Triangular solve (single) */
#define SRUKF_TRSV  cblas_strsv         /**< Triangular vector solve (single) */
#define SRUKF_GEQRF LAPACKE_sgeqrf_work /**< QR factorization (single) */
#define SRUKF_POTRF LAPACKE_spotrf      /**< Cholesky factorization (single) */
#define SRUKF_SQRT  sqrtf
#define SRUKF_FABS  fabsf
#else
#define SRUKF_GEMM  cblas_dgemm         /**< Matrix multiply (double) */
#define SRUKF_GEMV  cblas_dgemv         /**< Matrix-vector multiply (double) */
#define SRUKF_TRSM  cblas_dtrsm         /**< Triangular solve (double) */
#define SRUKF_TRSV  cblas_dtrsv         /**< Triangular vector solve (double) */
#define SRUKF_GEQRF LAPACKE_dgeqrf_work /**< QR factorization (double) */
#define SRUKF_POTRF LAPACKE_dpotrf      /**< Cholesky factorization (double) */
#define SRUKF_SQRT  sqrt
#define SRUKF_FABS  fabs
#endif

/** @brief Default sigma point spread parameter */
#define DEFAULT_ALPHA 1e-3
/** @brief Default prior distribution parameter (optimal for Gaussian) */
#define DEFAULT_BETA 2.0
/** @brief Default secondary scaling parameter (keeps N + kappa > 0 for
 * every N, unlike the textbook 3 - N) */
#define DEFAULT_KAPPA 1.0
/** @brief Initial sqrt-covariance: S = DEFAULT_INIT_STD * I on creation,
 * so a correct step can run before any predict/reset. Documented in
 * srukf_create(); srukf_reset() replaces it. */
#define DEFAULT_INIT_STD 0.001
/** @brief Numerical tolerance for near-zero checks, scaled to the
 * working precision: values below this are treated as zero. */
#ifdef SRUKF_SINGLE
#define SRUKF_EPS 1e-6f
#else
#define SRUKF_EPS 1e-12
#endif

const char *srukf_version(void) {
  return SRUKF_VERSION;
}

/*============================================================================
 * @internal
 * @defgroup impl_struct Filter Structure
 * @brief The filter instance (opaque in the public header)
 * @{
 *============================================================================*/

/**
 * @brief Square-Root Unscented Kalman Filter instance
 *
 * Opaque to consumers (see srukf.h); all access goes through accessors,
 * keeping the ABI stable across releases.
 */
struct srukf {
  /* State estimate */
  srukf_mat *x; /**< State estimate vector (N x 1) */
  srukf_mat *S; /**< Sqrt of state covariance (N x N, lower triangular).
                     Satisfies P = S*S' where P is the covariance. */

  /* Noise covariances, stored as square-roots */
  srukf_mat *Qsqrt; /**< Sqrt of process noise covariance (N x N).
                         Larger values = less trust in the model. */
  srukf_mat *Rsqrt; /**< Sqrt of measurement noise covariance (M x M).
                         Larger values = less trust in measurements. */

  /* UKF tuning parameters */
  srukf_value alpha;  /**< Spread of sigma points (typically 1e-3 to 1). */
  srukf_value beta;   /**< Prior distribution knowledge (2 for Gaussian). */
  srukf_value kappa;  /**< Secondary scaling (typically 0 or 3-N). */
  srukf_value lambda; /**< Derived: alpha^2 * (N + kappa) - N. */

  /* Sigma point weights (2N+1 elements each):
   *   wm[0] = lambda/(N+lambda),        wm[i>0] = 1/(2(N+lambda))
   *   wc[0] = wm[0] + (1 - alpha^2 + beta), wc[i>0] = wm[i]
   * wc[0] can be negative for small alpha (handled via downdates). */
  srukf_value *wm; /**< Mean weights */
  srukf_value *wc; /**< Covariance weights */

  srukf_workspace *ws; /**< Pre-allocated workspace (allocated on demand) */

  /* Custom mean/residual hooks for non-Euclidean spaces (see
   * srukf_set_state_ops() / srukf_set_meas_ops()); NULL entries fall
   * back to the Euclidean defaults. */
  srukf_mean_fn state_mean_fn;
  srukf_residual_fn state_residual_fn;
  void *state_ops_ctx;
  srukf_mean_fn meas_mean_fn;
  srukf_residual_fn meas_residual_fn;
  void *meas_ops_ctx;

  /* Per-instance diagnostic handler (see srukf_set_diag()); when NULL,
   * the global callback (if any) is used instead. */
  void (*diag_fn)(const char *msg, void *ctx);
  void *diag_ctx;
};

/** @} */ /* end impl_struct */
