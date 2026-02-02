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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "srukf.h"

/** @brief Default sigma point spread parameter */
#define DEFAULT_ALPHA 1e-3
/** @brief Default prior distribution parameter (optimal for Gaussian) */
#define DEFAULT_BETA 2.0
/** @brief Default secondary scaling parameter */
#define DEFAULT_KAPPA 1.0
/** @brief Numerical tolerance for near-zero checks */
#define SRUKF_EPS 1e-12

/*============================================================================
 * @internal
 * @defgroup impl_matrix Matrix Allocation
 * @brief Internal matrix utilities (derived from LAH by maj0e, MIT License)
 * @{
 *============================================================================*/

srukf_mat *srukf_mat_alloc(srukf_index rows, srukf_index cols, int alloc_data) {
  srukf_mat *mat = (srukf_mat *)calloc(1, sizeof(srukf_mat));
  if (!mat)
    return NULL;

  mat->n_rows = rows;
  mat->n_cols = cols;
  mat->inc_row = 1; /* column-major */
  mat->inc_col = rows;
  mat->type = SRUKF_TYPE_COL_MAJOR;

  if (cols == 1)
    SRUKF_SET_TYPE(mat, SRUKF_TYPE_VECTOR);
  if (rows == cols)
    SRUKF_SET_TYPE(mat, SRUKF_TYPE_SQUARE);

  if (alloc_data) {
    mat->data = (srukf_value *)calloc(rows * cols, sizeof(srukf_value));
    if (!mat->data) {
      free(mat);
      return NULL;
    }
  } else {
    mat->data = NULL;
    SRUKF_SET_TYPE(mat, SRUKF_TYPE_NO_DATA);
  }

  return mat;
}

void srukf_mat_free(srukf_mat *mat) {
  if (!mat)
    return;
  if (!SRUKF_IS_TYPE(mat, SRUKF_TYPE_NO_DATA))
    free(mat->data);
  free(mat);
}

/** @} */ /* end impl_matrix */

/*============================================================================
 * @internal
 * @defgroup impl_diag Diagnostics
 * @brief Internal diagnostic support
 * @{
 *============================================================================*/

/** Global diagnostic callback (NULL = disabled) */
static srukf_diag_fn g_diag_callback = NULL;

void srukf_set_diag_callback(srukf_diag_fn fn) {
  g_diag_callback = fn;
}

/**
 * @brief Report a diagnostic message
 *
 * If a diagnostic callback is registered, invokes it with the message.
 * Used throughout the implementation to report errors and warnings.
 *
 * @param msg Message to report
 */
static void diag_report(const char *msg) {
  if (g_diag_callback)
    g_diag_callback(msg);
}

/** @} */ /* end impl_diag */

/*============================================================================
 * Forward Declarations
 *============================================================================*/

/* Core sigma point operations */
static srukf_return
propagate_sigma_points(const srukf_mat *Xsig, srukf_mat *Ysig,
                       void (*func)(const srukf_mat *, srukf_mat *, void *),
                       void *user);
static srukf_return
compute_cross_covariance(const srukf_mat *Xsig, const srukf_mat *Ysig,
                         const srukf_mat *x_mean, const srukf_mat *y_mean,
                         const srukf_value *weights, srukf_mat *Pxz);

/* SR-UKF specific functions */
static srukf_return chol_downdate_rank1(srukf_mat *S, const srukf_value *v,
                                        srukf_value *work);
static srukf_return compute_weighted_deviations(const srukf_mat *Ysig,
                                                const srukf_mat *mean,
                                                const srukf_value *wc,
                                                srukf_mat *Dev);

/*============================================================================
 * @internal
 * @defgroup impl_workspace Workspace Management
 * @brief Pre-allocated temporaries for zero-allocation filter operation
 *
 * The workspace contains all temporary matrices and buffers needed by
 * predict() and correct(). By pre-allocating these, we avoid malloc/free
 * during filter operation, which is critical for real-time applications.
 *
 * **Memory layout:**
 * - Matrices for sigma point storage: Xsig, Ysig_N, Ysig_M
 * - Mean vectors: x_pred, y_mean
 * - Covariance temporaries: S_tmp, P_pred, Pyy, Pxz, etc.
 * - SR-UKF specific: Dev_N, Dev_M, qr_work_*, Syy
 * - Small buffers: tau (QR), downdate_work, dev0 (for negative wc[0])
 *
 * @{
 *============================================================================*/
/**
 * @brief Pre-allocated workspace for filter operations
 *
 * Contains all temporary storage needed by predict() and correct().
 * Sized for specific (N, M) dimensions; reallocated if dimensions change.
 */
struct srukf_workspace {
  srukf_index N; /**< State dimension this workspace was allocated for */
  srukf_index M; /**< Measurement dimension */

  /** @name Predict Temporaries
   *  @{ */
  srukf_mat *Xsig;   /**< N x (2N+1) - sigma points before propagation */
  srukf_mat *Ysig_N; /**< N x (2N+1) - sigma points after process model */
  srukf_mat *x_pred; /**< N x 1 - predicted state mean */
  srukf_mat *S_tmp;  /**< N x N - temporary for atomic update */
  srukf_mat *P_pred; /**< N x N - (unused, kept for compatibility) */
  srukf_mat *Qfull;  /**< N x N - (unused, kept for compatibility) */
  /** @} */

  /** @name Correct Temporaries
   *  @{ */
  srukf_mat *Ysig_M; /**< M x (2N+1) - sigma points in measurement space */
  srukf_mat *y_mean; /**< M x 1 - predicted measurement mean */
  srukf_mat *Pyy;    /**< M x M - (unused, we use Syy instead) */
  srukf_mat *Pxz; /**< N x M - cross-covariance between state and measurement */
  srukf_mat *K;   /**< N x M - Kalman gain */
  srukf_mat *innov; /**< M x 1 - innovation (z - z_predicted) */
  srukf_mat *P_new; /**< N x N - (unused) */
  srukf_mat *x_new; /**< N x 1 - updated state (for atomic update) */
  srukf_mat *S_new; /**< N x N - updated sqrt-covariance (for atomic update) */
  srukf_mat *Rtmp;  /**< M x M - (unused) */
  srukf_mat *dx;    /**< N x 1 - state correction K * innovation */
  srukf_mat *tmp1;  /**< N x M - temp for K * Syy product */
  srukf_mat *Kt;    /**< M x N - (unused) */
  srukf_mat *tmp2;  /**< N x N - (unused) */
  /** @} */

  /** @name SR-UKF Specific
   *  These are the key matrices for the square-root formulation
   *  @{ */
  srukf_mat *Dev_N; /**< N x (2N+1) - weighted deviations sqrt(|wc|)*(Y-mean)
                       for predict */
  srukf_mat *Dev_M; /**< M x (2N+1) - weighted deviations for correct */
  srukf_mat *qr_work_N; /**< (2N+1+N) x N - compound matrix for QR in predict */
  srukf_mat *qr_work_M; /**< (2N+1+M) x M - compound matrix for QR in correct */
  srukf_mat *Syy; /**< M x M - measurement sqrt-covariance (lower triangular) */
  /** @} */

  /** @name Small Buffers
   *  Avoid malloc in hot path
   *  @{ */
  srukf_value *tau_N; /**< QR householder scalars for predict (N elements) */
  srukf_value *tau_M; /**< QR householder scalars for correct (M elements) */
  srukf_value
      *downdate_work;  /**< Cholesky downdate scratch (max(N,M) elements) */
  srukf_value *dev0_N; /**< First deviation column for predict downdate */
  srukf_value *dev0_M; /**< First deviation column for correct downdate */
  /** @} */
};

/** @} */ /* end impl_workspace definition */

void srukf_free_workspace(srukf *ukf) {
  if (!ukf || !ukf->ws)
    return;

  srukf_workspace *ws = ukf->ws;

  /* Free all matrices */
  srukf_mat_free(ws->Xsig);
  srukf_mat_free(ws->Ysig_N);
  srukf_mat_free(ws->x_pred);
  srukf_mat_free(ws->S_tmp);
  srukf_mat_free(ws->P_pred);
  srukf_mat_free(ws->Qfull);
  srukf_mat_free(ws->Ysig_M);
  srukf_mat_free(ws->y_mean);
  srukf_mat_free(ws->Pyy);
  srukf_mat_free(ws->Pxz);
  srukf_mat_free(ws->K);
  srukf_mat_free(ws->innov);
  srukf_mat_free(ws->P_new);
  srukf_mat_free(ws->x_new);
  srukf_mat_free(ws->S_new);
  srukf_mat_free(ws->Rtmp);
  srukf_mat_free(ws->dx);
  srukf_mat_free(ws->tmp1);
  srukf_mat_free(ws->Kt);
  srukf_mat_free(ws->tmp2);
  srukf_mat_free(ws->Dev_N);
  srukf_mat_free(ws->Dev_M);
  srukf_mat_free(ws->qr_work_N);
  srukf_mat_free(ws->qr_work_M);
  srukf_mat_free(ws->Syy);

  /* Free pre-allocated buffers */
  free(ws->tau_N);
  free(ws->tau_M);
  free(ws->downdate_work);
  free(ws->dev0_N);
  free(ws->dev0_M);

  free(ws);
  ukf->ws = NULL;
}

/* Allocate workspace for given dimensions */
srukf_return srukf_alloc_workspace(srukf *ukf) {
  if (!ukf)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = srukf_state_dim(ukf);
  srukf_index M = srukf_meas_dim(ukf);
  if (N == 0 || M == 0)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* If workspace exists and dimensions match, nothing to do */
  if (ukf->ws && ukf->ws->N == N && ukf->ws->M == M)
    return SRUKF_RETURN_OK;

  /* Free existing workspace if dimensions changed */
  if (ukf->ws)
    srukf_free_workspace(ukf);

  srukf_index n_sigma = 2 * N + 1;

  /* Allocate workspace struct */
  srukf_workspace *ws = (srukf_workspace *)calloc(1, sizeof(srukf_workspace));
  if (!ws)
    return SRUKF_RETURN_PARAMETER_ERROR;

  ws->N = N;
  ws->M = M;

  /* Allocate all matrices */
  ws->Xsig = SRUKF_MAT_ALLOC(N, n_sigma);
  ws->Ysig_N = SRUKF_MAT_ALLOC(N, n_sigma);
  ws->x_pred = SRUKF_MAT_ALLOC(N, 1);
  ws->S_tmp = SRUKF_MAT_ALLOC(N, N);
  ws->P_pred = SRUKF_MAT_ALLOC(N, N);
  ws->Qfull = SRUKF_MAT_ALLOC(N, N);
  ws->Ysig_M = SRUKF_MAT_ALLOC(M, n_sigma);
  ws->y_mean = SRUKF_MAT_ALLOC(M, 1);
  ws->Pyy = SRUKF_MAT_ALLOC(M, M);
  ws->Pxz = SRUKF_MAT_ALLOC(N, M);
  ws->K = SRUKF_MAT_ALLOC(N, M);
  ws->innov = SRUKF_MAT_ALLOC(M, 1);
  ws->P_new = SRUKF_MAT_ALLOC(N, N);
  ws->x_new = SRUKF_MAT_ALLOC(N, 1);
  ws->S_new = SRUKF_MAT_ALLOC(N, N);
  ws->Rtmp = SRUKF_MAT_ALLOC(M, M);
  ws->dx = SRUKF_MAT_ALLOC(N, 1);
  ws->tmp1 = SRUKF_MAT_ALLOC(N, M);
  ws->Kt = SRUKF_MAT_ALLOC(M, N);
  ws->tmp2 = SRUKF_MAT_ALLOC(N, N);

  /* SR-UKF specific matrices */
  ws->Dev_N = SRUKF_MAT_ALLOC(N, n_sigma);
  ws->Dev_M = SRUKF_MAT_ALLOC(M, n_sigma);
  /* QR workspace: (n_sigma + dim) x dim, need extra row for safety */
  ws->qr_work_N = SRUKF_MAT_ALLOC(n_sigma + N + 1, N);
  ws->qr_work_M = SRUKF_MAT_ALLOC(n_sigma + M + 1, M);
  ws->Syy = SRUKF_MAT_ALLOC(M, M);

  /* Pre-allocated buffers for hot path (avoid malloc in predict/correct) */
  ws->tau_N = (srukf_value *)calloc(N, sizeof(srukf_value));
  ws->tau_M = (srukf_value *)calloc(M, sizeof(srukf_value));
  srukf_index max_dim = (N > M) ? N : M;
  ws->downdate_work = (srukf_value *)calloc(max_dim, sizeof(srukf_value));
  ws->dev0_N = (srukf_value *)calloc(N, sizeof(srukf_value));
  ws->dev0_M = (srukf_value *)calloc(M, sizeof(srukf_value));

  /* Check all allocations succeeded */
  if (!ws->Xsig || !ws->Ysig_N || !ws->x_pred || !ws->S_tmp || !ws->P_pred ||
      !ws->Qfull || !ws->Ysig_M || !ws->y_mean || !ws->Pyy || !ws->Pxz ||
      !ws->K || !ws->innov || !ws->P_new || !ws->x_new || !ws->S_new ||
      !ws->Rtmp || !ws->dx || !ws->tmp1 || !ws->Kt || !ws->tmp2 || !ws->Dev_N ||
      !ws->Dev_M || !ws->qr_work_N || !ws->qr_work_M || !ws->Syy ||
      !ws->tau_N || !ws->tau_M || !ws->downdate_work || !ws->dev0_N ||
      !ws->dev0_M) {
    ukf->ws = ws; /* Temporarily assign so free_workspace can clean up */
    srukf_free_workspace(ukf);
    return SRUKF_RETURN_PARAMETER_ERROR;
  }

  ukf->ws = ws;
  return SRUKF_RETURN_OK;
}

/**
 * @brief Ensure workspace is allocated (lazy allocation)
 *
 * Checks if workspace exists and matches current dimensions.
 * Allocates or reallocates as needed.
 *
 * @param ukf Filter instance
 * @return SRUKF_RETURN_OK if workspace is ready
 */
static srukf_return ensure_workspace(srukf *ukf) {
  if (ukf->ws) {
    /* Check dimensions still match */
    srukf_index N = srukf_state_dim(ukf);
    srukf_index M = srukf_meas_dim(ukf);
    if (ukf->ws->N == N && ukf->ws->M == M)
      return SRUKF_RETURN_OK;
  }
  return srukf_alloc_workspace(ukf);
}

/** @} */ /* end impl_workspace */

/*============================================================================
 * @internal
 * @defgroup impl_helpers Helper Functions
 * @brief Various utility functions
 * @{
 *============================================================================*/

/**
 * @brief Check if matrix contains any NaN or Inf values
 *
 * Used to validate callback outputs. If a process or measurement model
 * produces non-finite values, we detect it here and return an error
 * rather than letting garbage propagate through the filter.
 *
 * @param M Matrix to check
 * @return true if all values are finite, false if any NaN/Inf found
 */
static bool is_numeric_valid(const srukf_mat *M) {
  if (!M || !M->data)
    return false;
  for (srukf_index j = 0; j < M->n_cols; ++j)
    for (srukf_index i = 0; i < M->n_rows; ++i) {
      srukf_value v = SRUKF_ENTRY(M, i, j);
      if (isnan(v) || isinf(v))
        return false;
    }
  return true;
}

/**
 * @brief Allocate a vector of doubles
 * @param vec Output pointer
 * @param len Number of elements
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return alloc_vector(srukf_value **vec, srukf_index len) {
  *vec = (srukf_value *)calloc(len, sizeof(srukf_value));
  return (*vec) ? SRUKF_RETURN_OK : SRUKF_RETURN_PARAMETER_ERROR;
}

/** @} */ /* end impl_helpers */

/*============================================================================
 * @internal
 * @defgroup impl_weights Weight Computation
 * @brief UKF sigma point weights
 *
 * The UKF uses weighted averages to reconstruct mean and covariance from
 * sigma points. There are two sets of weights:
 *
 * - **Mean weights (wm):** Used to compute weighted mean
 * - **Covariance weights (wc):** Used to compute weighted covariance
 *
 * The weights sum to 1 for the mean, but wc[0] can differ from wm[0]
 * to better capture higher-order moments of the distribution.
 *
 * @{
 *============================================================================*/

/**
 * @brief Compute sigma point weights
 *
 * Given the current scaling parameters (alpha, beta, kappa, lambda),
 * computes the 2N+1 weights for mean and covariance calculations.
 *
 * **Weight formulas:**
 * @code
 * wm[0] = lambda / (N + lambda)
 * wm[i] = 1 / (2 * (N + lambda))  for i = 1..2N
 *
 * wc[0] = wm[0] + (1 - alpha^2 + beta)
 * wc[i] = wm[i]  for i = 1..2N
 * @endcode
 *
 * **Note on negative wc[0]:**
 * For small alpha, lambda approaches -N, making wm[0] large and negative.
 * Adding (1 - alpha^2 + beta) may not compensate enough, leaving wc[0] < 0.
 * This is handled correctly in the QR-based covariance computation.
 *
 * @param ukf Filter with alpha/beta/kappa/lambda set
 * @param n State dimension N
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return srukf_compute_weights(srukf *ukf, const srukf_index n) {
  srukf_index n_sigma = 2 * n + 1;

  /* Allocate weight vectors if needed */
  if (!ukf->wm) {
    if (alloc_vector(&ukf->wm, n_sigma) != SRUKF_RETURN_OK)
      return SRUKF_RETURN_PARAMETER_ERROR;
  }
  if (!ukf->wc) {
    if (alloc_vector(&ukf->wc, n_sigma) != SRUKF_RETURN_OK) {
      free(ukf->wm);
      ukf->wm = NULL;
      return SRUKF_RETURN_PARAMETER_ERROR;
    }
  }

  /* Common denominator for all weights */
  const srukf_value denom = (srukf_value)n + ukf->lambda;
  if (fabs(denom) < SRUKF_EPS) { /* safeguard against division by zero */
    diag_report("compute_weights: denominator too small (n + lambda ≈ 0)");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* Mean weights: wm[0] = λ / (n+λ), wm[i>0] = 1/(2(n+λ)) */
  for (srukf_index i = 0; i < n_sigma; ++i)
    ukf->wm[i] = 1.0 / (2.0 * denom);
  ukf->wm[0] = ukf->lambda / denom;

  /* Covariance weights: wc[0] = wm[0] + (1-α²+β), wc[i>0] = wm[i] */
  for (srukf_index i = 0; i < n_sigma; ++i)
    ukf->wc[i] = ukf->wm[i];
  ukf->wc[0] += (1.0 - ukf->alpha * ukf->alpha + ukf->beta);

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_weights */

srukf_return srukf_set_scale(srukf *ukf, srukf_value alpha, srukf_value beta,
                             srukf_value kappa) {
  if (!ukf || !ukf->x)
    return SRUKF_RETURN_PARAMETER_ERROR; /* filter or state not yet allocated */

  if (alpha <= 0.0)
    return SRUKF_RETURN_PARAMETER_ERROR; /* α must be positive */

  /* ----------------- compute λ --------------------------------- */
  srukf_index n = ukf->x->n_rows; /* state dimension */
  srukf_value lambda =
      alpha * alpha * ((srukf_value)n + kappa) - (srukf_value)n;

  /* --------- guard against λ ≈ –n ------------------------------ */
  if (fabs((double)(n + lambda)) < SRUKF_EPS) {
    /* Cannot recompute α if (n + κ) ≈ 0 (would divide by zero) */
    if (fabs((double)n + kappa) < SRUKF_EPS)
      return SRUKF_RETURN_PARAMETER_ERROR;
    /* clamp λ to –n + ε */
    lambda = -(srukf_value)n + SRUKF_EPS;
    /* recompute α from the clamped λ  (λ = α²(n+κ) – n)  */
    srukf_value a = sqrt((lambda + (srukf_value)n) / ((srukf_value)n + kappa));
    return srukf_set_scale(ukf, a, beta, kappa);
  } else {
    /* --------- store the (possibly adjusted) parameters ------------- */
    ukf->alpha = alpha;
    ukf->beta = beta;
    ukf->kappa = kappa;
    ukf->lambda = lambda;

    /* --------- recompute mean & covariance weights -------------- */
    return srukf_compute_weights(ukf, n);
  }
}

/* Free all memory allocated for the filter. */
void srukf_free(srukf *ukf) {
  if (!ukf)
    return;

  /* Free workspace if allocated */
  srukf_free_workspace(ukf);

  /* Free all internal matrices (they own their data) */
  if (ukf->x)
    srukf_mat_free(ukf->x);
  if (ukf->S)
    srukf_mat_free(ukf->S);
  if (ukf->Qsqrt)
    srukf_mat_free(ukf->Qsqrt);
  if (ukf->Rsqrt)
    srukf_mat_free(ukf->Rsqrt);

  /* Free weight vectors if they were allocated */
  if (ukf->wm) {
    free(ukf->wm);
    ukf->wm = NULL;
  }
  if (ukf->wc) {
    free(ukf->wc);
    ukf->wc = NULL;
  }

  /* Finally free the filter struct itself */
  free(ukf);
}

/*-------------------- Dimension accessors ----------------------------*/
srukf_index srukf_state_dim(const srukf *ukf) {
  if (!ukf || !ukf->x)
    return 0;
  return ukf->x->n_rows;
}

srukf_index srukf_meas_dim(const srukf *ukf) {
  if (!ukf || !ukf->Rsqrt)
    return 0;
  return ukf->Rsqrt->n_rows;
}

/*-------------------- State accessors --------------------------------*/

srukf_return srukf_get_state(const srukf *ukf, srukf_mat *x_out) {
  if (!ukf || !ukf->x || !x_out)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = ukf->x->n_rows;
  if (x_out->n_rows != N || x_out->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;

  memcpy(x_out->data, ukf->x->data, N * sizeof(srukf_value));
  return SRUKF_RETURN_OK;
}

srukf_return srukf_set_state(srukf *ukf, const srukf_mat *x_in) {
  if (!ukf || !ukf->x || !x_in)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = ukf->x->n_rows;
  if (x_in->n_rows != N || x_in->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;

  memcpy(ukf->x->data, x_in->data, N * sizeof(srukf_value));
  return SRUKF_RETURN_OK;
}

srukf_return srukf_get_sqrt_cov(const srukf *ukf, srukf_mat *S_out) {
  if (!ukf || !ukf->S || !S_out)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = ukf->S->n_rows;
  if (S_out->n_rows != N || S_out->n_cols != N)
    return SRUKF_RETURN_PARAMETER_ERROR;

  memcpy(S_out->data, ukf->S->data, N * N * sizeof(srukf_value));
  return SRUKF_RETURN_OK;
}

srukf_return srukf_set_sqrt_cov(srukf *ukf, const srukf_mat *S_in) {
  if (!ukf || !ukf->S || !S_in)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = ukf->S->n_rows;
  if (S_in->n_rows != N || S_in->n_cols != N)
    return SRUKF_RETURN_PARAMETER_ERROR;

  memcpy(ukf->S->data, S_in->data, N * N * sizeof(srukf_value));
  return SRUKF_RETURN_OK;
}

srukf_return srukf_reset(srukf *ukf, srukf_value init_std) {
  if (!ukf || !ukf->x || !ukf->S)
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (init_std <= 0.0)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = ukf->x->n_rows;

  /* Zero the state vector */
  memset(ukf->x->data, 0, N * sizeof(srukf_value));

  /* Set S to diagonal matrix with init_std on diagonal */
  for (srukf_index i = 0; i < N; ++i)
    for (srukf_index j = 0; j < N; ++j)
      SRUKF_ENTRY(ukf->S, i, j) = (i == j) ? init_std : 0.0;

  return SRUKF_RETURN_OK;
}

/* ------------------------------------------------------------------ */
/*  Shared internal initialisation routine – used by both `create()`  */
/*  and `create_from_noise()` to avoid code duplication.             */
/* ------------------------------------------------------------------ */
static srukf_return srukf_init(srukf *ukf, int N /* states */,
                               int M /* measurements */,
                               const srukf_mat *Qsqrt_src,
                               const srukf_mat *Rsqrt_src) {
  if (!ukf)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* ----------------- State vector --------------------------------- */
  ukf->x = SRUKF_MAT_ALLOC(N, 1);
  if (!ukf->x)
    return SRUKF_RETURN_PARAMETER_ERROR;
  SRUKF_SET_TYPE(ukf->x, SRUKF_TYPE_COL_MAJOR);

  /* ----------------- State covariance square‑root ----------------- */
  ukf->S = SRUKF_MAT_ALLOC(N, N);
  if (!ukf->S) {
    srukf_mat_free(ukf->x);
    return SRUKF_RETURN_PARAMETER_ERROR;
  }
  SRUKF_SET_TYPE(ukf->S, SRUKF_TYPE_SQUARE | SRUKF_TYPE_COL_MAJOR);
  /* initialize only the diagonal so that correction
   * can occur before prediction */
  for (srukf_index i = 0; i < (srukf_index)N; ++i)
    for (srukf_index j = 0; j < (srukf_index)N; ++j)
      SRUKF_ENTRY(ukf->S, i, j) = (i == j) ? 0.001 : 0.0;

  /* ---------- Process‑noise ---------- */
  if (Qsqrt_src) {
    ukf->Qsqrt = SRUKF_MAT_ALLOC(N, N);
    for (srukf_index j = 0; j < (srukf_index)N; ++j)
      for (srukf_index i = 0; i < (srukf_index)N; ++i)
        SRUKF_ENTRY(ukf->Qsqrt, i, j) = SRUKF_ENTRY(Qsqrt_src, i, j);
  } else {
    ukf->Qsqrt = SRUKF_MAT_ALLOC_NO_DATA(N, N);
  }
  if (!ukf->Qsqrt) {
    srukf_mat_free(ukf->x);
    srukf_mat_free(ukf->S);
    return SRUKF_RETURN_PARAMETER_ERROR;
  }
  SRUKF_SET_TYPE(ukf->Qsqrt, SRUKF_TYPE_SQUARE | SRUKF_TYPE_COL_MAJOR);

  /* ---------- Measurement‑noise ---------- */
  if (Rsqrt_src) {
    ukf->Rsqrt = SRUKF_MAT_ALLOC(M, M);
    for (srukf_index j = 0; j < (srukf_index)M; ++j)
      for (srukf_index i = 0; i < (srukf_index)M; ++i)
        SRUKF_ENTRY(ukf->Rsqrt, i, j) = SRUKF_ENTRY(Rsqrt_src, i, j);
  } else {
    ukf->Rsqrt = SRUKF_MAT_ALLOC_NO_DATA(M, M);
  }
  if (!ukf->Rsqrt) {
    srukf_mat_free(ukf->x);
    srukf_mat_free(ukf->S);
    srukf_mat_free(ukf->Qsqrt);
    return SRUKF_RETURN_PARAMETER_ERROR;
  }
  SRUKF_SET_TYPE(ukf->Rsqrt, SRUKF_TYPE_SQUARE | SRUKF_TYPE_COL_MAJOR);

  /* ----------------- Default scaling -------------------------------- */
  return srukf_set_scale(ukf, DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_KAPPA);
}

/* ------------------------------------------------------------------ */
/*  srukf_create – create a filter with uninitialised noise matrices */
/* ------------------------------------------------------------------ */
srukf *srukf_create(int N /* states */, int M /* measurements */) {
  /* Validate dimensions */
  if (N <= 0 || M <= 0)
    return NULL;

  srukf *ukf = (srukf *)calloc(1, sizeof(srukf));
  if (!ukf)
    return NULL; /* out‑of‑memory */

  /* initialise all internal data (noise matrices left empty) */
  if (srukf_init(ukf, N, M, NULL, NULL) != SRUKF_RETURN_OK) {
    srukf_free(ukf);
    return NULL;
  }
  return ukf;
}

/* ------------------------------------------------------------------ */
/*  srukf_create_from_noise – create a filter from supplied noise    */
/* ------------------------------------------------------------------ */
srukf *srukf_create_from_noise(const srukf_mat *Qsqrt, const srukf_mat *Rsqrt) {
  if (!Qsqrt || !Rsqrt)
    return NULL;

  /* Dimensions must agree */
  if (Qsqrt->n_rows != Qsqrt->n_cols || Rsqrt->n_rows != Rsqrt->n_cols)
    return NULL;

  srukf *ukf = (srukf *)calloc(1, sizeof(srukf));
  if (!ukf)
    return NULL;

  /* initialize all internal data and copy the supplied noise matrices */
  if (srukf_init(ukf, (int)Qsqrt->n_rows, (int)Rsqrt->n_rows, Qsqrt, Rsqrt) !=
      SRUKF_RETURN_OK) {
    srukf_free(ukf);
    return NULL;
  }
  return ukf;
}

/* Set the filter's noise square‑root covariance matrices. */
srukf_return srukf_set_noise(srukf *ukf, const srukf_mat *Qsqrt,
                             const srukf_mat *Rsqrt) {
  /* Basic checks */
  if (!ukf || !Qsqrt || !Rsqrt)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Ensure that the filter is well-formed. */
  if (!ukf->x)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (!ukf->Qsqrt || !ukf->Rsqrt)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* State dimension N and measurement dimension M */
  srukf_index N = ukf->x->n_rows;     /* x is N×1 */
  srukf_index M = ukf->Rsqrt->n_rows; /* previously allocated M×M */

  /* Check dimensions of the supplied matrices */
  if (Qsqrt->n_rows != N || Qsqrt->n_cols != N)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Rsqrt->n_rows != M || Rsqrt->n_cols != M)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Allocate new matrices BEFORE freeing old ones.
   * This ensures filter state is preserved if allocation fails. */
  srukf_mat *new_Qsqrt = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *new_Rsqrt = SRUKF_MAT_ALLOC(M, M);
  if (!new_Qsqrt || !new_Rsqrt) {
    /* On failure, clean up any partial allocation and preserve old matrices */
    if (new_Qsqrt)
      srukf_mat_free(new_Qsqrt);
    if (new_Rsqrt)
      srukf_mat_free(new_Rsqrt);
    return SRUKF_RETURN_PARAMETER_ERROR;
  }

  /* Copy data element‑wise (matrix is stored column‑major) */
  for (srukf_index j = 0; j < N; ++j)
    for (srukf_index i = 0; i < N; ++i)
      SRUKF_ENTRY(new_Qsqrt, i, j) = SRUKF_ENTRY(Qsqrt, i, j);

  for (srukf_index j = 0; j < M; ++j)
    for (srukf_index i = 0; i < M; ++i)
      SRUKF_ENTRY(new_Rsqrt, i, j) = SRUKF_ENTRY(Rsqrt, i, j);

  /* Set appropriate type flags */
  SRUKF_SET_TYPE(new_Qsqrt, SRUKF_TYPE_SQUARE | SRUKF_TYPE_COL_MAJOR);
  SRUKF_SET_TYPE(new_Rsqrt, SRUKF_TYPE_SQUARE | SRUKF_TYPE_COL_MAJOR);

  /* Now safe to free old matrices and assign new ones */
  srukf_mat_free(ukf->Qsqrt);
  srukf_mat_free(ukf->Rsqrt);
  ukf->Qsqrt = new_Qsqrt;
  ukf->Rsqrt = new_Rsqrt;

  return SRUKF_RETURN_OK;
}

/*============================================================================
 * @internal
 * @defgroup impl_sigma Sigma Point Generation
 * @brief Creating the 2N+1 sigma points that capture mean and covariance
 *
 * Sigma points are the core insight of the Unscented Transform. Instead of
 * linearizing a nonlinear function (like EKF does), we:
 *
 * 1. Choose 2N+1 carefully placed sample points around the mean
 * 2. Propagate each through the nonlinear function exactly
 * 3. Reconstruct statistics from the transformed samples
 *
 * The sigma points form a symmetric pattern: the mean, plus N points
 * offset by +gamma * column_of_S, plus N points offset by -gamma * column_of_S.
 * This pattern exactly matches the first two moments (mean and covariance)
 * of the original distribution.
 *
 * @{
 *============================================================================*/

/**
 * @brief Generate sigma points from state and sqrt-covariance
 *
 * Creates 2N+1 sigma points arranged symmetrically around the mean:
 *
 * @code
 * chi[0]   = x                        (the mean)
 * chi[i]   = x + gamma * S(:,i)       for i = 1..N
 * chi[i+N] = x - gamma * S(:,i)       for i = 1..N
 * @endcode
 *
 * where gamma = sqrt(N + lambda) is the spread factor.
 *
 * **Geometric interpretation:**
 * If P = S*S' is the covariance, then the sigma points lie on an ellipsoid
 * centered at x, scaled by gamma. The columns of S define the principal
 * axes of this ellipsoid.
 *
 * **Why use S instead of P?**
 * We need sqrt(P) to generate sigma points. In standard UKF, we'd compute
 * chol(P) every time. In SR-UKF, we already have S, saving a Cholesky
 * decomposition per step.
 *
 * @param x State vector (N x 1)
 * @param S Sqrt-covariance (N x N, lower triangular)
 * @param lambda Scaling parameter from UKF tuning
 * @param Xsig Output: sigma points (N x (2N+1))
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return generate_sigma_points_from(const srukf_mat *x,
                                               const srukf_mat *S,
                                               srukf_value lambda,
                                               srukf_mat *Xsig) {
  if (!x || !S || !Xsig)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index n = x->n_rows;       /* state dimension N */
  srukf_index n_sigma = 2 * n + 1; /* number of sigma points */
  if (Xsig->n_rows != n || Xsig->n_cols != n_sigma)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (S->n_rows != n || S->n_cols != n)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* scaling factor γ = sqrt( N + λ ) */
  srukf_value gamma = sqrt((srukf_value)n + lambda);
  if (gamma <= 0.0) {
    diag_report("generate_sigma_points: gamma <= 0 (N + lambda <= 0)");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* 1st column = mean state */
  for (srukf_index i = 0; i < n; ++i)
    SRUKF_ENTRY(Xsig, i, 0) = SRUKF_ENTRY(x, i, 0);

  /* Remaining columns */
  for (srukf_index k = 0; k < n; ++k) {
    for (srukf_index i = 0; i < n; ++i) {
      /* +γ * S(:,k)  */
      SRUKF_ENTRY(Xsig, i, k + 1) =
          SRUKF_ENTRY(x, i, 0) + gamma * SRUKF_ENTRY(S, i, k);
      /* -γ * S(:,k)  */
      SRUKF_ENTRY(Xsig, i, k + 1 + n) =
          SRUKF_ENTRY(x, i, 0) - gamma * SRUKF_ENTRY(S, i, k);
    }
  }
  return SRUKF_RETURN_OK;
}

/**
 * @brief Create a column view into a matrix (zero-copy)
 *
 * Sets up V to reference column k of M without copying data.
 * This is used extensively when propagating sigma points, where
 * we need to pass individual columns to the process/measurement model.
 *
 * @param V Output view (caller provides storage for the srukf_mat struct)
 * @param M Source matrix
 * @param k Column index
 */
static inline void srukf_mat_column_view(srukf_mat *V, const srukf_mat *M,
                                         srukf_index k) {
  V->n_rows = M->n_rows;
  V->n_cols = 1;
  V->inc_row = 1;         /* column vector */
  V->inc_col = M->n_rows; /* distance to next column in M */
  V->data = M->data + k * M->inc_col;
  V->type = 0;
  SRUKF_SET_TYPE(V, SRUKF_TYPE_COL_MAJOR);
}

/**
 * @brief Propagate sigma points through a function
 *
 * Applies a function (process model or measurement model) to each
 * sigma point individually. This is where the "unscented" magic happens:
 * we evaluate the nonlinear function exactly, rather than linearizing it.
 *
 * @param Xsig Input sigma points (N x (2N+1) for state, M x (2N+1) for meas)
 * @param Ysig Output propagated points (same size as Xsig or different for h)
 * @param func Function to apply: func(x_in, x_out, user)
 * @param user User data passed to func
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return
propagate_sigma_points(const srukf_mat *Xsig, srukf_mat *Ysig,
                       void (*func)(const srukf_mat *, srukf_mat *, void *),
                       void *user) {
  /* Basic sanity checks */
  if (!Xsig || !func || !Ysig)
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (Ysig->data == NULL)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Xsig->n_cols != Ysig->n_cols)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index n_sigma = Xsig->n_cols; /* number of sigma points */

  /* Temporary matrix descriptors for a single column vector */
  srukf_mat col_in, col_out;

  /* Loop over all sigma points */
  for (srukf_index k = 0; k < n_sigma; ++k) {
    /* Point to the k‑th column of the input and output matrices */
    srukf_mat_column_view(&col_in, Xsig, k);
    srukf_mat_column_view(&col_out, Ysig, k);
    /* Call the user supplied model */
    func(&col_in, &col_out, user);
  }

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_sigma */

/*============================================================================
 * @internal
 * @defgroup impl_srukf_core SR-UKF Core Operations
 * @brief QR-based covariance updates and Cholesky downdates
 *
 * These are the key numerical routines that make SR-UKF work. The central
 * insight is that we can maintain S (where P = S*S') without ever forming P,
 * using two key operations:
 *
 * 1. **QR-based update:** To compute S where S*S' = sum of outer products +
 *noise, we stack everything into a tall matrix and take QR. The R factor
 *    (transposed) gives us S.
 *
 * 2. **Cholesky downdate:** When wc[0] < 0, we need to subtract a rank-1 term.
 *    The downdate modifies S in-place: S_new * S_new' = S * S' - v * v'
 *
 * @{
 *============================================================================*/

/**
 * @brief Cholesky rank-1 downdate
 *
 * Updates a Cholesky factor to subtract a rank-1 term:
 * @code
 * S_new * S_new' = S * S' - v * v'
 * @endcode
 *
 * This is the inverse operation of a Cholesky update. While updates are
 * always stable (adding positive-definite term), downdates can fail if
 * the result would be non-positive-definite.
 *
 * **Algorithm:** Uses Givens rotations applied column-by-column.
 * For each column j:
 * 1. Compute rotation to zero out work[j] against S[j,j]
 * 2. Apply rotation to update S[j,j] and remaining elements
 * 3. Propagate effect to work[j+1..n]
 *
 * **Numerical stability:** Givens rotations are orthogonal transformations,
 * which are maximally stable. We detect failure (non-SPD result) by checking
 * if r^2 = S[j,j]^2 - work[j]^2 becomes negative.
 *
 * @param S Lower triangular Cholesky factor (N x N), modified in place
 * @param v Vector to downdate by (length N)
 * @param work Scratch buffer (length N), contents destroyed
 * @return SRUKF_RETURN_OK on success, SRUKF_RETURN_MATH_ERROR if non-SPD
 */
static srukf_return chol_downdate_rank1(srukf_mat *S, const srukf_value *v,
                                        srukf_value *work) {
  if (!S || !v || !work)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index n = S->n_rows;
  if (S->n_cols != n)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Copy v to work buffer */
  memcpy(work, v, n * sizeof(srukf_value));

  /* Apply Givens rotations to zero out work while updating S */
  for (srukf_index j = 0; j < n; ++j) {
    srukf_value Sjj = SRUKF_ENTRY(S, j, j);
    srukf_value wj = work[j];

    /* Compute r = sqrt(Sjj^2 - wj^2) */
    srukf_value r2 = Sjj * Sjj - wj * wj;
    if (r2 < 0.0) {
      /* Matrix would become non-SPD */
      return SRUKF_RETURN_MATH_ERROR;
    }
    srukf_value r = sqrt(r2);

    if (fabs(Sjj) < SRUKF_EPS) {
      /* Skip if diagonal is essentially zero */
      if (fabs(wj) > SRUKF_EPS) {
        return SRUKF_RETURN_MATH_ERROR;
      }
      continue;
    }

    /* Givens rotation parameters: c = r/Sjj, s = wj/Sjj */
    srukf_value c = r / Sjj;
    srukf_value s = wj / Sjj;

    /* Update diagonal */
    SRUKF_ENTRY(S, j, j) = r;

    /* Update remaining rows and work */
    for (srukf_index i = j + 1; i < n; ++i) {
      srukf_value Sij = SRUKF_ENTRY(S, i, j);
      srukf_value wi = work[i];

      /* Update S(i,j) and work(i) */
      SRUKF_ENTRY(S, i, j) = (Sij - s * wi) / c;
      work[i] = c * wi - s * Sij;
    }
  }

  return SRUKF_RETURN_OK;
}

/**
 * @brief Compute weighted deviations from sigma points
 *
 * Computes:
 * @code
 * Dev(:,k) = sqrt(|wc[k]|) * (Ysig(:,k) - mean)
 * @endcode
 *
 * These deviations are the building blocks for the QR-based covariance
 * computation. The covariance (if we computed it) would be:
 * @code
 * P = sum_k wc[k] * (Ysig(:,k) - mean) * (Ysig(:,k) - mean)'
 *   = sum_k sign(wc[k]) * Dev(:,k) * Dev(:,k)'
 * @endcode
 *
 * For positive weights, this is Dev * Dev'. For negative wc[0], we
 * compute QR without column 0, then downdate with Dev(:,0).
 *
 * **Note on negative weights:**
 * wc[0] can be negative for small alpha. We use |wc| for the sqrt and
 * track the sign separately. The caller must handle the negative case
 * via Cholesky downdate.
 *
 * @param Ysig Propagated sigma points (dim x (2N+1))
 * @param mean Weighted mean (dim x 1)
 * @param wc Covariance weights (2N+1 elements)
 * @param Dev Output deviations (dim x (2N+1))
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return compute_weighted_deviations(const srukf_mat *Ysig,
                                                const srukf_mat *mean,
                                                const srukf_value *wc,
                                                srukf_mat *Dev) {
  if (!Ysig || !mean || !wc || !Dev)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index M = Ysig->n_rows;
  srukf_index n_sigma = Ysig->n_cols;

  if (mean->n_rows != M || mean->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Dev->n_rows != M || Dev->n_cols != n_sigma)
    return SRUKF_RETURN_PARAMETER_ERROR;

  for (srukf_index k = 0; k < n_sigma; ++k) {
    srukf_value sw = sqrt(fabs(wc[k]));
    for (srukf_index i = 0; i < M; ++i) {
      SRUKF_ENTRY(Dev, i, k) =
          sw * (SRUKF_ENTRY(Ysig, i, k) - SRUKF_ENTRY(mean, i, 0));
    }
  }

  return SRUKF_RETURN_OK;
}

/**
 * @brief Compute sqrt-covariance from weighted deviations via QR
 *
 * This is the heart of the SR-UKF algorithm. Instead of computing:
 * @code
 * P = sum_k wc[k] * dev_k * dev_k' + Q
 * S = chol(P)
 * @endcode
 *
 * We directly compute S via QR decomposition of a compound matrix:
 * @code
 * A = [ Dev(:,1:2N)'  ]     <-- (2N) rows if wc[0] >= 0, else (2N) rows
 *     [ Noise_sqrt'   ]     <-- (dim) rows
 *
 * [Q, R] = qr(A)
 * S = R'                    <-- lower triangular
 * @endcode
 *
 * **Why this works:**
 * If A = [a1; a2; ...] (rows), then A'*A = sum(a_i' * a_i).
 * The R from QR satisfies R'*R = A'*A (by orthogonality of Q).
 * So R'*R = sum(dev_k * dev_k') + Noise_sqrt * Noise_sqrt' = P.
 * Thus R' is the Cholesky factor S.
 *
 * **Handling negative wc[0]:**
 * If wc[0] < 0, we exclude Dev(:,0) from the QR (which computes the sum
 * of positive terms), then apply a Cholesky downdate to subtract
 * Dev(:,0) * Dev(:,0)'.
 *
 * **Ensuring positive diagonal:**
 * QR can produce negative diagonal in R. We flip signs of entire columns
 * to ensure S has positive diagonal, which is the standard Cholesky convention.
 *
 * @param Dev Weighted deviations (dim x (2N+1))
 * @param Noise_sqrt Noise sqrt-covariance (dim x dim)
 * @param S Output sqrt-covariance (dim x dim)
 * @param work QR workspace matrix ((2N+1+dim) x dim)
 * @param tau QR householder scalars (dim elements)
 * @param downdate_work Scratch for downdate (dim elements)
 * @param wc0_negative True if wc[0] < 0 (requires downdate)
 * @param dev0 First deviation column (for downdate, NULL if wc0 >= 0)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return
srukf_sqrt_from_deviations_ex(const srukf_mat *Dev, const srukf_mat *Noise_sqrt,
                              srukf_mat *S, srukf_mat *work, srukf_value *tau,
                              srukf_value *downdate_work, bool wc0_negative,
                              const srukf_value *dev0) {
  if (!Dev || !Noise_sqrt || !S || !work || !tau)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index dim = Dev->n_rows;     /* dimension of output S */
  srukf_index n_sigma = Dev->n_cols; /* 2N+1 */

  if (Noise_sqrt->n_rows != dim || Noise_sqrt->n_cols != dim)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (S->n_rows != dim || S->n_cols != dim)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Build compound matrix for QR:
   * If wc0 >= 0: use all deviations
   * If wc0 < 0:  exclude first deviation column, downdate after
   *
   * Compound = [ Dev(:,start:end)' ; Noise_sqrt' ]
   * where start = wc0_negative ? 1 : 0
   */
  srukf_index start_col = wc0_negative ? 1 : 0;
  srukf_index n_dev_cols = n_sigma - start_col;
  srukf_index n_rows = n_dev_cols + dim; /* rows in compound matrix */

  /* Verify workspace size */
  if (work->n_rows < n_rows || work->n_cols < dim)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Fill compound matrix (column-major):
   * First n_dev_cols rows: transpose of Dev(:,start:end)
   * Last dim rows: transpose of Noise_sqrt
   */
  for (srukf_index j = 0; j < dim; ++j) {
    /* Dev' part */
    for (srukf_index i = 0; i < n_dev_cols; ++i) {
      SRUKF_ENTRY(work, i, j) = SRUKF_ENTRY(Dev, j, i + start_col);
    }
    /* Noise_sqrt' part */
    for (srukf_index i = 0; i < dim; ++i) {
      SRUKF_ENTRY(work, n_dev_cols + i, j) = SRUKF_ENTRY(Noise_sqrt, j, i);
    }
  }

  /* QR factorization to get R (using pre-allocated tau buffer) */
  int info = SRUKF_GEQRF(SRUKF_LAPACK_LAYOUT, (int)n_rows, (int)dim, work->data,
                         (int)SRUKF_LEADING_DIM(work), tau);
  if (info != 0) {
    diag_report("QR factorization (SRUKF_GEQRF) failed");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* Extract R' (transpose of upper triangular R) into S as lower triangular */
  for (srukf_index i = 0; i < dim; ++i) {
    for (srukf_index j = 0; j < dim; ++j) {
      if (j <= i) {
        /* S(i,j) = R(j,i) where R is upper triangular in work */
        SRUKF_ENTRY(S, i, j) = SRUKF_ENTRY(work, j, i);
      } else {
        SRUKF_ENTRY(S, i, j) = 0.0;
      }
    }
  }

  /* Ensure positive diagonal (QR can give negative diagonal elements) */
  for (srukf_index i = 0; i < dim; ++i) {
    if (SRUKF_ENTRY(S, i, i) < 0.0) {
      /* Flip sign of entire column */
      for (srukf_index k = i; k < dim; ++k)
        SRUKF_ENTRY(S, k, i) = -SRUKF_ENTRY(S, k, i);
    }
  }

  /* If wc[0] was negative, we need to downdate S with dev0 */
  if (wc0_negative && dev0) {
    if (!downdate_work)
      return SRUKF_RETURN_PARAMETER_ERROR;
    srukf_return ret = chol_downdate_rank1(S, dev0, downdate_work);
    if (ret != SRUKF_RETURN_OK)
      return ret;
  }

  return SRUKF_RETURN_OK;
}

/**
 * @brief Compute weighted mean from sigma points
 *
 * Computes:
 * @code
 * mean = sum_k wm[k] * Ysig(:,k)
 * @endcode
 *
 * The mean weights sum to 1, so this is a proper weighted average.
 * For the standard UKF parameters, the central sigma point (k=0)
 * gets the most weight when alpha is small.
 *
 * @param Ysig Sigma points (dim x (2N+1))
 * @param wm Mean weights ((2N+1) elements, sum to 1)
 * @param mean Output mean (dim x 1)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return compute_weighted_mean(const srukf_mat *Ysig,
                                          const srukf_value *wm,
                                          srukf_mat *mean) {
  if (!Ysig || !wm || !mean)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index M = Ysig->n_rows;
  srukf_index n_sigma = Ysig->n_cols;

  if (mean->n_rows != M || mean->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;

  for (srukf_index i = 0; i < M; ++i)
    SRUKF_ENTRY(mean, i, 0) = 0.0;

  for (srukf_index k = 0; k < n_sigma; ++k) {
    srukf_value wk = wm[k];
    for (srukf_index i = 0; i < M; ++i)
      SRUKF_ENTRY(mean, i, 0) += wk * SRUKF_ENTRY(Ysig, i, k);
  }

  return SRUKF_RETURN_OK;
}

/**
 * @brief Compute cross-covariance between state and measurement
 *
 * Computes:
 * @code
 * Pxz = sum_k wc[k] * (Xsig(:,k) - x_mean) * (Ysig(:,k) - y_mean)'
 * @endcode
 *
 * This cross-covariance measures how state uncertainty correlates with
 * measurement uncertainty. It's a key ingredient in the Kalman gain:
 * @code
 * K = Pxz * inv(Pyy)
 * @endcode
 *
 * **Intuition:** If a state variable and a measurement are highly
 * correlated (large Pxz entry), then observing that measurement tells
 * us a lot about that state variable.
 *
 * **Note:** Unlike auto-covariance (Pxx or Pyy), cross-covariance is not
 * symmetric and can have any shape (N x M here).
 *
 * @param Xsig State sigma points (N x (2N+1))
 * @param Ysig Measurement sigma points (M x (2N+1))
 * @param x_mean State mean (N x 1)
 * @param y_mean Measurement mean (M x 1)
 * @param weights Covariance weights wc ((2N+1) elements)
 * @param Pxz Output cross-covariance (N x M)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return
compute_cross_covariance(const srukf_mat *Xsig, const srukf_mat *Ysig,
                         const srukf_mat *x_mean, const srukf_mat *y_mean,
                         const srukf_value *weights, srukf_mat *Pxz) {
  if (!Xsig || !Ysig || !x_mean || !y_mean || !weights || !Pxz)
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (Xsig->n_rows != Pxz->n_rows || Ysig->n_rows != Pxz->n_cols)
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (Xsig->n_cols != Ysig->n_cols)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* zero‑initialize */
  for (srukf_index i = 0; i < Pxz->n_rows; ++i)
    for (srukf_index j = 0; j < Pxz->n_cols; ++j)
      SRUKF_ENTRY(Pxz, i, j) = 0.0;

  /* weighted outer products */
  for (srukf_index k = 0; k < Xsig->n_cols; ++k) {
    srukf_value wk = weights[k];
    for (srukf_index i = 0; i < Xsig->n_rows; ++i) {
      srukf_value xi = SRUKF_ENTRY(Xsig, i, k) - SRUKF_ENTRY(x_mean, i, 0);
      for (srukf_index j = 0; j < Ysig->n_rows; ++j) {
        srukf_value yj = SRUKF_ENTRY(Ysig, j, k) - SRUKF_ENTRY(y_mean, j, 0);
        SRUKF_ENTRY(Pxz, i, j) += wk * xi * yj;
      }
    }
  }
  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_srukf_core */

/*============================================================================
 * @internal
 * @defgroup impl_predict Predict Implementation
 * @brief The prediction step advances the state estimate forward in time
 *
 * The predict step answers: "Given our current estimate and the process model,
 * where do we expect the state to be next?"
 *
 * **Key operations:**
 * 1. Generate sigma points from current (x, S)
 * 2. Propagate each through the process model f
 * 3. Compute weighted mean of propagated points
 * 4. Compute new S via QR of deviations + process noise
 *
 * **Effect on uncertainty:**
 * Prediction always increases uncertainty (S grows) because:
 * - Process noise (Q) adds uncertainty
 * - Nonlinear transformation can spread the distribution
 *
 * @{
 *============================================================================*/

/**
 * @brief Core predict implementation
 *
 * Performs the SR-UKF predict step using QR-based covariance update.
 * Can operate in-place (x_out == x_in, S_out == S_in) or to separate buffers.
 *
 * **Algorithm:**
 * @code
 * 1. Xsig = generate_sigma_points(x_in, S_in)      // 2N+1 points
 * 2. Ysig = f(Xsig)                                // propagate each
 * 3. x_out = weighted_mean(Ysig, wm)               // predicted mean
 * 4. Dev = sqrt(|wc|) * (Ysig - x_out)             // weighted deviations
 * 5. S_out = qr([Dev'; Qsqrt'])'                   // sqrt-covariance via QR
 * 6. if wc[0] < 0: choldowndate(S_out, Dev(:,0))  // handle negative weight
 * @endcode
 *
 * @param ukf Filter (provides parameters and workspace)
 * @param x_in Current state (N x 1)
 * @param S_in Current sqrt-covariance (N x N)
 * @param f Process model function
 * @param user User data for f
 * @param x_out Output predicted state (N x 1, may alias x_in)
 * @param S_out Output predicted sqrt-covariance (N x N, may alias S_in)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return
srukf_predict_core(const srukf *ukf, const srukf_mat *x_in,
                   const srukf_mat *S_in,
                   void (*f)(const srukf_mat *, srukf_mat *, void *),
                   void *user, srukf_mat *x_out, srukf_mat *S_out) {
  srukf_return ret = SRUKF_RETURN_OK;
  if (!ukf || !f || !x_in || !S_in || !x_out || !S_out)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (!ukf->Qsqrt || !ukf->wm || !ukf->wc || !ukf->ws)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* --- Dimensions --------------------------------------------------- */
  srukf_index N = x_in->n_rows; /* state dimension */

  /* --- Validate output dimensions ---------------------------------- */
  if (x_out->n_rows != N || x_out->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (S_out->n_rows != N || S_out->n_cols != N)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* --- Use workspace temporaries ----------------------------------- */
  srukf_workspace *ws = ukf->ws;
  srukf_mat *Xsig = ws->Xsig;
  srukf_mat *Ysig = ws->Ysig_N;
  srukf_mat *x_mean = ws->x_pred;
  srukf_mat *Dev = ws->Dev_N;
  srukf_mat *qr_work = ws->qr_work_N;

  /* --- Generate and propagate sigma points ------------------------ */
  ret = generate_sigma_points_from(x_in, S_in, ukf->lambda, Xsig);
  if (ret != SRUKF_RETURN_OK) {
    diag_report("predict: sigma point generation failed");
    return ret;
  }

  ret = propagate_sigma_points(Xsig, Ysig, f, user);
  if (ret != SRUKF_RETURN_OK) {
    diag_report("predict: sigma point propagation failed");
    return ret;
  }

  /* --- Validate callback output ----------------------------------- */
  if (!is_numeric_valid(Ysig)) {
    diag_report("predict: callback f produced NaN or Inf");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* --- Compute weighted mean --------------------------------------- */
  ret = compute_weighted_mean(Ysig, ukf->wm, x_mean);
  if (ret != SRUKF_RETURN_OK) {
    diag_report("predict: compute_weighted_mean failed");
    return ret;
  }

  /* --- Compute weighted deviations --------------------------------- */
  ret = compute_weighted_deviations(Ysig, x_mean, ukf->wc, Dev);
  if (ret != SRUKF_RETURN_OK) {
    diag_report("predict: compute_weighted_deviations failed");
    return ret;
  }

  /* --- Compute S via QR of [Dev'; Qsqrt'] -------------------------- */
  /* Handle potential negative wc[0] (for alpha < 1) */
  bool wc0_negative = (ukf->wc[0] < 0.0);
  srukf_value *dev0 = NULL;
  if (wc0_negative) {
    /* Save first column of Dev for downdate */
    dev0 = ws->dev0_N;
    for (srukf_index i = 0; i < N; ++i)
      dev0[i] = SRUKF_ENTRY(Dev, i, 0);
  }

  ret =
      srukf_sqrt_from_deviations_ex(Dev, ukf->Qsqrt, S_out, qr_work, ws->tau_N,
                                    ws->downdate_work, wc0_negative, dev0);
  if (ret != SRUKF_RETURN_OK) {
    diag_report("predict: sqrt_from_deviations (QR/downdate) failed");
    return ret;
  }

  /* --- Write mean → x_out ---------------------------------------- */
  memcpy(x_out->data, x_mean->data, N * sizeof(srukf_value));

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_predict */

srukf_return srukf_predict_to(srukf *ukf, srukf_mat *x, srukf_mat *S,
                              void (*f)(const srukf_mat *, srukf_mat *, void *),
                              void *user) {
  if (!ukf || !x || !S)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Validate dimensions match filter */
  srukf_index N = srukf_state_dim(ukf);
  if (N == 0)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (x->n_rows != N || x->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (S->n_rows != N || S->n_cols != N)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Ensure workspace is allocated */
  srukf_return ret = ensure_workspace(ukf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  return srukf_predict_core(ukf, x, S, f, user, x, S);
}

srukf_return srukf_predict(srukf *ukf,
                           void (*f)(const srukf_mat *, srukf_mat *, void *),
                           void *user) {
  if (!ukf || !ukf->x || !ukf->S)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Ensure workspace is allocated */
  srukf_return ret = ensure_workspace(ukf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  srukf_index N = ukf->x->n_rows;

  /* Use workspace for output temporaries */
  srukf_mat *x_out = ukf->ws->x_pred;
  srukf_mat *S_out = ukf->ws->S_tmp;

  /* Run core: read from ukf->x/S, write to temps */
  ret = srukf_predict_core(ukf, ukf->x, ukf->S, f, user, x_out, S_out);

  /* Commit on success */
  if (ret == SRUKF_RETURN_OK) {
    memcpy(ukf->x->data, x_out->data, N * sizeof(srukf_value));
    memcpy(ukf->S->data, S_out->data, N * N * sizeof(srukf_value));
  }

  return ret;
}

/*============================================================================
 * @internal
 * @defgroup impl_correct Correct Implementation
 * @brief The correction step incorporates a measurement
 *
 * The correct step answers: "Given a measurement, how should we update
 * our estimate?"
 *
 * **Key insight:** The Kalman gain K determines how to blend prediction
 * and measurement:
 * @code
 * x_new = x_predicted + K * (z_actual - z_predicted)
 * @endcode
 *
 * K is large when:
 * - Measurement is precise (small R) → trust measurement
 * - Prediction is uncertain (large P) → don't trust prediction
 *
 * K is small when:
 * - Measurement is noisy (large R) → don't trust measurement
 * - Prediction is confident (small P) → trust prediction
 *
 * **Effect on uncertainty:**
 * Correction always decreases uncertainty (S shrinks) because:
 * - New information can only reduce our ignorance
 * - Mathematically: S_new² = S_prior² - (K*Syy)(K*Syy)' (a downdate)
 *
 * @{
 *============================================================================*/

/**
 * @brief Core correct implementation
 *
 * Performs the SR-UKF correction step using QR for measurement covariance
 * and Cholesky downdates for state covariance update.
 *
 * **Algorithm:**
 * @code
 * 1. Xsig = generate_sigma_points(x_in, S_in)
 * 2. Zsig = h(Xsig)                              // propagate to measurement
 * space
 * 3. z_pred = weighted_mean(Zsig, wm)            // predicted measurement
 * 4. Syy = qr([Dev_z'; Rsqrt'])'                 // measurement sqrt-covariance
 * 5. Pxz = cross_covariance(Xsig, Zsig)          // state-measurement
 * correlation
 * 6. K = Pxz * inv(Syy' * Syy)                   // Kalman gain via triangular
 * solves
 * 7. innovation = z - z_pred
 * 8. x_out = x_in + K * innovation               // state update
 * 9. U = K * Syy
 * 10. for each column u of U:                    // M Cholesky downdates
 *       S_out = choldowndate(S_out, u)
 * @endcode
 *
 * **Why Cholesky downdates for covariance update?**
 * The covariance update formula is:
 * @code
 * P_new = P - K * Pyy * K'
 *       = P - (K * Syy) * (K * Syy)'
 * @endcode
 * If U = K * Syy (N x M), then P_new = P - U * U'.
 * This is exactly M rank-1 downdates using the columns of U.
 *
 * @param ukf Filter (provides parameters and workspace)
 * @param x_in Predicted state (N x 1)
 * @param S_in Predicted sqrt-covariance (N x N)
 * @param z Measurement (M x 1)
 * @param h Measurement model function
 * @param user User data for h
 * @param x_out Output corrected state (N x 1, may alias x_in)
 * @param S_out Output corrected sqrt-covariance (N x N, may alias S_in)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return
srukf_correct_core(const srukf *ukf, const srukf_mat *x_in,
                   const srukf_mat *S_in, srukf_mat *z,
                   void (*h)(const srukf_mat *, srukf_mat *, void *),
                   void *user, srukf_mat *x_out, srukf_mat *S_out) {
  srukf_return ret = SRUKF_RETURN_OK;
  if (!ukf || !h || !z || !x_in || !S_in || !x_out || !S_out)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (!ukf->Rsqrt || !ukf->wm || !ukf->wc || !ukf->ws)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = x_in->n_rows;       /* state dimension */
  srukf_index M = ukf->Rsqrt->n_rows; /* measurement dimension */

  if (z->n_rows != M || z->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (x_out->n_rows != N || x_out->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (S_out->n_rows != N || S_out->n_cols != N)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* --- Use workspace temporaries ----------------------------------- */
  srukf_workspace *ws = ukf->ws;
  srukf_mat *Xsig = ws->Xsig;
  srukf_mat *Ysig = ws->Ysig_M;
  srukf_mat *x_mean = ws->x_pred;
  srukf_mat *y_mean = ws->y_mean;
  srukf_mat *Syy = ws->Syy;
  srukf_mat *Pxz = ws->Pxz;
  srukf_mat *K = ws->K;
  srukf_mat *innov = ws->innov;
  srukf_mat *x_new = ws->x_new;
  srukf_mat *dx = ws->dx;
  srukf_mat *Dev_M = ws->Dev_M;
  srukf_mat *qr_work = ws->qr_work_M;
  srukf_mat *tmp1 = ws->tmp1; /* Used for K*Syy (N x M) */

  /* 1. Generate & propagate σ‑points -------------------------------- */
  ret = generate_sigma_points_from(x_in, S_in, ukf->lambda, Xsig);
  if (ret != SRUKF_RETURN_OK)
    return ret;
  ret = propagate_sigma_points(Xsig, Ysig, h, user);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* --- Validate callback output ----------------------------------- */
  if (!is_numeric_valid(Ysig)) {
    diag_report("correct: callback h produced NaN or Inf");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* 2. Copy prior state mean for cross-covariance computation ------ */
  for (srukf_index i = 0; i < N; ++i)
    SRUKF_ENTRY(x_mean, i, 0) = SRUKF_ENTRY(x_in, i, 0);

  /* 3. Compute weighted mean for measurement ------------------------- */
  ret = compute_weighted_mean(Ysig, ukf->wm, y_mean);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* 4. Compute weighted deviations for measurement ------------------- */
  ret = compute_weighted_deviations(Ysig, y_mean, ukf->wc, Dev_M);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* 5. Compute Syy via QR of [sqrt(wc)*Dev_M'; Rsqrt'] --------------- */
  bool wc0_negative = (ukf->wc[0] < 0.0);
  srukf_value *dev0_M_buf = NULL;
  if (wc0_negative) {
    /* Save first column of Dev_M for downdate */
    dev0_M_buf = ws->dev0_M;
    for (srukf_index i = 0; i < M; ++i)
      dev0_M_buf[i] = SRUKF_ENTRY(Dev_M, i, 0);
  }

  ret = srukf_sqrt_from_deviations_ex(Dev_M, ukf->Rsqrt, Syy, qr_work,
                                      ws->tau_M, ws->downdate_work,
                                      wc0_negative, dev0_M_buf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* --- Check if Syy is essentially zero ----------------------------- */
  bool Syy_zero = true;
  for (srukf_index i = 0; i < M && Syy_zero; ++i)
    if (fabs(SRUKF_ENTRY(Syy, i, i)) > SRUKF_EPS)
      Syy_zero = false;
  if (Syy_zero) {
    memcpy(x_out->data, x_in->data, N * sizeof(srukf_value));
    memcpy(S_out->data, S_in->data, N * N * sizeof(srukf_value));
    return SRUKF_RETURN_OK;
  }

  /* 6. Cross‑covariance between state & measurement σ‑points -------- */
  ret = compute_cross_covariance(Xsig, Ysig, x_mean, y_mean, ukf->wc, Pxz);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* 7. Kalman gain K = Pxz * inv(Syy') * inv(Syy) -------------------- */
  /* We have Syy as lower triangular.
   * K = Pxz * (Syy * Syy')^{-1} = Pxz * Syy'^{-1} * Syy^{-1}
   *
   * Using BLAS triangular solve:
   * Step 1: K = Pxz * Syy'^{-1}  → solve K * Syy' = Pxz (Syy' is upper tri)
   *         SRUKF_TRSM: side=Right, uplo=Lower, trans=Trans → K * Syy' = Pxz
   * Step 2: K = K * Syy^{-1}     → solve K * Syy = K (Syy is lower tri)
   *         SRUKF_TRSM: side=Right, uplo=Lower, trans=NoTrans → K * Syy = K
   */
  memcpy(K->data, Pxz->data, N * M * sizeof(srukf_value));

  /* K * Syy' = Pxz → K = Pxz * Syy'^{-1} */
  SRUKF_TRSM(SRUKF_CBLAS_LAYOUT, CblasRight, CblasLower, CblasTrans,
             CblasNonUnit, (int)N, (int)M, (srukf_value)1.0, Syy->data,
             (int)SRUKF_LEADING_DIM(Syy), K->data, (int)SRUKF_LEADING_DIM(K));

  /* K * Syy = K → K = K * Syy^{-1} */
  SRUKF_TRSM(SRUKF_CBLAS_LAYOUT, CblasRight, CblasLower, CblasNoTrans,
             CblasNonUnit, (int)N, (int)M, (srukf_value)1.0, Syy->data,
             (int)SRUKF_LEADING_DIM(Syy), K->data, (int)SRUKF_LEADING_DIM(K));

  /* 8. Innovation ------------------------------------------------------- */
  for (srukf_index i = 0; i < M; ++i)
    SRUKF_ENTRY(innov, i, 0) = SRUKF_ENTRY(z, i, 0) - SRUKF_ENTRY(y_mean, i, 0);

  /* 9. State update: dx = K * innov, x_new = x_in + dx ------------------ */
  /* dx (N x 1) = K (N x M) * innov (M x 1) */
  SRUKF_GEMM(SRUKF_CBLAS_LAYOUT, CblasNoTrans, CblasNoTrans, (int)N, 1, (int)M,
             (srukf_value)1.0, K->data, (int)SRUKF_LEADING_DIM(K), innov->data,
             (int)SRUKF_LEADING_DIM(innov), (srukf_value)0.0, dx->data,
             (int)SRUKF_LEADING_DIM(dx));

  for (srukf_index i = 0; i < N; ++i)
    SRUKF_ENTRY(x_new, i, 0) = SRUKF_ENTRY(x_in, i, 0) + SRUKF_ENTRY(dx, i, 0);

  /* 10. Covariance update via Cholesky downdates ----------------------- */
  /* S_out² = S_in² - K * Syy * (K * Syy)'
   * Let U = K * Syy (N x M), then S_out² = S_in² - U * U'
   * This is M rank-1 downdates using columns of U.
   */

  /* First, copy S_in to S_out */
  memcpy(S_out->data, S_in->data, N * N * sizeof(srukf_value));

  /* Compute U = K * Syy (N x M) */
  SRUKF_GEMM(SRUKF_CBLAS_LAYOUT, CblasNoTrans, CblasNoTrans, (int)N, (int)M,
             (int)M, (srukf_value)1.0, K->data, (int)SRUKF_LEADING_DIM(K),
             Syy->data, (int)SRUKF_LEADING_DIM(Syy), (srukf_value)0.0,
             tmp1->data, (int)SRUKF_LEADING_DIM(tmp1));

  /* Apply M successive rank-1 Cholesky downdates */
  /* Use dev0_N as temporary buffer for column extraction (N elements) */
  srukf_value *u_col = ws->dev0_N;

  for (srukf_index j = 0; j < M; ++j) {
    /* Extract column j of U */
    for (srukf_index i = 0; i < N; ++i)
      u_col[i] = SRUKF_ENTRY(tmp1, i, j);

    /* Perform rank-1 downdate */
    ret = chol_downdate_rank1(S_out, u_col, ws->downdate_work);
    if (ret != SRUKF_RETURN_OK) {
      diag_report("correct: Cholesky downdate failed, matrix not SPD");
      return ret;
    }
  }

  /* 11. Write state output --------------------------------------------- */
  memcpy(x_out->data, x_new->data, N * sizeof(srukf_value));

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_correct */

srukf_return srukf_correct_to(srukf *ukf, srukf_mat *x, srukf_mat *S,
                              srukf_mat *z,
                              void (*h)(const srukf_mat *, srukf_mat *, void *),
                              void *user) {
  if (!ukf || !x || !S || !z)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Validate dimensions match filter */
  srukf_index N = srukf_state_dim(ukf);
  srukf_index M = srukf_meas_dim(ukf);
  if (N == 0 || M == 0)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (x->n_rows != N || x->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (S->n_rows != N || S->n_cols != N)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (z->n_rows != M || z->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Ensure workspace is allocated */
  srukf_return ret = ensure_workspace(ukf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  return srukf_correct_core(ukf, x, S, z, h, user, x, S);
}

srukf_return srukf_correct(srukf *ukf, srukf_mat *z,
                           void (*h)(const srukf_mat *, srukf_mat *, void *),
                           void *user) {
  if (!ukf || !ukf->x || !ukf->S)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Ensure workspace is allocated */
  srukf_return ret = ensure_workspace(ukf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  srukf_index N = ukf->x->n_rows;

  /* Use workspace for output temporaries */
  srukf_mat *x_out = ukf->ws->x_new;
  srukf_mat *S_out = ukf->ws->S_new;

  /* Run core: read from ukf->x/S, write to temps */
  ret = srukf_correct_core(ukf, ukf->x, ukf->S, z, h, user, x_out, S_out);

  /* Commit on success */
  if (ret == SRUKF_RETURN_OK) {
    memcpy(ukf->x->data, x_out->data, N * sizeof(srukf_value));
    memcpy(ukf->S->data, S_out->data, N * N * sizeof(srukf_value));
  }

  return ret;
}
