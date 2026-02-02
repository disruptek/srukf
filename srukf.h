#ifndef _SRUKF_H_
#define _SRUKF_H_

/**
 * @file srukf.h
 * @brief Square-Root Unscented Kalman Filter (SR-UKF) Library
 *
 * @mainpage SR-UKF: A Teaching Implementation
 *
 * @section intro_sec Introduction
 *
 * This library implements the **Square-Root Unscented Kalman Filter (SR-UKF)**,
 * a numerically robust algorithm for state estimation in nonlinear dynamic
 * systems. It's designed both for practical use and as a learning resource.
 *
 * @section motivation_sec Why Kalman Filtering?
 *
 * Imagine you're tracking a robot's position. You have:
 * - A **motion model** predicting where the robot should be
 * - **Sensor readings** telling you where sensors think it is
 *
 * Neither source is perfect. The motion model accumulates drift; sensors are
 * noisy. The Kalman filter **optimally fuses** these imperfect sources,
 * weighting each by its uncertainty. The result: an estimate better than
 * either source alone.
 *
 * @section ukf_intuition_sec The Unscented Transform: Intuition
 *
 * The classic Kalman filter only works for **linear** systems. For nonlinear
 * systems, we need tricks. The **Extended Kalman Filter (EKF)** linearizes
 * via Jacobians, but this can be inaccurate and requires you to derive
 * analytical derivatives.
 *
 * The **Unscented Kalman Filter (UKF)** takes a different approach:
 *
 * > "It's easier to approximate a probability distribution than to
 * > approximate an arbitrary nonlinear function."
 * > — Julier & Uhlmann
 *
 * Instead of linearizing the function, the UKF:
 * 1. Picks a small set of carefully chosen **sigma points** that capture the
 *    mean and covariance of the current state estimate
 * 2. Propagates **each sigma point** through the (possibly nonlinear) function
 * 3. Reconstructs the mean and covariance from the transformed points
 *
 * This captures nonlinear effects up to 2nd order (vs 1st for EKF) without
 * needing any Jacobians.
 *
 * @section sigma_intuition_sec Sigma Points: The Core Idea
 *
 * For an N-dimensional state, we generate \f$2N+1\f$ sigma points:
 *
 * \f[
 * \begin{aligned}
 * \chi_0 &= \bar{x} & \text{(the mean)} \\
 * \chi_i &= \bar{x} + \gamma \cdot S_{:,i} & i = 1, \ldots, N \\
 * \chi_{i+N} &= \bar{x} - \gamma \cdot S_{:,i} & i = 1, \ldots, N
 * \end{aligned}
 * \f]
 *
 * Where:
 * - \f$\bar{x}\f$ is the current state estimate
 * - \f$S\f$ is the **square-root of the covariance** (so \f$P = SS^T\f$)
 * - \f$\gamma = \sqrt{N + \lambda}\f$ is a scaling factor
 * - \f$\lambda = \alpha^2(N + \kappa) - N\f$ controls the spread
 *
 * These points form a "cloud" around the mean that captures the uncertainty
 * ellipse. When propagated through a nonlinear function, the cloud deforms,
 * and we can recover the new mean and covariance from the deformed cloud.
 *
 * @section srukf_motivation_sec Why Square-Root?
 *
 * The standard UKF works with the full covariance matrix \f$P\f$, but this
 * has problems:
 *
 * 1. **Numerical instability**: Small numerical errors can make \f$P\f$
 *    non-positive-definite, causing the filter to fail
 * 2. **Efficiency**: We often need \f$\sqrt{P}\f$ anyway (for sigma points)
 *
 * The **Square-Root UKF** works directly with \f$S\f$ where \f$P = SS^T\f$:
 *
 * - \f$S\f$ is always well-defined (no need to ensure positive-definiteness)
 * - Numerical precision is effectively **doubled** (errors in \f$S\f$ become
 *   squared errors in \f$P\f$)
 * - The Cholesky decomposition is computed only once at initialization
 *
 * The key operations become:
 * - **QR decomposition** to compute square-root of sums of outer products
 * - **Cholesky rank-1 updates/downdates** to efficiently modify \f$S\f$
 *
 * @section algorithm_overview_sec Algorithm Overview
 *
 * @subsection predict_overview Predict Step
 *
 * Given state \f$\hat{x}\f$ with sqrt-covariance \f$S\f$:
 *
 * 1. Generate sigma points from \f$(\hat{x}, S)\f$
 * 2. Propagate each through the process model: \f$\mathcal{Y}_i = f(\chi_i)\f$
 * 3. Compute predicted mean: \f$\bar{x}^- = \sum w^m_i \mathcal{Y}_i\f$
 * 4. Compute deviations: \f$\mathcal{D}_i = \sqrt{|w^c_i|}(\mathcal{Y}_i -
 * \bar{x}^-)\f$
 * 5. Update sqrt-covariance via QR:
 *    \f$S^- = \text{qr}\left([\mathcal{D}_{1:2N}^T \;|\;
 * Q_{sqrt}^T]\right)^T\f$
 * 6. If \f$w^c_0 < 0\f$, apply Cholesky downdate with \f$\mathcal{D}_0\f$
 *
 * @subsection correct_overview Correct Step
 *
 * Given predicted state \f$(\bar{x}^-, S^-)\f$ and measurement \f$z\f$:
 *
 * 1. Generate sigma points from \f$(\bar{x}^-, S^-)\f$
 * 2. Propagate through measurement model: \f$\mathcal{Z}_i = h(\chi_i)\f$
 * 3. Compute predicted measurement: \f$\bar{z} = \sum w^m_i \mathcal{Z}_i\f$
 * 4. Compute measurement sqrt-covariance \f$S_{yy}\f$ via QR (like step 4-6
 * above)
 * 5. Compute cross-covariance: \f$P_{xz} = \sum w^c_i (\chi_i -
 * \bar{x}^-)(\mathcal{Z}_i - \bar{z})^T\f$
 * 6. Compute Kalman gain: \f$K = P_{xz} S_{yy}^{-T} S_{yy}^{-1}\f$
 * 7. Update state: \f$\hat{x} = \bar{x}^- + K(z - \bar{z})\f$
 * 8. Update sqrt-covariance via M Cholesky downdates:
 *    \f$(S^-)^2 \leftarrow (S^-)^2 - (KS_{yy})(KS_{yy})^T\f$
 *
 * @section tuning_sec Tuning Parameters
 *
 * The UKF has three tuning parameters (\f$\alpha, \beta, \kappa\f$):
 *
 * | Parameter | Typical Value | Effect |
 * |-----------|---------------|--------|
 * | \f$\alpha\f$ | 0.001 - 1 | Controls sigma point spread. Smaller = tighter
 * around mean | | \f$\beta\f$  | 2.0 | Prior knowledge of distribution. 2.0 is
 * optimal for Gaussian | | \f$\kappa\f$ | 0 or 3-N | Secondary scaling. Often
 * set to 0 or to ensure \f$N + \kappa = 3\f$ |
 *
 * The derived parameter \f$\lambda = \alpha^2(N + \kappa) - N\f$ determines
 * the actual spread. For small \f$\alpha\f$, \f$\lambda \approx -N\f$, which
 * can make \f$w^c_0\f$ negative. This library handles negative \f$w^c_0\f$
 * correctly via Cholesky downdates.
 *
 * @section usage_sec Basic Usage
 *
 * @code{.c}
 * // 1. Create filter (N=3 states, M=2 measurements)
 * srukf *ukf = srukf_create(3, 2);
 *
 * // 2. Set noise covariances (square-roots)
 * srukf_mat *Qsqrt = SRUKF_MAT_ALLOC(3, 3);  // process noise
 * srukf_mat *Rsqrt = SRUKF_MAT_ALLOC(2, 2);  // measurement noise
 * // ... fill in Qsqrt, Rsqrt as lower-triangular Cholesky factors
 * srukf_set_noise(ukf, Qsqrt, Rsqrt);
 *
 * // 3. Initialize state
 * srukf_reset(ukf, 1.0);  // zero state, identity covariance scaled by 1.0
 *
 * // 4. Run filter loop
 * while (have_data) {
 *     srukf_predict(ukf, process_model, NULL);
 *     srukf_correct(ukf, measurement, measurement_model, NULL);
 *
 *     // Read out current estimate
 *     srukf_get_state(ukf, state_out);
 * }
 *
 * // 5. Cleanup
 * srukf_free(ukf);
 * @endcode
 *
 * @section thread_safety_sec Thread Safety
 *
 * This library is **NOT thread-safe**:
 * - The diagnostic callback (srukf_set_diag_callback()) is global
 * - Each srukf instance should only be accessed from one thread
 *
 * @section dependencies_sec Dependencies
 *
 * Requires CBLAS and LAPACKE (e.g., OpenBLAS).
 *
 * @section attribution_sec Attribution
 *
 * Matrix utilities derived from LAH (Linear Algebra Helpers) by maj0e,
 * MIT License. See https://github.com/maj0e/lah
 *
 * @section references_sec References
 *
 * - Julier, S.J. & Uhlmann, J.K. (1997). "A New Extension of the Kalman Filter
 *   to Nonlinear Systems"
 * - Van der Merwe, R. & Wan, E.A. (2001). "The Square-Root Unscented Kalman
 *   Filter for State and Parameter-Estimation"
 * - Wan, E.A. & Van der Merwe, R. (2000). "The Unscented Kalman Filter for
 *   Nonlinear Estimation"
 */

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

/*============================================================================
 * @defgroup matrix Matrix Types and Utilities
 * @brief Basic matrix infrastructure for the SR-UKF
 *
 * The library uses a lightweight matrix type optimized for the specific
 * needs of the Kalman filter: column-major storage (for BLAS compatibility),
 * support for matrix views (zero-copy column extraction), and simple
 * allocation semantics.
 * @{
 *============================================================================*/

/**
 * @brief Scalar index type
 *
 * Used for matrix dimensions and loop indices. Using size_t ensures we can
 * handle matrices up to the platform's memory limits.
 */
typedef size_t srukf_index;

/**
 * @brief Scalar value type
 *
 * Double precision by default. Define SRUKF_SINGLE before including this
 * header for single precision (float), which may be faster on some hardware
 * but reduces numerical precision.
 */
#ifdef SRUKF_SINGLE
typedef float srukf_value;
#else
typedef double srukf_value;
#endif

/**
 * @brief Function return codes
 *
 * All functions that can fail return one of these codes. Check the return
 * value and handle errors appropriately.
 */
typedef enum {
  SRUKF_RETURN_OK = 0,          /**< Success */
  SRUKF_RETURN_PARAMETER_ERROR, /**< Invalid parameter (NULL, wrong dims, etc.)
                                 */
  SRUKF_RETURN_MATH_ERROR       /**< Numerical failure (non-SPD matrix, etc.) */
} srukf_return;

/**
 * @brief Matrix type flags
 *
 * Bit flags describing matrix properties. Used internally for validation
 * and optimization.
 */
typedef enum {
  SRUKF_TYPE_COL_MAJOR = 0x01, /**< Column-major storage (always set) */
  SRUKF_TYPE_NO_DATA = 0x02, /**< Matrix struct exists but data not allocated */
  SRUKF_TYPE_VECTOR = 0x04,  /**< Single column (n_cols == 1) */
  SRUKF_TYPE_SQUARE = 0x08   /**< Square matrix (n_rows == n_cols) */
} srukf_mat_type;

/**
 * @brief Matrix structure
 *
 * Represents a 2D matrix in column-major storage. The inc_row and inc_col
 * fields allow representing submatrices and column views without copying data.
 *
 * **Memory layout (column-major):**
 * For a 3x2 matrix, elements are stored as:
 * @code
 * data[0] = A(0,0)  data[3] = A(0,1)
 * data[1] = A(1,0)  data[4] = A(1,1)
 * data[2] = A(2,0)  data[5] = A(2,1)
 * @endcode
 *
 * Access element (i,j) as: `data[i * inc_row + j * inc_col]`
 */
typedef struct {
  srukf_index n_cols;  /**< Number of columns */
  srukf_index n_rows;  /**< Number of rows */
  srukf_index inc_row; /**< Row stride (typically 1 for column-major) */
  srukf_index inc_col; /**< Column stride (typically n_rows for column-major) */
  srukf_value *data;   /**< Pointer to data buffer */
  srukf_mat_type type; /**< Type flags (see srukf_mat_type) */
} srukf_mat;

/**
 * @brief Element access macro (column-major)
 *
 * @param A Pointer to srukf_mat
 * @param i Row index (0-based)
 * @param j Column index (0-based)
 * @return Reference to element A(i,j)
 *
 * Example:
 * @code
 * SRUKF_ENTRY(mat, 0, 0) = 1.0;  // Set top-left element
 * double val = SRUKF_ENTRY(mat, 1, 2);  // Read element at row 1, col 2
 * @endcode
 */
#define SRUKF_ENTRY(A, i, j)                                                   \
  ((A)->data[(i) * (A)->inc_row + (j) * (A)->inc_col])

/** @brief Set a type flag on a matrix */
#define SRUKF_SET_TYPE(A, t) ((A)->type |= (t))
/** @brief Clear a type flag on a matrix */
#define SRUKF_UNSET_TYPE(A, t) ((A)->type &= ~(t))
/** @brief Test if a type flag is set */
#define SRUKF_IS_TYPE(A, t) ((A)->type & (t))

/** @brief Get leading dimension for BLAS/LAPACK (always n_rows for col-major)
 */
#define SRUKF_LEADING_DIM(A) ((A)->n_rows)

/** @brief BLAS layout constant */
#define SRUKF_CBLAS_LAYOUT CblasColMajor
/** @brief LAPACK layout constant */
#define SRUKF_LAPACK_LAYOUT LAPACK_COL_MAJOR

/* BLAS/LAPACK routine selection based on precision */
#ifdef SRUKF_SINGLE
#define SRUKF_GEMM  cblas_sgemm    /**< Matrix multiply (single) */
#define SRUKF_TRSM  cblas_strsm    /**< Triangular solve (single) */
#define SRUKF_TRSV  cblas_strsv    /**< Triangular vector solve (single) */
#define SRUKF_GEQRF LAPACKE_sgeqrf /**< QR factorization (single) */
#define SRUKF_POTRF LAPACKE_spotrf /**< Cholesky factorization (single) */
#else
#define SRUKF_GEMM  cblas_dgemm    /**< Matrix multiply (double) */
#define SRUKF_TRSM  cblas_dtrsm    /**< Triangular solve (double) */
#define SRUKF_TRSV  cblas_dtrsv    /**< Triangular vector solve (double) */
#define SRUKF_GEQRF LAPACKE_dgeqrf /**< QR factorization (double) */
#define SRUKF_POTRF LAPACKE_dpotrf /**< Cholesky factorization (double) */
#endif

/**
 * @brief Allocate a matrix
 *
 * Creates a new matrix with the specified dimensions. Memory is
 * zero-initialized.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param alloc_data If non-zero, allocate data buffer; otherwise data is NULL
 * @return Allocated matrix, or NULL on failure
 *
 * @note Use srukf_mat_free() to release memory when done.
 */
srukf_mat *srukf_mat_alloc(srukf_index rows, srukf_index cols, int alloc_data);

/**
 * @brief Free a matrix and its data
 *
 * Releases all memory associated with the matrix. Safe to call with NULL.
 *
 * @param mat Matrix to free (may be NULL)
 */
void srukf_mat_free(srukf_mat *mat);

/** @brief Allocate a matrix with data buffer */
#define SRUKF_MAT_ALLOC(rows, cols) srukf_mat_alloc((rows), (cols), 1)
/** @brief Allocate a matrix struct only (data pointer will be NULL) */
#define SRUKF_MAT_ALLOC_NO_DATA(rows, cols) srukf_mat_alloc((rows), (cols), 0)

/** @} */ /* end of matrix group */

/*============================================================================
 * @defgroup filter SR-UKF Filter
 * @brief The main Square-Root Unscented Kalman Filter interface
 * @{
 *============================================================================*/

/**
 * @brief Opaque workspace structure
 *
 * Pre-allocated temporaries to avoid malloc during predict/correct.
 * Allocated on demand, freed with filter or explicitly.
 */
typedef struct srukf_workspace srukf_workspace;

/**
 * @brief Square-Root Unscented Kalman Filter
 *
 * This structure holds the complete state of an SR-UKF instance:
 * - Current state estimate and its sqrt-covariance
 * - Noise covariances (as square-roots)
 * - UKF tuning parameters and derived weights
 * - Pre-allocated workspace for efficiency
 *
 * **Key insight:** We store \f$S\f$ where \f$P = SS^T\f$, not \f$P\f$ itself.
 * This is the "square-root" in SR-UKF, providing numerical stability.
 *
 * @see srukf_create, srukf_create_from_noise
 */
typedef struct {
  /** @name State Estimate
   *  The current best estimate of the system state
   *  @{
   */
  srukf_mat *x; /**< State estimate vector (N x 1) */
  srukf_mat *S; /**< Sqrt of state covariance (N x N, lower triangular).
                     Satisfies \f$P = SS^T\f$ where P is the covariance. */
  /** @} */

  /** @name Noise Covariances
   *  Process and measurement noise, stored as square-roots
   *  @{
   */
  srukf_mat *Qsqrt; /**< Sqrt of process noise covariance (N x N).
                         Models uncertainty in the state transition.
                         Larger values = less trust in the model. */
  srukf_mat *Rsqrt; /**< Sqrt of measurement noise covariance (M x M).
                         Models sensor noise.
                         Larger values = less trust in measurements. */
  /** @} */

  /** @name UKF Parameters
   *  Tuning parameters controlling sigma point distribution
   *  @{
   */
  srukf_value alpha;  /**< Spread of sigma points (typically 1e-3 to 1).
                           Smaller values keep points closer to the mean. */
  srukf_value beta;   /**< Prior distribution knowledge (2.0 for Gaussian).
                           Affects the zeroth covariance weight. */
  srukf_value kappa;  /**< Secondary scaling (typically 0 or 3-N).
                           Can be used to ensure semi-positive definiteness. */
  srukf_value lambda; /**< Computed: \f$\alpha^2 (N + \kappa) - N\f$.
                           Determines actual sigma point spread. */
  /** @} */

  /** @name Sigma Point Weights
   *  Weights for reconstructing mean and covariance from sigma points
   *  @{
   */
  srukf_value *wm; /**< Mean weights (2N+1 elements).
                        \f$w^m_0 = \frac{\lambda}{N+\lambda}\f$,
                        \f$w^m_i = \frac{1}{2(N+\lambda)}\f$ for i>0 */
  srukf_value *wc; /**< Covariance weights (2N+1 elements).
                        \f$w^c_0 = w^m_0 + (1 - \alpha^2 + \beta)\f$,
                        \f$w^c_i = w^m_i\f$ for i>0.
                        Note: \f$w^c_0\f$ can be negative for small alpha! */
  /** @} */

  srukf_workspace *ws; /**< Pre-allocated workspace (allocated on demand) */
} srukf;

/*----------------------------------------------------------------------------
 * @defgroup diagnostics Diagnostics
 * @brief Diagnostic and debugging support
 * @{
 *----------------------------------------------------------------------------*/

/**
 * @brief Diagnostic callback function type
 *
 * User-provided function to receive diagnostic messages from the library.
 * Useful for debugging filter issues.
 *
 * @param msg Null-terminated diagnostic message
 */
typedef void (*srukf_diag_fn)(const char *msg);

/**
 * @brief Set the global diagnostic callback
 *
 * When set, the library will call this function with diagnostic messages
 * about errors and warnings (e.g., "QR factorization failed").
 *
 * @param fn Callback function, or NULL to disable diagnostics
 *
 * @note This is a global setting affecting all filter instances.
 *
 * Example:
 * @code
 * void my_diag(const char *msg) {
 *     fprintf(stderr, "SR-UKF: %s\n", msg);
 * }
 * srukf_set_diag_callback(my_diag);
 * @endcode
 */
void srukf_set_diag_callback(srukf_diag_fn fn);

/** @} */ /* end of diagnostics group */

/*----------------------------------------------------------------------------
 * @defgroup lifecycle Creation and Destruction
 * @brief Filter lifecycle management
 * @{
 *----------------------------------------------------------------------------*/

/**
 * @brief Create a filter from noise covariance matrices
 *
 * Creates and initializes a filter with the given noise characteristics.
 * The state dimension N is inferred from Qsqrt, measurement dimension M
 * from Rsqrt.
 *
 * @param Qsqrt Process noise sqrt-covariance (N x N, lower triangular)
 * @param Rsqrt Measurement noise sqrt-covariance (M x M, lower triangular)
 * @return Allocated filter, or NULL on failure
 *
 * @note Both matrices must be square. They are copied, so the originals
 *       can be freed after this call.
 *
 * @see srukf_create, srukf_free
 */
srukf *srukf_create_from_noise(const srukf_mat *Qsqrt, const srukf_mat *Rsqrt);

/**
 * @brief Create a filter with uninitialized noise matrices
 *
 * Creates a filter with the specified dimensions but without setting
 * noise covariances. You must call srukf_set_noise() before using
 * predict/correct.
 *
 * @param N State dimension (must be > 0)
 * @param M Measurement dimension (must be > 0)
 * @return Allocated filter, or NULL on failure
 *
 * This two-step initialization is useful when noise parameters are
 * computed or loaded separately from filter creation.
 *
 * @see srukf_set_noise, srukf_free
 */
srukf *srukf_create(int N, int M);

/**
 * @brief Free all memory allocated for the filter
 *
 * Releases the filter struct, all matrices, weight vectors, and workspace.
 * Safe to call with NULL.
 *
 * @param ukf Filter to free (may be NULL)
 */
void srukf_free(srukf *ukf);

/** @} */ /* end of lifecycle group */

/*----------------------------------------------------------------------------
 * @defgroup initialization Initialization
 * @brief Setting filter parameters and initial conditions
 * @{
 *----------------------------------------------------------------------------*/

/**
 * @brief Set the noise sqrt-covariance matrices
 *
 * Updates the process and measurement noise covariances. This affects
 * how the filter trades off between trusting the model vs measurements.
 *
 * @param ukf Filter instance
 * @param Qsqrt Process noise sqrt-covariance (N x N)
 * @param Rsqrt Measurement noise sqrt-covariance (M x M)
 * @return SRUKF_RETURN_OK on success
 *
 * **Intuition:**
 * - Larger Qsqrt = "my model is unreliable" = trust measurements more
 * - Larger Rsqrt = "my sensors are noisy" = trust model more
 *
 * @note The matrices are copied. Originals can be freed after this call.
 * @note On failure, the existing noise matrices are preserved.
 */
srukf_return srukf_set_noise(srukf *ukf, const srukf_mat *Qsqrt,
                             const srukf_mat *Rsqrt);

/**
 * @brief Set the UKF scaling parameters
 *
 * Configures the sigma point distribution parameters. These affect how
 * the filter captures nonlinear effects.
 *
 * @param ukf Filter instance
 * @param alpha Spread of sigma points around mean (must be > 0, typically 1e-3
 * to 1)
 * @param beta Prior knowledge of distribution (2.0 is optimal for Gaussian)
 * @param kappa Secondary scaling parameter (typically 0 or 3-N)
 * @return SRUKF_RETURN_OK on success, SRUKF_RETURN_PARAMETER_ERROR if alpha <=
 * 0
 *
 * **Derived parameters:**
 * - \f$\lambda = \alpha^2 (N + \kappa) - N\f$
 * - \f$\gamma = \sqrt{N + \lambda}\f$ (sigma point spread factor)
 *
 * **Weight formulas:**
 * - \f$w^m_0 = \lambda / (N + \lambda)\f$
 * - \f$w^m_i = 1 / (2(N + \lambda))\f$ for i > 0
 * - \f$w^c_0 = w^m_0 + (1 - \alpha^2 + \beta)\f$
 * - \f$w^c_i = w^m_i\f$ for i > 0
 *
 * @note For very small alpha, \f$w^c_0\f$ can become negative. This library
 *       handles this correctly via Cholesky downdates.
 */
srukf_return srukf_set_scale(srukf *ukf, srukf_value alpha, srukf_value beta,
                             srukf_value kappa);

/** @} */ /* end of initialization group */

/*----------------------------------------------------------------------------
 * @defgroup accessors State Accessors
 * @brief Reading and writing filter state
 * @{
 *----------------------------------------------------------------------------*/

/**
 * @brief Get the state dimension N
 * @param ukf Filter instance
 * @return State dimension, or 0 if ukf is NULL/invalid
 */
srukf_index srukf_state_dim(const srukf *ukf);

/**
 * @brief Get the measurement dimension M
 * @param ukf Filter instance
 * @return Measurement dimension, or 0 if ukf is NULL/invalid
 */
srukf_index srukf_meas_dim(const srukf *ukf);

/**
 * @brief Get the current state estimate
 *
 * Copies the state vector to a user-provided buffer.
 *
 * @param ukf Filter instance
 * @param x_out Output buffer (N x 1), must be pre-allocated
 * @return SRUKF_RETURN_OK on success
 */
srukf_return srukf_get_state(const srukf *ukf, srukf_mat *x_out);

/**
 * @brief Set the state estimate
 *
 * Overwrites the current state with user-provided values.
 *
 * @param ukf Filter instance
 * @param x_in New state vector (N x 1)
 * @return SRUKF_RETURN_OK on success
 *
 * @note This does not update the covariance. Use srukf_set_sqrt_cov()
 *       or srukf_reset() to also set uncertainty.
 */
srukf_return srukf_set_state(srukf *ukf, const srukf_mat *x_in);

/**
 * @brief Get the sqrt-covariance matrix
 *
 * Copies S where \f$P = SS^T\f$ is the state covariance.
 *
 * @param ukf Filter instance
 * @param S_out Output buffer (N x N), must be pre-allocated
 * @return SRUKF_RETURN_OK on success
 *
 * To get the full covariance P, compute \f$P = S \cdot S^T\f$.
 */
srukf_return srukf_get_sqrt_cov(const srukf *ukf, srukf_mat *S_out);

/**
 * @brief Set the sqrt-covariance matrix
 *
 * @param ukf Filter instance
 * @param S_in New sqrt-covariance (N x N, should be lower triangular)
 * @return SRUKF_RETURN_OK on success
 *
 * @warning S_in should be a valid Cholesky factor (lower triangular,
 *          positive diagonal). Invalid values may cause filter divergence.
 */
srukf_return srukf_set_sqrt_cov(srukf *ukf, const srukf_mat *S_in);

/**
 * @brief Reset filter to initial conditions
 *
 * Sets state to zero and sqrt-covariance to a scaled identity matrix.
 * Useful for reinitializing a filter without reallocating.
 *
 * @param ukf Filter instance
 * @param init_std Initial standard deviation (> 0). The sqrt-covariance
 *                 is set to `init_std * I`.
 * @return SRUKF_RETURN_OK on success
 *
 * **Interpretation:** After reset, each state variable has zero mean and
 * variance `init_std^2`, with no correlation between variables.
 */
srukf_return srukf_reset(srukf *ukf, srukf_value init_std);

/** @} */ /* end of accessors group */

/*----------------------------------------------------------------------------
 * @defgroup operations Core Operations
 * @brief The main predict and correct steps
 * @{
 *----------------------------------------------------------------------------*/

/**
 * @brief Process model callback type
 *
 * User-provided function implementing the state transition model:
 * \f$x_{k+1} = f(x_k)\f$
 *
 * @param x_in Current state (N x 1), read-only
 * @param x_out Next state (N x 1), write output here
 * @param user User data pointer (passed through from predict call)
 *
 * Example (constant velocity model):
 * @code
 * void process_model(const srukf_mat *x_in, srukf_mat *x_out, void *user) {
 *     double dt = *(double*)user;
 *     // State: [position, velocity]
 *     SRUKF_ENTRY(x_out, 0, 0) = SRUKF_ENTRY(x_in, 0, 0)
 *                              + dt * SRUKF_ENTRY(x_in, 1, 0);
 *     SRUKF_ENTRY(x_out, 1, 0) = SRUKF_ENTRY(x_in, 1, 0);  // velocity
 * unchanged
 * }
 * @endcode
 */

/**
 * @brief Measurement model callback type
 *
 * User-provided function implementing the measurement model:
 * \f$z = h(x)\f$
 *
 * @param x_in Current state (N x 1), read-only
 * @param z_out Predicted measurement (M x 1), write output here
 * @param user User data pointer (passed through from correct call)
 *
 * Example (observe position only):
 * @code
 * void meas_model(const srukf_mat *x_in, srukf_mat *z_out, void *user) {
 *     (void)user;
 *     // We only measure position (first state variable)
 *     SRUKF_ENTRY(z_out, 0, 0) = SRUKF_ENTRY(x_in, 0, 0);
 * }
 * @endcode
 */

/**
 * @brief Predict step: propagate state through process model
 *
 * Advances the state estimate forward in time using the process model.
 * This increases uncertainty (covariance grows due to process noise).
 *
 * @param ukf Filter instance
 * @param f Process model function \f$x_{k+1} = f(x_k)\f$
 * @param user User data passed to f
 * @return SRUKF_RETURN_OK on success. On error, filter state is unchanged.
 *
 * **Algorithm:**
 * 1. Generate 2N+1 sigma points from current (x, S)
 * 2. Propagate each sigma point through f
 * 3. Compute new mean as weighted average of propagated points
 * 4. Compute new S via QR decomposition of weighted deviations + Qsqrt
 *
 * @see srukf_correct
 */
srukf_return srukf_predict(srukf *ukf,
                           void (*f)(const srukf_mat *, srukf_mat *, void *),
                           void *user);

/**
 * @brief Correct step: incorporate measurement
 *
 * Updates the state estimate using a new measurement. This decreases
 * uncertainty (measurement reduces our ignorance about the state).
 *
 * @param ukf Filter instance
 * @param z Measurement vector (M x 1)
 * @param h Measurement model function \f$z = h(x)\f$
 * @param user User data passed to h
 * @return SRUKF_RETURN_OK on success. On error, filter state is unchanged.
 *
 * **Algorithm:**
 * 1. Generate sigma points from predicted state
 * 2. Propagate through measurement model h
 * 3. Compute predicted measurement mean and sqrt-covariance Syy
 * 4. Compute cross-covariance Pxz between state and measurement
 * 5. Compute Kalman gain K = Pxz * inv(Syy' * Syy)
 * 6. Update state: x = x + K * (z - z_predicted)
 * 7. Update S via Cholesky downdates: S^2 -= (K*Syy)(K*Syy)'
 *
 * **Intuition:** The Kalman gain K determines how much to trust the
 * measurement vs the prediction. If Rsqrt is large (noisy sensor), K is
 * small and we mostly keep our prediction. If Syy is large (uncertain
 * prediction), K is large and we trust the measurement more.
 *
 * @see srukf_predict
 */
srukf_return srukf_correct(srukf *ukf, srukf_mat *z,
                           void (*h)(const srukf_mat *, srukf_mat *, void *),
                           void *user);

/** @} */ /* end of operations group */

/*----------------------------------------------------------------------------
 * @defgroup transactional Transactional Operations
 * @brief Advanced: operate on external state buffers
 * @{
 *----------------------------------------------------------------------------*/

/**
 * @brief Transactional predict: operate on external buffers
 *
 * Like srukf_predict(), but reads/writes state from user-provided buffers
 * instead of the filter's internal state. Useful for:
 * - Implementing particle filters (multiple hypotheses)
 * - What-if analysis without modifying the filter
 * - Custom rollback/checkpoint schemes
 *
 * @param ukf Filter instance (provides parameters and workspace)
 * @param x State vector (N x 1), modified in-place
 * @param S Sqrt-covariance (N x N), modified in-place
 * @param f Process model function
 * @param user User data passed to f
 * @return SRUKF_RETURN_OK on success
 */
srukf_return srukf_predict_to(srukf *ukf, srukf_mat *x, srukf_mat *S,
                              void (*f)(const srukf_mat *, srukf_mat *, void *),
                              void *user);

/**
 * @brief Transactional correct: operate on external buffers
 *
 * Like srukf_correct(), but reads/writes state from user-provided buffers.
 *
 * @param ukf Filter instance (provides parameters and workspace)
 * @param x State vector (N x 1), modified in-place
 * @param S Sqrt-covariance (N x N), modified in-place
 * @param z Measurement vector (M x 1)
 * @param h Measurement model function
 * @param user User data passed to h
 * @return SRUKF_RETURN_OK on success
 */
srukf_return srukf_correct_to(srukf *ukf, srukf_mat *x, srukf_mat *S,
                              srukf_mat *z,
                              void (*h)(const srukf_mat *, srukf_mat *, void *),
                              void *user);

/** @} */ /* end of transactional group */

/*----------------------------------------------------------------------------
 * @defgroup workspace Workspace Management
 * @brief Control over internal memory allocation
 * @{
 *----------------------------------------------------------------------------*/

/**
 * @brief Pre-allocate workspace
 *
 * The workspace holds temporary matrices used during predict/correct.
 * By default, it's allocated on first use. Call this to allocate it
 * explicitly (e.g., during initialization to avoid allocation during
 * real-time operation).
 *
 * @param ukf Filter instance
 * @return SRUKF_RETURN_OK on success
 *
 * @note Workspace size depends on N and M. If these change (via
 *       srukf_set_noise with different dimensions), workspace is
 *       reallocated automatically.
 */
srukf_return srukf_alloc_workspace(srukf *ukf);

/**
 * @brief Free workspace to reclaim memory
 *
 * Releases the workspace memory. The next predict/correct call will
 * reallocate it. Useful for long-running applications where the filter
 * is dormant for extended periods.
 *
 * @param ukf Filter instance (may be NULL)
 *
 * @note Called automatically by srukf_free().
 */
void srukf_free_workspace(srukf *ukf);

/** @} */ /* end of workspace group */

#endif /* _SRUKF_H_ */
