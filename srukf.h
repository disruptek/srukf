#ifndef _SRUKF_H_
#define _SRUKF_H_

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include <lah.h>

/* Forward declaration for workspace (opaque, defined in srukf.c) */
typedef struct srukf_workspace srukf_workspace;

/* Square‑Root Unscented Kalman Filter (SR‑UKF) */
typedef struct {
  /* State estimate (N×1) */
  lah_mat *x; /* (N×1) */

  /* Square‑root of state covariance (upper‑triangular, N×N) */
  lah_mat *S; /* (N×N) */

  /* Square‑root process noise covariance (N×N) */
  lah_mat *Qsqrt; /* (N×N) */

  /* Square‑root measurement noise covariance (M×M) */
  lah_mat *Rsqrt; /* (M×M) */

  lah_value alpha;  /* Spread factor α */
  lah_value beta;   /* Prior knowledge β */
  lah_value kappa;  /* Secondary scaling κ */
  lah_value lambda; /* Scaling factor λ = α² (N+κ) – N */

  lah_value *wm; /* Weights for mean (2N+1) */
  lah_value *wc; /* Weights for covariance (2N+1) */

  /* Workspace for temporary allocations (allocated on demand, NULL initially).
   * Not part of filter state - excluded from serialization. */
  srukf_workspace *ws;
} srukf;

/*-------------------- Diagnostics --------------------------------*/
/* Callback type for diagnostic messages (e.g., fallback activations).
 * Set to NULL (default) to disable diagnostics. */
typedef void (*srukf_diag_fn)(const char *msg);

/* Set global diagnostic callback. Pass NULL to disable. */
void srukf_set_diag_callback(srukf_diag_fn fn);

/*-------------------- Creation / destruction ---------------------*/
/* Allocate a filter using the noise square‑root matrices. */
srukf *srukf_create_from_noise(const lah_mat *Qsqrt, const lah_mat *Rsqrt);

/* Allocate a filter with uninitialized noise matrices. */
srukf *srukf_create(int N /* states */, int M /* measurements */);

/* Free all memory allocated for the filter. */
void srukf_free(srukf *ukf);

/*-------------------- Initialization -------------------------------*/
/* Set the filter's noise square‑root covariance matrices. */
lah_Return srukf_set_noise(srukf *ukf, const lah_mat *Qsqrt,
                           const lah_mat *Rsqrt);

/* Set the filter's scaling parameters (α, β, κ). */
lah_Return srukf_set_scale(srukf *ukf, lah_value alpha, lah_value beta,
                           lah_value kappa);

/*-------------------- Dimension accessors ----------------------------*/
/* Return the state dimension N. Returns 0 if ukf is NULL or invalid. */
lah_index srukf_state_dim(const srukf *ukf);

/* Return the measurement dimension M. Returns 0 if ukf is NULL or invalid. */
lah_index srukf_meas_dim(const srukf *ukf);

/*-------------------- Core operations ------------------------------*/
/* Predict step: propagate the current state through the process
   model f(x,u).  The function must accept a pointer to the current
   state (N×1) and return the propagated state.
   Safe: on error, filter state is unchanged. */
lah_Return srukf_predict(srukf *ukf,
                         void (*f)(const lah_mat *, lah_mat *, void *),
                         void *user);

/* Correct step: incorporate measurement z.
   The measurement model h(x,u) must accept a pointer to the state
   vector and return the predicted measurement vector.
   Safe: on error, filter state is unchanged. */
lah_Return srukf_correct(srukf *ukf, lah_mat *z,
                         void (*h)(const lah_mat *, lah_mat *, void *),
                         void *user);

/*-------------------- Transactional operations ---------------------*/
/* These _to variants operate in-place on user-provided x/S buffers.
   The filter's state (x, S) is not modified - only reads parameters.
   Uses internal workspace for temporaries (allocated on demand).
   Zero-copy for performance-critical or speculative use cases.
   On error, x/S may be partially modified (caller's responsibility). */

/* Transactional predict: operates in-place on user-provided x/S. */
lah_Return srukf_predict_to(srukf *ukf, lah_mat *x, lah_mat *S,
                            void (*f)(const lah_mat *, lah_mat *, void *),
                            void *user);

/* Transactional correct: operates in-place on user-provided x/S. */
lah_Return srukf_correct_to(srukf *ukf, lah_mat *x, lah_mat *S, lah_mat *z,
                            void (*h)(const lah_mat *, lah_mat *, void *),
                            void *user);

/*-------------------- Workspace management (optional) --------------*/
/* Pre-allocate workspace. Called automatically on first predict/correct.
   Returns lahReturnOk on success, error if allocation fails. */
lah_Return srukf_alloc_workspace(srukf *ukf);

/* Free workspace to reclaim memory. Called automatically by srukf_free.
   Safe to call multiple times or on NULL workspace. */
void srukf_free_workspace(srukf *ukf);

/* NOTE: A common error here is to assume the function signature
 * matAlloc(rows, columns, ...) when it is actually
 * matAlloc(columns, rows, ...).  Take care to get the dimension
 * order correct; even the LAH author was bitten by this choice!
 */
/* we always use allocMatrix instead of matAlloc because the former
 * takes arguments of (rows, cols) while the later takes (cols, rows) */
#define allocMatrix(rows, columns, o)                                          \
  lah_matAlloc((lah_index)(columns), (lah_index)(rows), (o))
#define allocMatrixNow(rows, columns) allocMatrix(rows, columns, 1)
#define allocMatrixLater(rows, columns) allocMatrix(rows, columns, 0)
/* NEVER use lah_matAlloc directly! */

static_assert(LAH_LAYOUT == lahColMajor,
              "layout not col‑major; update copy, propagate");

#endif /* _SRUKF_H_ */
