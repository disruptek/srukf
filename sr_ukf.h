#ifndef _SR_UKF_H_
#define _SR_UKF_H_

#include <assert.h>
#include <stdbool.h>
#include <string.h>

#include <lah.h>

/* Forward declaration for workspace (opaque, defined in sr_ukf.c) */
typedef struct sr_ukf_workspace sr_ukf_workspace;

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
  sr_ukf_workspace *ws;
} sr_ukf;

/*-------------------- Diagnostics --------------------------------*/
/* Callback type for diagnostic messages (e.g., fallback activations).
 * Set to NULL (default) to disable diagnostics. */
typedef void (*sr_ukf_diag_fn)(const char *msg);

/* Set global diagnostic callback. Pass NULL to disable. */
void sr_ukf_set_diag_callback(sr_ukf_diag_fn fn);

/*-------------------- Creation / destruction ---------------------*/
/* Allocate a filter using the noise square‑root matrices. */
sr_ukf *sr_ukf_create_from_noise(const lah_mat *Qsqrt, const lah_mat *Rsqrt);

/* Allocate a filter with uninitialized noise matrices. */
sr_ukf *sr_ukf_create(int N /* states */, int M /* measurements */);

/* Free all memory allocated for the filter. */
void sr_ukf_free(sr_ukf *ukf);

/*-------------------- Initialization -------------------------------*/
/* Set the filter's noise square‑root covariance matrices. */
lah_Return sr_ukf_set_noise(sr_ukf *ukf, const lah_mat *Qsqrt,
                            const lah_mat *Rsqrt);

/* Set the filter's scaling parameters (α, β, κ). */
lah_Return sr_ukf_set_scale(sr_ukf *ukf, lah_value alpha, lah_value beta,
                            lah_value kappa);

/*-------------------- Dimension accessors ----------------------------*/
/* Return the state dimension N. Returns 0 if ukf is NULL or invalid. */
lah_index sr_ukf_state_dim(const sr_ukf *ukf);

/* Return the measurement dimension M. Returns 0 if ukf is NULL or invalid. */
lah_index sr_ukf_meas_dim(const sr_ukf *ukf);

/*-------------------- Core operations ------------------------------*/
/* Predict step: propagate the current state through the process
   model f(x,u).  The function must accept a pointer to the current
   state (N×1) and return the propagated state.
   Safe: on error, filter state is unchanged. */
lah_Return sr_ukf_predict(sr_ukf *ukf,
                          void (*f)(const lah_mat *, lah_mat *, void *),
                          void *user);

/* Correct step: incorporate measurement z.
   The measurement model h(x,u) must accept a pointer to the state
   vector and return the predicted measurement vector.
   Safe: on error, filter state is unchanged. */
lah_Return sr_ukf_correct(sr_ukf *ukf, lah_mat *z,
                          void (*h)(const lah_mat *, lah_mat *, void *),
                          void *user);

/*-------------------- Transactional operations ---------------------*/
/* These _to variants operate in-place on user-provided x/S buffers.
   The filter's state (x, S) is not modified - only reads parameters.
   Uses internal workspace for temporaries (allocated on demand).
   Zero-copy for performance-critical or speculative use cases.
   On error, x/S may be partially modified (caller's responsibility). */

/* Transactional predict: operates in-place on user-provided x/S. */
lah_Return sr_ukf_predict_to(sr_ukf *ukf, lah_mat *x, lah_mat *S,
                             void (*f)(const lah_mat *, lah_mat *, void *),
                             void *user);

/* Transactional correct: operates in-place on user-provided x/S. */
lah_Return sr_ukf_correct_to(sr_ukf *ukf, lah_mat *x, lah_mat *S, lah_mat *z,
                             void (*h)(const lah_mat *, lah_mat *, void *),
                             void *user);

/*-------------------- Workspace management (optional) --------------*/
/* Pre-allocate workspace. Called automatically on first predict/correct.
   Returns lahReturnOk on success, error if allocation fails. */
lah_Return sr_ukf_alloc_workspace(sr_ukf *ukf);

/* Free workspace to reclaim memory. Called automatically by sr_ukf_free.
   Safe to call multiple times or on NULL workspace. */
void sr_ukf_free_workspace(sr_ukf *ukf);

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

#endif /* _SR_UKF_H_ */
