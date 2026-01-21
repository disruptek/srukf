#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sr_ukf.h"

#define DEFAULT_ALPHA 1e-3
#define DEFAULT_BETA 2.0
#define DEFAULT_KAPPA 1.0
#define SR_UKF_EPS 1e-12 /* safety margin for λ close to –N */

/* BLAS triangular solve: solves op(A)*X = alpha*B or X*op(A) = alpha*B */
#ifdef LAH_SINGLE
#define TRSM cblas_strsm
#define TRSV cblas_strsv
#else
#define TRSM cblas_dtrsm
#define TRSV cblas_dtrsv
#endif

/* --------- Diagnostic callback ----------------------------------- */
static sr_ukf_diag_fn g_diag_callback = NULL;

void sr_ukf_set_diag_callback(sr_ukf_diag_fn fn) { g_diag_callback = fn; }

/* Report a diagnostic message via the callback (if set). */
static void diag_report(const char *msg) {
  if (g_diag_callback)
    g_diag_callback(msg);
}

/* Forward declarations of internal (static) functions */
static lah_Return propagate_sigma_points(const lah_mat *Xsig, lah_mat *Ysig,
                                         void (*func)(const lah_mat *,
                                                      lah_mat *, void *),
                                         void *user);
static lah_Return sqrt_to_covariance(const lah_mat *S, lah_mat *P);
static lah_Return
compute_cross_covariance(const lah_mat *Xsig, const lah_mat *Ysig,
                         const lah_mat *x_mean, const lah_mat *y_mean,
                         const lah_value *weights, lah_mat *Pxz);
static bool is_spd(const lah_mat *A);

/* SR-UKF specific functions */
static lah_Return chol_downdate_rank1(lah_mat *S, const lah_value *v);
static lah_Return compute_weighted_deviations(const lah_mat *Ysig,
                                               const lah_mat *mean,
                                               const lah_value *wc,
                                               lah_mat *Dev);

/*-------------------- Workspace definition and management ----------*/
/* Workspace holds pre-allocated temporaries to avoid malloc on each call.
 * Allocated on demand, freed with filter or explicitly. */
struct sr_ukf_workspace {
  lah_index N; /* state dimension this workspace was allocated for */
  lah_index M; /* measurement dimension */

  /* Predict temporaries */
  lah_mat *Xsig;   /* N × (2N+1) - sigma points */
  lah_mat *Ysig_N; /* N × (2N+1) - propagated sigma points (predict) */
  lah_mat *x_pred; /* N × 1 - predicted mean */
  lah_mat *S_tmp;  /* N × N - temporary sqrt covariance */
  lah_mat *P_pred; /* N × N - predicted covariance */
  lah_mat *Qfull;  /* N × N - full process noise covariance */

  /* Correct temporaries */
  lah_mat *Ysig_M; /* M × (2N+1) - propagated sigma points (correct) */
  lah_mat *y_mean; /* M × 1 - predicted measurement mean */
  lah_mat *Pyy;    /* M × M - innovation covariance */
  lah_mat *Pxz;    /* N × M - cross covariance */
  lah_mat *K;      /* N × M - Kalman gain */
  lah_mat *innov;  /* M × 1 - innovation */
  lah_mat *P_new;  /* N × N - updated covariance */
  lah_mat *x_new;  /* N × 1 - updated state */
  lah_mat *S_new;  /* N × N - updated sqrt covariance */
  lah_mat *Rtmp;   /* M × M - temp for measurement noise */
  lah_mat *dx;     /* N × 1 - state correction */
  lah_mat *tmp1;   /* N × M - temp for K*Pyy */
  lah_mat *Kt;     /* M × N - K transpose */
  lah_mat *tmp2;   /* N × N - temp for K*Pyy*K' */

  /* SR-UKF specific workspace */
  lah_mat *Dev_N;     /* N × (2N+1) - weighted deviations for predict */
  lah_mat *Dev_M;     /* M × (2N+1) - weighted deviations for correct */
  lah_mat *qr_work_N; /* (3N+1) × N - QR workspace for predict */
  lah_mat *qr_work_M; /* (2N+1+M) × M - QR workspace for correct */
  lah_mat *Syy;       /* M × M - measurement sqrt covariance */
};

/* Free workspace and set pointer to NULL */
void sr_ukf_free_workspace(sr_ukf *ukf) {
  if (!ukf || !ukf->ws)
    return;

  sr_ukf_workspace *ws = ukf->ws;

  /* Free all matrices */
  lah_matFree(ws->Xsig);
  lah_matFree(ws->Ysig_N);
  lah_matFree(ws->x_pred);
  lah_matFree(ws->S_tmp);
  lah_matFree(ws->P_pred);
  lah_matFree(ws->Qfull);
  lah_matFree(ws->Ysig_M);
  lah_matFree(ws->y_mean);
  lah_matFree(ws->Pyy);
  lah_matFree(ws->Pxz);
  lah_matFree(ws->K);
  lah_matFree(ws->innov);
  lah_matFree(ws->P_new);
  lah_matFree(ws->x_new);
  lah_matFree(ws->S_new);
  lah_matFree(ws->Rtmp);
  lah_matFree(ws->dx);
  lah_matFree(ws->tmp1);
  lah_matFree(ws->Kt);
  lah_matFree(ws->tmp2);
  lah_matFree(ws->Dev_N);
  lah_matFree(ws->Dev_M);
  lah_matFree(ws->qr_work_N);
  lah_matFree(ws->qr_work_M);
  lah_matFree(ws->Syy);

  free(ws);
  ukf->ws = NULL;
}

/* Allocate workspace for given dimensions */
lah_Return sr_ukf_alloc_workspace(sr_ukf *ukf) {
  if (!ukf)
    return lahReturnParameterError;

  lah_index N = sr_ukf_state_dim(ukf);
  lah_index M = sr_ukf_meas_dim(ukf);
  if (N == 0 || M == 0)
    return lahReturnParameterError;

  /* If workspace exists and dimensions match, nothing to do */
  if (ukf->ws && ukf->ws->N == N && ukf->ws->M == M)
    return lahReturnOk;

  /* Free existing workspace if dimensions changed */
  if (ukf->ws)
    sr_ukf_free_workspace(ukf);

  lah_index n_sigma = 2 * N + 1;

  /* Allocate workspace struct */
  sr_ukf_workspace *ws =
      (sr_ukf_workspace *)calloc(1, sizeof(sr_ukf_workspace));
  if (!ws)
    return lahReturnParameterError;

  ws->N = N;
  ws->M = M;

  /* Allocate all matrices */
  ws->Xsig = allocMatrixNow(N, n_sigma);
  ws->Ysig_N = allocMatrixNow(N, n_sigma);
  ws->x_pred = allocMatrixNow(N, 1);
  ws->S_tmp = allocMatrixNow(N, N);
  ws->P_pred = allocMatrixNow(N, N);
  ws->Qfull = allocMatrixNow(N, N);
  ws->Ysig_M = allocMatrixNow(M, n_sigma);
  ws->y_mean = allocMatrixNow(M, 1);
  ws->Pyy = allocMatrixNow(M, M);
  ws->Pxz = allocMatrixNow(N, M);
  ws->K = allocMatrixNow(N, M);
  ws->innov = allocMatrixNow(M, 1);
  ws->P_new = allocMatrixNow(N, N);
  ws->x_new = allocMatrixNow(N, 1);
  ws->S_new = allocMatrixNow(N, N);
  ws->Rtmp = allocMatrixNow(M, M);
  ws->dx = allocMatrixNow(N, 1);
  ws->tmp1 = allocMatrixNow(N, M);
  ws->Kt = allocMatrixNow(M, N);
  ws->tmp2 = allocMatrixNow(N, N);

  /* SR-UKF specific matrices */
  ws->Dev_N = allocMatrixNow(N, n_sigma);
  ws->Dev_M = allocMatrixNow(M, n_sigma);
  /* QR workspace: (n_sigma + dim) x dim, need extra row for safety */
  ws->qr_work_N = allocMatrixNow(n_sigma + N + 1, N);
  ws->qr_work_M = allocMatrixNow(n_sigma + M + 1, M);
  ws->Syy = allocMatrixNow(M, M);

  /* Check all allocations succeeded */
  if (!ws->Xsig || !ws->Ysig_N || !ws->x_pred || !ws->S_tmp || !ws->P_pred ||
      !ws->Qfull || !ws->Ysig_M || !ws->y_mean || !ws->Pyy || !ws->Pxz ||
      !ws->K || !ws->innov || !ws->P_new || !ws->x_new || !ws->S_new ||
      !ws->Rtmp || !ws->dx || !ws->tmp1 || !ws->Kt || !ws->tmp2 ||
      !ws->Dev_N || !ws->Dev_M || !ws->qr_work_N || !ws->qr_work_M || !ws->Syy) {
    ukf->ws = ws; /* Temporarily assign so free_workspace can clean up */
    sr_ukf_free_workspace(ukf);
    return lahReturnParameterError;
  }

  ukf->ws = ws;
  return lahReturnOk;
}

/* Ensure workspace is allocated (lazy allocation helper) */
static lah_Return ensure_workspace(sr_ukf *ukf) {
  if (ukf->ws) {
    /* Check dimensions still match */
    lah_index N = sr_ukf_state_dim(ukf);
    lah_index M = sr_ukf_meas_dim(ukf);
    if (ukf->ws->N == N && ukf->ws->M == M)
      return lahReturnOk;
  }
  return sr_ukf_alloc_workspace(ukf);
}

/* Check if matrix contains any NaN or Inf values.
 * Returns true if the matrix is numerically valid, false otherwise. */
static bool is_numeric_valid(const lah_mat *M) {
  if (!M || !M->data)
    return false;
  for (lah_index j = 0; j < M->nC; ++j)
    for (lah_index i = 0; i < M->nR; ++i) {
      lah_value v = LAH_ENTRY(M, i, j);
      if (isnan(v) || isinf(v))
        return false;
    }
  return true;
}

/* Check if matrix A is symmetric positive definite (SPD). */
static bool is_spd(const lah_mat *A) {
  /* 1. Basic sanity checks ------------------------------------------------ */
  if (!A)
    return false;
  if (A->nR != A->nC)
    return false; /* must be square */

  /* 2. Symmetry test – 4 significant digits is usually sufficient.      */
  /*    lah_isSymmetric returns 1 on success, 0 otherwise.                 */
  if (!lah_isSymmetric((lah_mat *)A, 4)) /* cast away const for old API */
    return false;

  /* 3. Copy the matrix (lah_chol destroys its argument).                 */
  lah_mat *tmp = allocMatrixNow(A->nR, A->nC);
  if (!tmp)
    return false; /* out‑of‑memory */
  memcpy(tmp->data, A->data, (size_t)(A->nC * A->nR) * sizeof(lah_value));

  /* tiny jitter to guard against round‑off that might make a
   * positive‑definite matrix look singular. */
  for (lah_index i = 0; i < tmp->nR; ++i)
    LAH_ENTRY(tmp, i, i) += SR_UKF_EPS;

  /* 4. Cholesky – success ⇒ SPD, failure ⇒ not SPD.                      */
  bool ok = (lah_chol(tmp, 0) == lahReturnOk);

  lah_matFree(tmp); /* release copy */
  return ok;
}

/* Allocate a vector of length len */
static lah_Return alloc_vector(lah_value **vec, lah_index len) {
  *vec = (lah_value *)calloc(len, sizeof(lah_value));
  return (*vec) ? lahReturnOk : lahReturnParameterError;
}

/* Compute weights for mean (wm) and covariance (wc)
 * for a given filter dimension n and scaling λ. */
static lah_Return sr_ukf_compute_weights(sr_ukf *ukf, const lah_index n) {
  lah_index n_sigma = 2 * n + 1;

  /* Allocate weight vectors if needed */
  if (!ukf->wm) {
    if (alloc_vector(&ukf->wm, n_sigma) != lahReturnOk)
      return lahReturnParameterError;
  }
  if (!ukf->wc) {
    if (alloc_vector(&ukf->wc, n_sigma) != lahReturnOk) {
      free(ukf->wm);
      ukf->wm = NULL;
      return lahReturnParameterError;
    }
  }

  /* Common denominator for all weights */
  const lah_value denom = (lah_value)n + ukf->lambda;
  if (fabs(denom) < SR_UKF_EPS) /* safeguard against division by zero */
    return lahReturnMathError;

  /* Mean weights: wm[0] = λ / (n+λ), wm[i>0] = 1/(2(n+λ)) */
  for (lah_index i = 0; i < n_sigma; ++i)
    ukf->wm[i] = 1.0 / (2.0 * denom);
  ukf->wm[0] = ukf->lambda / denom;

  /* Covariance weights: wc[0] = wm[0] + (1-α²+β), wc[i>0] = wm[i] */
  for (lah_index i = 0; i < n_sigma; ++i)
    ukf->wc[i] = ukf->wm[i];
  ukf->wc[0] += (1.0 - ukf->alpha * ukf->alpha + ukf->beta);

  return lahReturnOk;
}

/* Set the UKF scaling parameters (α, β, κ) and recompute the
 * scaling factor λ and the weight vectors for mean and covariance. */
lah_Return sr_ukf_set_scale(sr_ukf *ukf, lah_value alpha, lah_value beta,
                            lah_value kappa) {
  if (!ukf || !ukf->x)
    return lahReturnParameterError; /* filter or state not yet allocated */

  if (alpha <= 0.0)
    return lahReturnParameterError; /* α must be positive */

  /* ----------------- compute λ --------------------------------- */
  lah_index n = ukf->x->nR; /* state dimension */
  lah_value lambda = alpha * alpha * ((lah_value)n + kappa) - (lah_value)n;

  /* --------- guard against λ ≈ –n ------------------------------ */
  if (fabs((double)(n + lambda)) < SR_UKF_EPS) {
    /* Cannot recompute α if (n + κ) ≈ 0 (would divide by zero) */
    if (fabs((double)n + kappa) < SR_UKF_EPS)
      return lahReturnParameterError;
    /* clamp λ to –n + ε */
    lambda = -(lah_value)n + SR_UKF_EPS;
    /* recompute α from the clamped λ  (λ = α²(n+κ) – n)  */
    lah_value a = sqrt((lambda + (lah_value)n) / ((lah_value)n + kappa));
    return sr_ukf_set_scale(ukf, a, beta, kappa);
  } else {
    /* --------- store the (possibly adjusted) parameters ------------- */
    ukf->alpha = alpha;
    ukf->beta = beta;
    ukf->kappa = kappa;
    ukf->lambda = lambda;

    /* --------- recompute mean & covariance weights -------------- */
    return sr_ukf_compute_weights(ukf, n);
  }
}

/* Free all memory allocated for the filter. */
void sr_ukf_free(sr_ukf *ukf) {
  if (!ukf)
    return;

  /* Free workspace if allocated */
  sr_ukf_free_workspace(ukf);

  /* Free all internal matrices (they own their data) */
  if (ukf->x)
    lah_matFree(ukf->x);
  if (ukf->S)
    lah_matFree(ukf->S);
  if (ukf->Qsqrt)
    lah_matFree(ukf->Qsqrt);
  if (ukf->Rsqrt)
    lah_matFree(ukf->Rsqrt);

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
lah_index sr_ukf_state_dim(const sr_ukf *ukf) {
  if (!ukf || !ukf->x)
    return 0;
  return ukf->x->nR;
}

lah_index sr_ukf_meas_dim(const sr_ukf *ukf) {
  if (!ukf || !ukf->Rsqrt)
    return 0;
  return ukf->Rsqrt->nR;
}

/* ------------------------------------------------------------------ */
/*  Shared internal initialisation routine – used by both `create()`  */
/*  and `create_from_noise()` to avoid code duplication.             */
/* ------------------------------------------------------------------ */
static lah_Return sr_ukf_init(sr_ukf *ukf, int N /* states */,
                              int M /* measurements */,
                              const lah_mat *Qsqrt_src,
                              const lah_mat *Rsqrt_src) {
  if (!ukf)
    return lahReturnParameterError;

  /* ----------------- State vector --------------------------------- */
  ukf->x = allocMatrixNow(N, 1);
  if (!ukf->x)
    return lahReturnParameterError;
  LAH_SETTYPE(ukf->x, lahTypeColMajor);

  /* ----------------- State covariance square‑root ----------------- */
  ukf->S = allocMatrixNow(N, N);
  if (!ukf->S) {
    lah_matFree(ukf->x);
    return lahReturnParameterError;
  }
  LAH_SETTYPE(ukf->S, lahTypeSquare | lahTypeColMajor);
  /* initialize only the diagonal so that correction
   * can occur before prediction */
  for (lah_index i = 0; i < (lah_index)N; ++i)
    for (lah_index j = 0; j < (lah_index)N; ++j)
      LAH_ENTRY(ukf->S, i, j) = (i == j) ? 0.001 : 0.0;

  /* ---------- Process‑noise ---------- */
  if (Qsqrt_src) {
    ukf->Qsqrt = allocMatrixNow(N, N);
    for (lah_index j = 0; j < (lah_index)N; ++j)
      for (lah_index i = 0; i < (lah_index)N; ++i)
        LAH_ENTRY(ukf->Qsqrt, i, j) = LAH_ENTRY(Qsqrt_src, i, j);
  } else {
    ukf->Qsqrt = allocMatrixLater(N, N);
  }
  if (!ukf->Qsqrt) {
    lah_matFree(ukf->x);
    lah_matFree(ukf->S);
    return lahReturnParameterError;
  }
  LAH_SETTYPE(ukf->Qsqrt, lahTypeSquare | lahTypeColMajor);

  /* ---------- Measurement‑noise ---------- */
  if (Rsqrt_src) {
    ukf->Rsqrt = allocMatrixNow(M, M);
    for (lah_index j = 0; j < (lah_index)M; ++j)
      for (lah_index i = 0; i < (lah_index)M; ++i)
        LAH_ENTRY(ukf->Rsqrt, i, j) = LAH_ENTRY(Rsqrt_src, i, j);
  } else {
    ukf->Rsqrt = allocMatrixLater(M, M);
  }
  if (!ukf->Rsqrt) {
    lah_matFree(ukf->x);
    lah_matFree(ukf->S);
    lah_matFree(ukf->Qsqrt);
    return lahReturnParameterError;
  }
  LAH_SETTYPE(ukf->Rsqrt, lahTypeSquare | lahTypeColMajor);

  /* ----------------- Default scaling -------------------------------- */
  return sr_ukf_set_scale(ukf, DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_KAPPA);
}

/* ------------------------------------------------------------------ */
/*  sr_ukf_create – create a filter with uninitialised noise matrices */
/* ------------------------------------------------------------------ */
sr_ukf *sr_ukf_create(int N /* states */, int M /* measurements */) {
  sr_ukf *ukf = (sr_ukf *)calloc(1, sizeof(sr_ukf));
  if (!ukf)
    return NULL; /* out‑of‑memory */

  /* initialise all internal data (noise matrices left empty) */
  if (sr_ukf_init(ukf, N, M, NULL, NULL) != lahReturnOk) {
    sr_ukf_free(ukf);
    return NULL;
  }
  return ukf;
}

/* ------------------------------------------------------------------ */
/*  sr_ukf_create_from_noise – create a filter from supplied noise    */
/* ------------------------------------------------------------------ */
sr_ukf *sr_ukf_create_from_noise(const lah_mat *Qsqrt, const lah_mat *Rsqrt) {
  if (!Qsqrt || !Rsqrt)
    return NULL;

  /* Dimensions must agree */
  if (Qsqrt->nR != Qsqrt->nC || Rsqrt->nR != Rsqrt->nC)
    return NULL;

  sr_ukf *ukf = (sr_ukf *)calloc(1, sizeof(sr_ukf));
  if (!ukf)
    return NULL;

  /* initialize all internal data and copy the supplied noise matrices */
  if (sr_ukf_init(ukf, (int)Qsqrt->nR, (int)Rsqrt->nR, Qsqrt, Rsqrt) !=
      lahReturnOk) {
    sr_ukf_free(ukf);
    return NULL;
  }
  return ukf;
}

/* Set the filter's noise square‑root covariance matrices. */
lah_Return sr_ukf_set_noise(sr_ukf *ukf, const lah_mat *Qsqrt,
                            const lah_mat *Rsqrt) {
  /* Basic checks */
  if (!ukf || !Qsqrt || !Rsqrt)
    return lahReturnParameterError;

  /* Ensure that the filter is well-formed. */
  if (!ukf->x)
    return lahReturnParameterError;
  if (!ukf->Qsqrt || !ukf->Rsqrt)
    return lahReturnParameterError;

  /* State dimension N and measurement dimension M */
  lah_index N = ukf->x->nR;     /* x is N×1 */
  lah_index M = ukf->Rsqrt->nR; /* previously allocated M×M */

  /* Check dimensions of the supplied matrices */
  if (Qsqrt->nR != N || Qsqrt->nC != N)
    return lahReturnParameterError;
  if (Rsqrt->nR != M || Rsqrt->nC != M)
    return lahReturnParameterError;

  /* Free existing noise matrices (they own no data). */
  if (ukf->Qsqrt) {
    lah_matFree(ukf->Qsqrt);
    ukf->Qsqrt = NULL;
  }
  if (ukf->Rsqrt) {
    lah_matFree(ukf->Rsqrt);
    ukf->Rsqrt = NULL;
  }

  /* Allocate new matrices */
  ukf->Qsqrt = allocMatrixNow(N, N);
  ukf->Rsqrt = allocMatrixNow(M, M);
  if (!ukf->Qsqrt || !ukf->Rsqrt) {
    /* On failure clean up */
    if (ukf->Qsqrt)
      lah_matFree(ukf->Qsqrt);
    if (ukf->Rsqrt)
      lah_matFree(ukf->Rsqrt);
    ukf->Qsqrt = NULL;
    ukf->Rsqrt = NULL;
    return lahReturnParameterError;
  }

  /* Copy data element‑wise (matrix is stored column‑major) */
  for (lah_index j = 0; j < N; ++j)
    for (lah_index i = 0; i < N; ++i)
      LAH_ENTRY(ukf->Qsqrt, i, j) = LAH_ENTRY(Qsqrt, i, j);

  for (lah_index j = 0; j < M; ++j)
    for (lah_index i = 0; i < M; ++i)
      LAH_ENTRY(ukf->Rsqrt, i, j) = LAH_ENTRY(Rsqrt, i, j);

  /* Set appropriate type flags */
  LAH_SETTYPE(ukf->Qsqrt, lahTypeSquare | lahTypeColMajor);
  LAH_SETTYPE(ukf->Rsqrt, lahTypeSquare | lahTypeColMajor);

  return lahReturnOk;
}

/* Internal helper: generate sigma points from arbitrary x/S.
 *  x      : state vector (N×1)
 *  S      : sqrt covariance (N×N)
 *  lambda : scaling factor
 *  Xsig   : output (N×(2N+1))
 */
static lah_Return generate_sigma_points_from(const lah_mat *x, const lah_mat *S,
                                             lah_value lambda, lah_mat *Xsig) {
  if (!x || !S || !Xsig)
    return lahReturnParameterError;

  lah_index n = x->nR;           /* state dimension N */
  lah_index n_sigma = 2 * n + 1; /* number of sigma points */
  if (Xsig->nR != n || Xsig->nC != n_sigma)
    return lahReturnParameterError;
  if (S->nR != n || S->nC != n)
    return lahReturnParameterError;

  /* scaling factor γ = sqrt( N + λ ) */
  lah_value gamma = sqrt((lah_value)n + lambda);
  if (gamma <= 0.0)
    return lahReturnMathError;

  /* 1st column = mean state */
  for (lah_index i = 0; i < n; ++i)
    LAH_ENTRY(Xsig, i, 0) = LAH_ENTRY(x, i, 0);

  /* Remaining columns */
  for (lah_index k = 0; k < n; ++k) {
    for (lah_index i = 0; i < n; ++i) {
      /* +γ * S(:,k)  */
      LAH_ENTRY(Xsig, i, k + 1) =
          LAH_ENTRY(x, i, 0) + gamma * LAH_ENTRY(S, i, k);
      /* -γ * S(:,k)  */
      LAH_ENTRY(Xsig, i, k + 1 + n) =
          LAH_ENTRY(x, i, 0) - gamma * LAH_ENTRY(S, i, k);
    }
  }
  return lahReturnOk;
}

/* Set matrix V as a column view into column k of matrix M. */
static inline void lah_matSetColumnView(lah_mat *V, const lah_mat *M,
                                        lah_index k) {
  V->nR = M->nR;
  V->nC = 1;
  V->incRow = 1;     /* column vector */
  V->incCol = M->nR; /* distance to next column in M */
  V->data = M->data + k * M->incCol;
  V->matType = 0;
  LAH_SETTYPE(V, lahTypeColMajor);
}

static lah_Return propagate_sigma_points(const lah_mat *Xsig, lah_mat *Ysig,
                                         void (*func)(const lah_mat *,
                                                      lah_mat *, void *),
                                         void *user) {
  /* Basic sanity checks */
  if (!Xsig || !func || !Ysig)
    return lahReturnParameterError;

  if (Ysig->data == NULL)
    return lahReturnParameterError;
  if (Xsig->nC != Ysig->nC)
    return lahReturnParameterError;

  lah_index n_sigma = Xsig->nC; /* number of sigma points */

  /* Temporary matrix descriptors for a single column vector */
  lah_mat col_in, col_out;

  /* Loop over all sigma points */
  for (lah_index k = 0; k < n_sigma; ++k) {
    /* Point to the k‑th column of the input and output matrices */
    lah_matSetColumnView(&col_in, Xsig, k);
    lah_matSetColumnView(&col_out, Ysig, k);
    /* Call the user supplied model */
    func(&col_in, &col_out, user);
  }

  return lahReturnOk;
}

/* Compute square‑root covariance to full covariance matrix */
static lah_Return sqrt_to_covariance(const lah_mat *S, lah_mat *P) {
  /* Argument validation */
  if (!S || !P)
    return lahReturnParameterError;

  if (S->nR != S->nC || P->nR != P->nC)
    return lahReturnParameterError;

  if (S->nR != P->nR)
    return lahReturnParameterError;

  /* Compute P = S * Sᵀ */
  size_t n = S->nR;
  GEMM(LAH_CBLAS_LAYOUT, CblasNoTrans, CblasTrans, (int)n, (int)n, (int)n,
       (lah_value)1.0, S->data, (int)LAH_LEADING_DIM(S), S->data,
       (int)LAH_LEADING_DIM(S), (lah_value)0.0, P->data,
       (int)LAH_LEADING_DIM(P));

  return lahReturnOk;
}

/* --------------------------------------------------------------------------
 * SR-UKF Helper Functions
 * -------------------------------------------------------------------------- */

/* Cholesky rank-1 downdate: S = choldowndate(S, v)
 * Updates S such that S_new * S_new' = S * S' - v * v'
 * Uses Givens rotations for numerical stability.
 * S must be lower triangular (n x n), v is a vector of length n.
 * Returns lahReturnMathError if the downdate would make the matrix non-SPD.
 */
static lah_Return chol_downdate_rank1(lah_mat *S, const lah_value *v) {
  if (!S || !v)
    return lahReturnParameterError;

  lah_index n = S->nR;
  if (S->nC != n)
    return lahReturnParameterError;

  /* Work with a copy of v */
  lah_value *w = (lah_value *)malloc(n * sizeof(lah_value));
  if (!w)
    return lahReturnParameterError;
  memcpy(w, v, n * sizeof(lah_value));

  /* Apply Givens rotations to zero out w while updating S */
  for (lah_index j = 0; j < n; ++j) {
    lah_value Sjj = LAH_ENTRY(S, j, j);
    lah_value wj = w[j];

    /* Compute r = sqrt(Sjj^2 - wj^2) */
    lah_value r2 = Sjj * Sjj - wj * wj;
    if (r2 < 0.0) {
      /* Matrix would become non-SPD */
      free(w);
      return lahReturnMathError;
    }
    lah_value r = sqrt(r2);

    if (fabs(Sjj) < SR_UKF_EPS) {
      /* Skip if diagonal is essentially zero */
      if (fabs(wj) > SR_UKF_EPS) {
        free(w);
        return lahReturnMathError;
      }
      continue;
    }

    /* Givens rotation parameters: c = r/Sjj, s = wj/Sjj */
    lah_value c = r / Sjj;
    lah_value s = wj / Sjj;

    /* Update diagonal */
    LAH_ENTRY(S, j, j) = r;

    /* Update remaining rows and w */
    for (lah_index i = j + 1; i < n; ++i) {
      lah_value Sij = LAH_ENTRY(S, i, j);
      lah_value wi = w[i];

      /* Update S(i,j) and w(i) */
      LAH_ENTRY(S, i, j) = (Sij - s * wi) / c;
      w[i] = c * wi - s * Sij;
    }
  }

  free(w);
  return lahReturnOk;
}

/* Compute weighted deviations: Dev(:,k) = sqrt(|wc[k]|) * (Ysig(:,k) - mean)
 * Ysig : M x (2N+1) sigma points
 * mean : M x 1 weighted mean
 * wc   : (2N+1) covariance weights
 * Dev  : M x (2N+1) output deviations (same size as Ysig)
 *
 * Note: wc[0] can be negative for certain parameter choices (α < 1).
 * We handle this by using sqrt(|wc|) and tracking the sign separately
 * for the Cholesky update/downdate later.
 */
static lah_Return compute_weighted_deviations(const lah_mat *Ysig,
                                              const lah_mat *mean,
                                              const lah_value *wc,
                                              lah_mat *Dev) {
  if (!Ysig || !mean || !wc || !Dev)
    return lahReturnParameterError;

  lah_index M = Ysig->nR;
  lah_index n_sigma = Ysig->nC;

  if (mean->nR != M || mean->nC != 1)
    return lahReturnParameterError;
  if (Dev->nR != M || Dev->nC != n_sigma)
    return lahReturnParameterError;

  for (lah_index k = 0; k < n_sigma; ++k) {
    lah_value sw = sqrt(fabs(wc[k]));
    for (lah_index i = 0; i < M; ++i) {
      LAH_ENTRY(Dev, i, k) = sw * (LAH_ENTRY(Ysig, i, k) - LAH_ENTRY(mean, i, 0));
    }
  }

  return lahReturnOk;
}

/* Compute square-root covariance from weighted deviations using QR.
 * This is the core SR-UKF operation that avoids forming full covariance.
 *
 * The compound matrix is: [ sqrt(wc)*Dev' ; Noise_sqrt' ]
 *                         (2N+1 + M) x M  for predict
 *                         (2N+1 + N) x N  for correct
 *
 * We compute QR of this compound matrix and S = R' (lower triangular).
 *
 * Dev        : dim x (2N+1) weighted deviations (from compute_weighted_deviations)
 * Noise_sqrt : dim x dim noise square-root (Qsqrt or Rsqrt)
 * S          : dim x dim output square-root covariance
 * work       : workspace matrix, at least (2N+1 + dim) x dim
 * wc0_negative: if true, wc[0] < 0 and we need to do a downdate after QR
 * dev0       : first column of deviations (for potential downdate)
 */
static lah_Return sr_ukf_sqrt_from_deviations_ex(const lah_mat *Dev,
                                                  const lah_mat *Noise_sqrt,
                                                  lah_mat *S, lah_mat *work,
                                                  bool wc0_negative,
                                                  const lah_value *dev0) {
  if (!Dev || !Noise_sqrt || !S || !work)
    return lahReturnParameterError;

  lah_index dim = Dev->nR;      /* dimension of output S */
  lah_index n_sigma = Dev->nC;  /* 2N+1 */

  if (Noise_sqrt->nR != dim || Noise_sqrt->nC != dim)
    return lahReturnParameterError;
  if (S->nR != dim || S->nC != dim)
    return lahReturnParameterError;

  /* Build compound matrix for QR:
   * If wc0 >= 0: use all deviations
   * If wc0 < 0:  exclude first deviation column, downdate after
   *
   * Compound = [ Dev(:,start:end)' ; Noise_sqrt' ]
   * where start = wc0_negative ? 1 : 0
   */
  lah_index start_col = wc0_negative ? 1 : 0;
  lah_index n_dev_cols = n_sigma - start_col;
  lah_index n_rows = n_dev_cols + dim; /* rows in compound matrix */

  /* Verify workspace size */
  if (work->nR < n_rows || work->nC < dim)
    return lahReturnParameterError;

  /* Fill compound matrix (column-major):
   * First n_dev_cols rows: transpose of Dev(:,start:end)
   * Last dim rows: transpose of Noise_sqrt
   */
  for (lah_index j = 0; j < dim; ++j) {
    /* Dev' part */
    for (lah_index i = 0; i < n_dev_cols; ++i) {
      LAH_ENTRY(work, i, j) = LAH_ENTRY(Dev, j, i + start_col);
    }
    /* Noise_sqrt' part */
    for (lah_index i = 0; i < dim; ++i) {
      LAH_ENTRY(work, n_dev_cols + i, j) = LAH_ENTRY(Noise_sqrt, j, i);
    }
  }

  /* QR factorization to get R */
  lah_value *tau = (lah_value *)malloc(dim * sizeof(lah_value));
  if (!tau)
    return lahReturnParameterError;

  int info = GEQRF(LAH_LAPACK_LAYOUT, (int)n_rows, (int)dim, work->data,
                   (int)LAH_LEADING_DIM(work), tau);
  free(tau);
  if (info != 0)
    return lahReturnMathError;

  /* Extract R' (transpose of upper triangular R) into S as lower triangular */
  for (lah_index i = 0; i < dim; ++i) {
    for (lah_index j = 0; j < dim; ++j) {
      if (j <= i) {
        /* S(i,j) = R(j,i) where R is upper triangular in work */
        LAH_ENTRY(S, i, j) = LAH_ENTRY(work, j, i);
      } else {
        LAH_ENTRY(S, i, j) = 0.0;
      }
    }
  }

  /* Ensure positive diagonal (QR can give negative diagonal elements) */
  for (lah_index i = 0; i < dim; ++i) {
    if (LAH_ENTRY(S, i, i) < 0.0) {
      /* Flip sign of entire column */
      for (lah_index k = i; k < dim; ++k)
        LAH_ENTRY(S, k, i) = -LAH_ENTRY(S, k, i);
    }
  }

  /* If wc[0] was negative, we need to downdate S with dev0 */
  if (wc0_negative && dev0) {
    lah_Return ret = chol_downdate_rank1(S, dev0);
    if (ret != lahReturnOk)
      return ret;
  }

  return lahReturnOk;
}

/* Compute weighted mean from sigma points.
 * Ysig : M x (2N+1) sigma points
 * wm   : (2N+1) mean weights
 * mean : M x 1 output mean
 */
static lah_Return compute_weighted_mean(const lah_mat *Ysig,
                                         const lah_value *wm,
                                         lah_mat *mean) {
  if (!Ysig || !wm || !mean)
    return lahReturnParameterError;

  lah_index M = Ysig->nR;
  lah_index n_sigma = Ysig->nC;

  if (mean->nR != M || mean->nC != 1)
    return lahReturnParameterError;

  for (lah_index i = 0; i < M; ++i)
    LAH_ENTRY(mean, i, 0) = 0.0;

  for (lah_index k = 0; k < n_sigma; ++k) {
    lah_value wk = wm[k];
    for (lah_index i = 0; i < M; ++i)
      LAH_ENTRY(mean, i, 0) += wk * LAH_ENTRY(Ysig, i, k);
  }

  return lahReturnOk;
}

/* --------------------------------------------------------------------------
 * End SR-UKF Helper Functions
 * -------------------------------------------------------------------------- */

/* Compute cross‑covariance matrix between two sets of sigma points
 * Xsig    : N × (2N+1)   – state sigma points
 * Ysig    : M × (2N+1)   – measurement sigma points
 * x_mean  : N × 1        – weighted state mean
 * y_mean  : M × 1        – weighted measurement mean
 * weights : array (2N+1) – covariance weights wc
 * Pxz     : N × M        – output cross‑covariance (allocated by caller)
 */
static lah_Return
compute_cross_covariance(const lah_mat *Xsig, const lah_mat *Ysig,
                         const lah_mat *x_mean, const lah_mat *y_mean,
                         const lah_value *weights, lah_mat *Pxz) {
  if (!Xsig || !Ysig || !x_mean || !y_mean || !weights || !Pxz)
    return lahReturnParameterError;

  if (Xsig->nR != Pxz->nR || Ysig->nR != Pxz->nC)
    return lahReturnParameterError;

  if (Xsig->nC != Ysig->nC)
    return lahReturnParameterError;

  /* zero‑initialize */
  for (lah_index i = 0; i < Pxz->nR; ++i)
    for (lah_index j = 0; j < Pxz->nC; ++j)
      LAH_ENTRY(Pxz, i, j) = 0.0;

  /* weighted outer products */
  for (lah_index k = 0; k < Xsig->nC; ++k) {
    lah_value wk = weights[k];
    for (lah_index i = 0; i < Xsig->nR; ++i) {
      lah_value xi = LAH_ENTRY(Xsig, i, k) - LAH_ENTRY(x_mean, i, 0);
      for (lah_index j = 0; j < Ysig->nR; ++j) {
        lah_value yj = LAH_ENTRY(Ysig, j, k) - LAH_ENTRY(y_mean, j, 0);
        LAH_ENTRY(Pxz, i, j) += wk * xi * yj;
      }
    }
  }
  return lahReturnOk;
}

/* ---------- sr_ukf_predict_core ----------
 * Core prediction using proper SR-UKF with QR decomposition.
 * Reads from x_in/S_in, writes to x_out/S_out.
 * ukf is used for parameters (lambda, weights, Qsqrt) and workspace.
 * Caller must ensure workspace is allocated (via ensure_workspace).
 * x_in/S_in and x_out/S_out may alias (in-place operation).
 *
 * The SR-UKF approach computes S directly via:
 *   S = qr([sqrt(wc)*Dev'; Qsqrt'])'
 * where Dev(:,k) = Ysig(:,k) - y_mean
 *
 * This avoids forming full covariance and maintains numerical stability.
 */
static lah_Return
sr_ukf_predict_core(const sr_ukf *ukf, const lah_mat *x_in, const lah_mat *S_in,
                    void (*f)(const lah_mat *, lah_mat *, void *), void *user,
                    lah_mat *x_out, lah_mat *S_out) {
  lah_Return ret = lahReturnOk;
  if (!ukf || !f || !x_in || !S_in || !x_out || !S_out)
    return lahReturnParameterError;
  if (!ukf->Qsqrt || !ukf->wm || !ukf->wc || !ukf->ws)
    return lahReturnParameterError;

  /* --- Dimensions --------------------------------------------------- */
  lah_index N = x_in->nR; /* state dimension */

  /* --- Validate output dimensions ---------------------------------- */
  if (x_out->nR != N || x_out->nC != 1)
    return lahReturnParameterError;
  if (S_out->nR != N || S_out->nC != N)
    return lahReturnParameterError;

  /* --- Use workspace temporaries ----------------------------------- */
  sr_ukf_workspace *ws = ukf->ws;
  lah_mat *Xsig = ws->Xsig;
  lah_mat *Ysig = ws->Ysig_N;
  lah_mat *x_mean = ws->x_pred;
  lah_mat *Dev = ws->Dev_N;
  lah_mat *qr_work = ws->qr_work_N;

  /* --- Generate and propagate sigma points ------------------------ */
  ret = generate_sigma_points_from(x_in, S_in, ukf->lambda, Xsig);
  if (ret != lahReturnOk)
    return ret;

  ret = propagate_sigma_points(Xsig, Ysig, f, user);
  if (ret != lahReturnOk)
    return ret;

  /* --- Validate callback output ----------------------------------- */
  if (!is_numeric_valid(Ysig)) {
    diag_report("predict: callback f produced NaN or Inf");
    return lahReturnMathError;
  }

  /* --- Compute weighted mean --------------------------------------- */
  ret = compute_weighted_mean(Ysig, ukf->wm, x_mean);
  if (ret != lahReturnOk)
    return ret;

  /* --- Compute weighted deviations --------------------------------- */
  ret = compute_weighted_deviations(Ysig, x_mean, ukf->wc, Dev);
  if (ret != lahReturnOk)
    return ret;

  /* --- Compute S via QR of [Dev'; Qsqrt'] -------------------------- */
  /* Handle potential negative wc[0] (for alpha < 1) */
  bool wc0_negative = (ukf->wc[0] < 0.0);
  lah_value *dev0 = NULL;
  if (wc0_negative) {
    /* Save first column of Dev for downdate */
    dev0 = (lah_value *)malloc(N * sizeof(lah_value));
    if (!dev0)
      return lahReturnParameterError;
    for (lah_index i = 0; i < N; ++i)
      dev0[i] = LAH_ENTRY(Dev, i, 0);
  }

  ret = sr_ukf_sqrt_from_deviations_ex(Dev, ukf->Qsqrt, S_out, qr_work,
                                        wc0_negative, dev0);
  if (dev0)
    free(dev0);
  if (ret != lahReturnOk)
    return ret;

  /* --- Write mean → x_out ---------------------------------------- */
  memcpy(x_out->data, x_mean->data, N * sizeof(lah_value));

  return lahReturnOk;
}

/* ---------- sr_ukf_predict_to ----------
 * Transactional predict: operates in-place on user-provided x/S.
 * Uses cached workspace for temporaries (allocated on demand).
 */
lah_Return sr_ukf_predict_to(sr_ukf *ukf, lah_mat *x, lah_mat *S,
                             void (*f)(const lah_mat *, lah_mat *, void *),
                             void *user) {
  if (!ukf || !x || !S)
    return lahReturnParameterError;

  /* Validate dimensions match filter */
  lah_index N = sr_ukf_state_dim(ukf);
  if (N == 0)
    return lahReturnParameterError;
  if (x->nR != N || x->nC != 1)
    return lahReturnParameterError;
  if (S->nR != N || S->nC != N)
    return lahReturnParameterError;

  /* Ensure workspace is allocated */
  lah_Return ret = ensure_workspace(ukf);
  if (ret != lahReturnOk)
    return ret;

  return sr_ukf_predict_core(ukf, x, S, f, user, x, S);
}

/* ---------- sr_ukf_predict ----------
 * Safe predict: atomic update of ukf->x and ukf->S.
 * On error, filter state is unchanged.
 */
lah_Return sr_ukf_predict(sr_ukf *ukf,
                          void (*f)(const lah_mat *, lah_mat *, void *),
                          void *user) {
  if (!ukf || !ukf->x || !ukf->S)
    return lahReturnParameterError;

  /* Ensure workspace is allocated */
  lah_Return ret = ensure_workspace(ukf);
  if (ret != lahReturnOk)
    return ret;

  lah_index N = ukf->x->nR;

  /* Use workspace for output temporaries */
  lah_mat *x_out = ukf->ws->x_pred;
  lah_mat *S_out = ukf->ws->S_tmp;

  /* Run core: read from ukf->x/S, write to temps */
  ret = sr_ukf_predict_core(ukf, ukf->x, ukf->S, f, user, x_out, S_out);

  /* Commit on success */
  if (ret == lahReturnOk) {
    memcpy(ukf->x->data, x_out->data, N * sizeof(lah_value));
    memcpy(ukf->S->data, S_out->data, N * N * sizeof(lah_value));
  }

  return ret;
}

/* ---------- sr_ukf_correct_core ----------
 * Core correction using proper SR-UKF with QR and Cholesky downdates.
 * Reads from x_in/S_in, writes to x_out/S_out.
 * ukf is used for parameters (lambda, weights, Rsqrt) and workspace.
 * Caller must ensure workspace is allocated (via ensure_workspace).
 * x_in/S_in and x_out/S_out may alias (in-place operation).
 *
 * The SR-UKF approach:
 *   1. Compute Syy via QR of [sqrt(wc)*Dev_M'; Rsqrt']
 *   2. Compute K = Pxz * Syy^{-T} * Syy^{-1} via triangular solves
 *   3. Update x_new = x_in + K * innovation
 *   4. Update S via Cholesky downdates: S_new² = S_in² - (K*Syy)(K*Syy)'
 *
 * This avoids forming full covariance and maintains numerical stability.
 */
static lah_Return
sr_ukf_correct_core(const sr_ukf *ukf, const lah_mat *x_in, const lah_mat *S_in,
                    lah_mat *z, void (*h)(const lah_mat *, lah_mat *, void *),
                    void *user, lah_mat *x_out, lah_mat *S_out) {
  lah_Return ret = lahReturnOk;
  if (!ukf || !h || !z || !x_in || !S_in || !x_out || !S_out)
    return lahReturnParameterError;
  if (!ukf->Rsqrt || !ukf->wm || !ukf->wc || !ukf->ws)
    return lahReturnParameterError;

  lah_index N = x_in->nR;       /* state dimension */
  lah_index M = ukf->Rsqrt->nR; /* measurement dimension */

  if (z->nR != M || z->nC != 1)
    return lahReturnParameterError;
  if (x_out->nR != N || x_out->nC != 1)
    return lahReturnParameterError;
  if (S_out->nR != N || S_out->nC != N)
    return lahReturnParameterError;

  /* --- Use workspace temporaries ----------------------------------- */
  sr_ukf_workspace *ws = ukf->ws;
  lah_mat *Xsig = ws->Xsig;
  lah_mat *Ysig = ws->Ysig_M;
  lah_mat *x_mean = ws->x_pred;
  lah_mat *y_mean = ws->y_mean;
  lah_mat *Syy = ws->Syy;
  lah_mat *Pxz = ws->Pxz;
  lah_mat *K = ws->K;
  lah_mat *innov = ws->innov;
  lah_mat *x_new = ws->x_new;
  lah_mat *dx = ws->dx;
  lah_mat *Dev_M = ws->Dev_M;
  lah_mat *qr_work = ws->qr_work_M;
  lah_mat *tmp1 = ws->tmp1; /* Used for K*Syy (N x M) */

  /* 1. Generate & propagate σ‑points -------------------------------- */
  ret = generate_sigma_points_from(x_in, S_in, ukf->lambda, Xsig);
  if (ret != lahReturnOk)
    return ret;
  ret = propagate_sigma_points(Xsig, Ysig, h, user);
  if (ret != lahReturnOk)
    return ret;

  /* --- Validate callback output ----------------------------------- */
  if (!is_numeric_valid(Ysig)) {
    diag_report("correct: callback h produced NaN or Inf");
    return lahReturnMathError;
  }

  /* 2. Copy prior state mean for cross-covariance computation ------ */
  for (lah_index i = 0; i < N; ++i)
    LAH_ENTRY(x_mean, i, 0) = LAH_ENTRY(x_in, i, 0);

  /* 3. Compute weighted mean for measurement ------------------------- */
  ret = compute_weighted_mean(Ysig, ukf->wm, y_mean);
  if (ret != lahReturnOk)
    return ret;

  /* 4. Compute weighted deviations for measurement ------------------- */
  ret = compute_weighted_deviations(Ysig, y_mean, ukf->wc, Dev_M);
  if (ret != lahReturnOk)
    return ret;

  /* 5. Compute Syy via QR of [sqrt(wc)*Dev_M'; Rsqrt'] --------------- */
  bool wc0_negative = (ukf->wc[0] < 0.0);
  lah_value *dev0_M = NULL;
  if (wc0_negative) {
    dev0_M = (lah_value *)malloc(M * sizeof(lah_value));
    if (!dev0_M)
      return lahReturnParameterError;
    for (lah_index i = 0; i < M; ++i)
      dev0_M[i] = LAH_ENTRY(Dev_M, i, 0);
  }

  ret = sr_ukf_sqrt_from_deviations_ex(Dev_M, ukf->Rsqrt, Syy, qr_work,
                                        wc0_negative, dev0_M);
  if (dev0_M)
    free(dev0_M);
  if (ret != lahReturnOk)
    return ret;

  /* --- Check if Syy is essentially zero ----------------------------- */
  bool Syy_zero = true;
  for (lah_index i = 0; i < M && Syy_zero; ++i)
    if (fabs(LAH_ENTRY(Syy, i, i)) > SR_UKF_EPS)
      Syy_zero = false;
  if (Syy_zero) {
    memcpy(x_out->data, x_in->data, N * sizeof(lah_value));
    memcpy(S_out->data, S_in->data, N * N * sizeof(lah_value));
    return lahReturnOk;
  }

  /* 6. Cross‑covariance between state & measurement σ‑points -------- */
  ret = compute_cross_covariance(Xsig, Ysig, x_mean, y_mean, ukf->wc, Pxz);
  if (ret != lahReturnOk)
    return ret;

  /* 7. Kalman gain K = Pxz * inv(Syy') * inv(Syy) -------------------- */
  /* We have Syy as lower triangular.
   * K = Pxz * (Syy * Syy')^{-1} = Pxz * Syy'^{-1} * Syy^{-1}
   *
   * Using BLAS triangular solve:
   * Step 1: K = Pxz * Syy'^{-1}  → solve K * Syy' = Pxz (Syy' is upper tri)
   *         TRSM: side=Right, uplo=Lower, trans=Trans → K * Syy' = Pxz
   * Step 2: K = K * Syy^{-1}     → solve K * Syy = K (Syy is lower tri)
   *         TRSM: side=Right, uplo=Lower, trans=NoTrans → K * Syy = K
   */
  memcpy(K->data, Pxz->data, N * M * sizeof(lah_value));

  /* K * Syy' = Pxz → K = Pxz * Syy'^{-1} */
  TRSM(LAH_CBLAS_LAYOUT, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
       (int)N, (int)M, (lah_value)1.0, Syy->data, (int)LAH_LEADING_DIM(Syy),
       K->data, (int)LAH_LEADING_DIM(K));

  /* K * Syy = K → K = K * Syy^{-1} */
  TRSM(LAH_CBLAS_LAYOUT, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
       (int)N, (int)M, (lah_value)1.0, Syy->data, (int)LAH_LEADING_DIM(Syy),
       K->data, (int)LAH_LEADING_DIM(K));

  /* 8. Innovation ------------------------------------------------------- */
  for (lah_index i = 0; i < M; ++i)
    LAH_ENTRY(innov, i, 0) = LAH_ENTRY(z, i, 0) - LAH_ENTRY(y_mean, i, 0);

  /* 9. State update → x_new -------------------------------------------- */
  ret = lah_matMul(lahNorm, lahNorm, (lah_value)0.0, (lah_value)1.0, dx, K,
                   innov);
  if (ret != lahReturnOk)
    return ret;
  for (lah_index i = 0; i < N; ++i)
    LAH_ENTRY(x_new, i, 0) = LAH_ENTRY(x_in, i, 0) + LAH_ENTRY(dx, i, 0);

  /* 10. Covariance update via Cholesky downdates ----------------------- */
  /* S_out² = S_in² - K * Syy * (K * Syy)'
   * Let U = K * Syy (N x M), then S_out² = S_in² - U * U'
   * This is M rank-1 downdates using columns of U.
   */

  /* First, copy S_in to S_out */
  memcpy(S_out->data, S_in->data, N * N * sizeof(lah_value));

  /* Compute U = K * Syy (N x M) */
  GEMM(LAH_CBLAS_LAYOUT, CblasNoTrans, CblasNoTrans, (int)N, (int)M, (int)M,
       (lah_value)1.0, K->data, (int)LAH_LEADING_DIM(K), Syy->data,
       (int)LAH_LEADING_DIM(Syy), (lah_value)0.0, tmp1->data,
       (int)LAH_LEADING_DIM(tmp1));

  /* Apply M successive rank-1 Cholesky downdates */
  lah_value *u_col = (lah_value *)malloc(N * sizeof(lah_value));
  if (!u_col)
    return lahReturnParameterError;

  for (lah_index j = 0; j < M; ++j) {
    /* Extract column j of U */
    for (lah_index i = 0; i < N; ++i)
      u_col[i] = LAH_ENTRY(tmp1, i, j);

    /* Perform rank-1 downdate */
    ret = chol_downdate_rank1(S_out, u_col);
    if (ret != lahReturnOk) {
      free(u_col);
      diag_report("correct: Cholesky downdate failed, matrix not SPD");
      return ret;
    }
  }
  free(u_col);

  /* 11. Write state output --------------------------------------------- */
  memcpy(x_out->data, x_new->data, N * sizeof(lah_value));

  return lahReturnOk;
}

/* ---------- sr_ukf_correct_to ----------
 * Transactional correct: operates in-place on user-provided x/S.
 * Uses cached workspace for temporaries (allocated on demand).
 */
lah_Return sr_ukf_correct_to(sr_ukf *ukf, lah_mat *x, lah_mat *S, lah_mat *z,
                             void (*h)(const lah_mat *, lah_mat *, void *),
                             void *user) {
  if (!ukf || !x || !S || !z)
    return lahReturnParameterError;

  /* Validate dimensions match filter */
  lah_index N = sr_ukf_state_dim(ukf);
  lah_index M = sr_ukf_meas_dim(ukf);
  if (N == 0 || M == 0)
    return lahReturnParameterError;
  if (x->nR != N || x->nC != 1)
    return lahReturnParameterError;
  if (S->nR != N || S->nC != N)
    return lahReturnParameterError;
  if (z->nR != M || z->nC != 1)
    return lahReturnParameterError;

  /* Ensure workspace is allocated */
  lah_Return ret = ensure_workspace(ukf);
  if (ret != lahReturnOk)
    return ret;

  return sr_ukf_correct_core(ukf, x, S, z, h, user, x, S);
}

/* ---------- sr_ukf_correct ----------
 * Safe correct: atomic update of ukf->x and ukf->S.
 * On error, filter state is unchanged.
 */
lah_Return sr_ukf_correct(sr_ukf *ukf, lah_mat *z,
                          void (*h)(const lah_mat *, lah_mat *, void *),
                          void *user) {
  if (!ukf || !ukf->x || !ukf->S)
    return lahReturnParameterError;

  /* Ensure workspace is allocated */
  lah_Return ret = ensure_workspace(ukf);
  if (ret != lahReturnOk)
    return ret;

  lah_index N = ukf->x->nR;

  /* Use workspace for output temporaries */
  lah_mat *x_out = ukf->ws->x_new;
  lah_mat *S_out = ukf->ws->S_new;

  /* Run core: read from ukf->x/S, write to temps */
  ret = sr_ukf_correct_core(ukf, ukf->x, ukf->S, z, h, user, x_out, S_out);

  /* Commit on success */
  if (ret == lahReturnOk) {
    memcpy(ukf->x->data, x_out->data, N * sizeof(lah_value));
    memcpy(ukf->S->data, S_out->data, N * N * sizeof(lah_value));
  }

  return ret;
}
