/* workspace.c -- pre-allocated temporaries for the hot path
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

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
  /** @} */

  /** @name Correct Temporaries
   *  @{ */
  srukf_mat *Ysig_M; /**< M x (2N+1) - sigma points in measurement space */
  srukf_mat *y_mean; /**< M x 1 - predicted measurement mean */
  srukf_mat *Pxz; /**< N x M - cross-covariance between state and measurement */
  srukf_mat *K;   /**< N x M - Kalman gain */
  srukf_mat *innov; /**< M x 1 - innovation (z - z_predicted) */
  srukf_mat *x_new; /**< N x 1 - updated state (for atomic update) */
  srukf_mat *S_new; /**< N x N - updated sqrt-covariance (for atomic update) */
  srukf_mat *dx;    /**< N x 1 - state correction K * innovation */
  srukf_mat *tmp1;  /**< N x M - temp for K * Syy product */
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

  /** @name Commit Staging
   *  Results are computed here and copied to their destination only on
   *  success, giving every predict/correct variant transactional
   *  semantics. The cores never touch these buffers internally.
   *  @{ */
  srukf_mat *x_stage; /**< N x 1 - staged state output */
  /** S staging reuses S_tmp (predict) and S_new (correct). */
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
  srukf_value *lapack_work; /**< LAPACK QR work buffer (lwork elements),
                                 queried at allocation so GEQRF never
                                 mallocs in the hot path */
  int lwork;                /**< Size of lapack_work */
  /** @} */

  /** @name Innovation Bookkeeping
   *  @{ */
  bool correct_valid; /**< innov/Syy describe a completed correct step */
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
  srukf_mat_free(ws->Ysig_M);
  srukf_mat_free(ws->y_mean);
  srukf_mat_free(ws->Pxz);
  srukf_mat_free(ws->K);
  srukf_mat_free(ws->innov);
  srukf_mat_free(ws->x_new);
  srukf_mat_free(ws->S_new);
  srukf_mat_free(ws->dx);
  srukf_mat_free(ws->tmp1);
  srukf_mat_free(ws->Dev_N);
  srukf_mat_free(ws->Dev_M);
  srukf_mat_free(ws->qr_work_N);
  srukf_mat_free(ws->qr_work_M);
  srukf_mat_free(ws->Syy);
  srukf_mat_free(ws->x_stage);

  /* Free pre-allocated buffers */
  free(ws->tau_N);
  free(ws->tau_M);
  free(ws->downdate_work);
  free(ws->dev0_N);
  free(ws->dev0_M);
  free(ws->lapack_work);

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
    return SRUKF_RETURN_MEMORY_ERROR;

  ws->N = N;
  ws->M = M;

  /* Allocate all matrices */
  ws->Xsig = SRUKF_MAT_ALLOC(N, n_sigma);
  ws->Ysig_N = SRUKF_MAT_ALLOC(N, n_sigma);
  ws->x_pred = SRUKF_MAT_ALLOC(N, 1);
  ws->S_tmp = SRUKF_MAT_ALLOC(N, N);
  ws->Ysig_M = SRUKF_MAT_ALLOC(M, n_sigma);
  ws->y_mean = SRUKF_MAT_ALLOC(M, 1);
  ws->Pxz = SRUKF_MAT_ALLOC(N, M);
  ws->K = SRUKF_MAT_ALLOC(N, M);
  ws->innov = SRUKF_MAT_ALLOC(M, 1);
  ws->x_new = SRUKF_MAT_ALLOC(N, 1);
  ws->S_new = SRUKF_MAT_ALLOC(N, N);
  ws->dx = SRUKF_MAT_ALLOC(N, 1);
  ws->tmp1 = SRUKF_MAT_ALLOC(N, M);

  /* SR-UKF specific matrices */
  ws->Dev_N = SRUKF_MAT_ALLOC(N, n_sigma);
  ws->Dev_M = SRUKF_MAT_ALLOC(M, n_sigma);
  /* QR compound matrix: at most n_sigma deviation rows (all of them
   * when wc[0] >= 0) plus dim noise rows. */
  ws->qr_work_N = SRUKF_MAT_ALLOC(n_sigma + N, N);
  ws->qr_work_M = SRUKF_MAT_ALLOC(n_sigma + M, M);
  ws->Syy = SRUKF_MAT_ALLOC(M, M);
  ws->x_stage = SRUKF_MAT_ALLOC(N, 1);

  /* Pre-allocated buffers for hot path (avoid malloc in predict/correct) */
  ws->tau_N = (srukf_value *)calloc(N, sizeof(srukf_value));
  ws->tau_M = (srukf_value *)calloc(M, sizeof(srukf_value));
  srukf_index max_dim = (N > M) ? N : M;
  ws->downdate_work = (srukf_value *)calloc(max_dim, sizeof(srukf_value));
  ws->dev0_N = (srukf_value *)calloc(N, sizeof(srukf_value));
  ws->dev0_M = (srukf_value *)calloc(M, sizeof(srukf_value));

  /* Check all allocations succeeded */
  if (!ws->Xsig || !ws->Ysig_N || !ws->x_pred || !ws->S_tmp || !ws->Ysig_M ||
      !ws->y_mean || !ws->Pxz || !ws->K || !ws->innov || !ws->x_new ||
      !ws->S_new || !ws->dx || !ws->tmp1 || !ws->Dev_N || !ws->Dev_M ||
      !ws->qr_work_N || !ws->qr_work_M || !ws->Syy || !ws->x_stage ||
      !ws->tau_N || !ws->tau_M || !ws->downdate_work || !ws->dev0_N ||
      !ws->dev0_M) {
    ukf->ws = ws; /* Temporarily assign so free_workspace can clean up */
    srukf_free_workspace(ukf);
    return SRUKF_RETURN_MEMORY_ERROR;
  }

  /* Size the LAPACK QR work buffer once (lwork = -1 is a query), so the
   * hot path can use the _work variant, which never mallocs. The
   * requirement grows with the column count; take the max of both QR
   * shapes used by predict and correct. */
  srukf_value wq = 0;
  int lw = 0;
  int info =
      SRUKF_GEQRF(SRUKF_LAPACK_LAYOUT, (int)ws->qr_work_N->n_rows, (int)N,
                  ws->qr_work_N->data, (int)SRUKF_LEADING_DIM(ws->qr_work_N),
                  ws->tau_N, &wq, -1);
  if (info == 0)
    lw = (int)wq;
  info = SRUKF_GEQRF(SRUKF_LAPACK_LAYOUT, (int)ws->qr_work_M->n_rows, (int)M,
                     ws->qr_work_M->data, (int)SRUKF_LEADING_DIM(ws->qr_work_M),
                     ws->tau_M, &wq, -1);
  if (info == 0 && (int)wq > lw)
    lw = (int)wq;
  if (lw < 1) {
    ukf->ws = ws;
    srukf_free_workspace(ukf);
    return SRUKF_RETURN_PARAMETER_ERROR;
  }
  ws->lapack_work = (srukf_value *)calloc((size_t)lw, sizeof(srukf_value));
  ws->lwork = lw;
  if (!ws->lapack_work) {
    ukf->ws = ws;
    srukf_free_workspace(ukf);
    return SRUKF_RETURN_MEMORY_ERROR;
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
