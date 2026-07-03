/* lifecycle.c -- creation, destruction, and noise configuration
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

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

  /* On any failure we return with whatever was allocated still attached
   * to the (calloc'd) struct; the callers respond to failure with
   * srukf_free(), which owns ALL cleanup. Freeing anything here would
   * leave a dangling pointer behind for srukf_free to free again. */

  /* ----------------- State vector --------------------------------- */
  ukf->x = SRUKF_MAT_ALLOC(N, 1);
  if (!ukf->x)
    return SRUKF_RETURN_PARAMETER_ERROR;
  SRUKF_SET_TYPE(ukf->x, SRUKF_TYPE_COL_MAJOR);

  /* ----------------- State covariance square‑root ----------------- */
  ukf->S = SRUKF_MAT_ALLOC(N, N);
  if (!ukf->S)
    return SRUKF_RETURN_PARAMETER_ERROR;
  SRUKF_SET_TYPE(ukf->S, SRUKF_TYPE_SQUARE | SRUKF_TYPE_COL_MAJOR);
  /* initialize only the diagonal so that correction
   * can occur before prediction */
  for (srukf_index i = 0; i < (srukf_index)N; ++i)
    for (srukf_index j = 0; j < (srukf_index)N; ++j)
      SRUKF_ENTRY(ukf->S, i, j) = (i == j) ? DEFAULT_INIT_STD : 0.0;

  /* ---------- Process‑noise ---------- */
  ukf->Qsqrt =
      Qsqrt_src ? SRUKF_MAT_ALLOC(N, N) : SRUKF_MAT_ALLOC_NO_DATA(N, N);
  if (!ukf->Qsqrt)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Qsqrt_src)
    for (srukf_index j = 0; j < (srukf_index)N; ++j)
      for (srukf_index i = 0; i < (srukf_index)N; ++i)
        SRUKF_ENTRY(ukf->Qsqrt, i, j) = SRUKF_ENTRY(Qsqrt_src, i, j);
  SRUKF_SET_TYPE(ukf->Qsqrt, SRUKF_TYPE_SQUARE | SRUKF_TYPE_COL_MAJOR);

  /* ---------- Measurement‑noise ---------- */
  ukf->Rsqrt =
      Rsqrt_src ? SRUKF_MAT_ALLOC(M, M) : SRUKF_MAT_ALLOC_NO_DATA(M, M);
  if (!ukf->Rsqrt)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Rsqrt_src)
    for (srukf_index j = 0; j < (srukf_index)M; ++j)
      for (srukf_index i = 0; i < (srukf_index)M; ++i)
        SRUKF_ENTRY(ukf->Rsqrt, i, j) = SRUKF_ENTRY(Rsqrt_src, i, j);
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

  /* Ensure destination buffers exist: srukf_create() leaves the noise
   * matrices as NO_DATA descriptors, so the first set_noise supplies
   * them. Allocate both before attaching either, keeping the call
   * atomic on failure. Once the buffers exist, updating the noise is a
   * pure copy -- it cannot fail and allocates nothing, so per-step
   * adaptive noise preserves the zero-allocation guarantee. */
  srukf_value *qbuf = NULL;
  srukf_value *rbuf = NULL;
  if (!ukf->Qsqrt->data) {
    qbuf = (srukf_value *)calloc(N * N, sizeof(srukf_value));
    if (!qbuf)
      return SRUKF_RETURN_MEMORY_ERROR;
  }
  if (!ukf->Rsqrt->data) {
    rbuf = (srukf_value *)calloc(M * M, sizeof(srukf_value));
    if (!rbuf) {
      free(qbuf);
      return SRUKF_RETURN_MEMORY_ERROR;
    }
  }
  if (qbuf) {
    ukf->Qsqrt->data = qbuf;
    SRUKF_UNSET_TYPE(ukf->Qsqrt, SRUKF_TYPE_NO_DATA);
  }
  if (rbuf) {
    ukf->Rsqrt->data = rbuf;
    SRUKF_UNSET_TYPE(ukf->Rsqrt, SRUKF_TYPE_NO_DATA);
  }

  /* Copy element-wise: the source may carry arbitrary strides. */
  for (srukf_index j = 0; j < N; ++j)
    for (srukf_index i = 0; i < N; ++i)
      SRUKF_ENTRY(ukf->Qsqrt, i, j) = SRUKF_ENTRY(Qsqrt, i, j);

  for (srukf_index j = 0; j < M; ++j)
    for (srukf_index i = 0; i < M; ++i)
      SRUKF_ENTRY(ukf->Rsqrt, i, j) = SRUKF_ENTRY(Rsqrt, i, j);

  return SRUKF_RETURN_OK;
}
