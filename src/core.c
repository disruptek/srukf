/* core.c -- QR sqrt-covariance updates and Cholesky downdates
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

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
 * which are maximally stable. Failure (a non-SPD result) is detected by
 * requiring r^2 = S[j,j]^2 - work[j]^2 to stay positive *relative to*
 * S[j,j]^2: an r^2 below SRUKF_EPS * S[j,j]^2 -- negative, zero, or merely
 * vanishing -- would divide the rest of the column by (nearly) zero.
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

    if (SRUKF_FABS(Sjj) < SRUKF_EPS) {
      /* Essentially-zero diagonal: nothing to rotate against. Only a
       * likewise-zero downdate component is representable. */
      if (SRUKF_FABS(wj) > SRUKF_EPS)
        return SRUKF_RETURN_MATH_ERROR;
      continue;
    }

    /* r^2 = Sjj^2 - wj^2. Fail on indefinite (r2 < 0) AND on near-total
     * cancellation: c = r/Sjj is the divisor for the rest of the column,
     * so r2 vanishing relative to Sjj^2 means amplifying it by ~1/sqrt(r2)
     * -- at exact cancellation, dividing by zero. The result would not be
     * a usable Cholesky factor either way. */
    srukf_value r2 = Sjj * Sjj - wj * wj;
    if (r2 <= SRUKF_EPS * Sjj * Sjj)
      return SRUKF_RETURN_MATH_ERROR;
    srukf_value r = SRUKF_SQRT(r2);

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
 * @param res Custom residual hook, or NULL for Euclidean a - b
 * @param res_ctx Context for the residual hook
 * @param Dev Output deviations (dim x (2N+1))
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return compute_weighted_deviations(const srukf_mat *Ysig,
                                                const srukf_mat *mean,
                                                const srukf_value *wc,
                                                srukf_residual_fn res,
                                                void *res_ctx, srukf_mat *Dev) {
  if (!Ysig || !mean || !wc || !Dev)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index M = Ysig->n_rows;
  srukf_index n_sigma = Ysig->n_cols;

  if (mean->n_rows != M || mean->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Dev->n_rows != M || Dev->n_cols != n_sigma)
    return SRUKF_RETURN_PARAMETER_ERROR;

  for (srukf_index k = 0; k < n_sigma; ++k) {
    srukf_value sw = SRUKF_SQRT(SRUKF_FABS(wc[k]));
    if (res) {
      /* Residual into the Dev column (zero-copy view), then scale. */
      srukf_mat sig_col, dev_col;
      srukf_mat_column_view(&sig_col, Ysig, k);
      srukf_mat_column_view(&dev_col, Dev, k);
      res(&sig_col, mean, &dev_col, res_ctx);
      for (srukf_index i = 0; i < M; ++i)
        SRUKF_ENTRY(Dev, i, k) *= sw;
    } else {
      for (srukf_index i = 0; i < M; ++i) {
        SRUKF_ENTRY(Dev, i, k) =
            sw * (SRUKF_ENTRY(Ysig, i, k) - SRUKF_ENTRY(mean, i, 0));
      }
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
 * @param lapack_work LAPACK QR scratch (lwork elements)
 * @param lwork Size of lapack_work (from the workspace-sizing query)
 * @param wc0_negative True if wc[0] < 0 (requires downdate)
 * @param dev0 First deviation column (for downdate, NULL if wc0 >= 0)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return
srukf_sqrt_from_deviations_ex(const srukf_mat *Dev, const srukf_mat *Noise_sqrt,
                              srukf_mat *S, srukf_mat *work, srukf_value *tau,
                              srukf_value *downdate_work,
                              srukf_value *lapack_work, int lwork,
                              bool wc0_negative, const srukf_value *dev0) {
  if (!Dev || !Noise_sqrt || !S || !work || !tau || !lapack_work)
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

  /* QR factorization to get R. The _work variant uses the caller's
   * scratch buffer, so no heap allocation happens here. */
  int info = SRUKF_GEQRF(SRUKF_LAPACK_LAYOUT, (int)n_rows, (int)dim, work->data,
                         (int)SRUKF_LEADING_DIM(work), tau, lapack_work, lwork);
  if (info != 0) {
    diag_report(NULL, "QR factorization (SRUKF_GEQRF) failed");
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

  /* mean = Ysig * wm: one GEMV instead of a scalar accumulation loop */
  SRUKF_GEMV(SRUKF_CBLAS_LAYOUT, CblasNoTrans, (int)M, (int)n_sigma,
             (srukf_value)1.0, Ysig->data, (int)SRUKF_LEADING_DIM(Ysig), wm, 1,
             (srukf_value)0.0, mean->data, 1);

  return SRUKF_RETURN_OK;
}

/**
 * @brief Weighted mean via custom hook or Euclidean default
 *
 * Routes to the user's mean hook when one is registered (non-Euclidean
 * spaces), else to compute_weighted_mean(). Hook output flows into
 * deviations and covariance; non-finite values are caught by the
 * commit-time validity check.
 *
 * @param Ysig Sigma points (dim x (2N+1))
 * @param wm Mean weights (2N+1 elements)
 * @param fn Custom mean hook, or NULL for the default
 * @param ctx Context for the hook
 * @param mean Output mean (dim x 1)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return apply_weighted_mean(const srukf_mat *Ysig,
                                        const srukf_value *wm, srukf_mean_fn fn,
                                        void *ctx, srukf_mat *mean) {
  if (!fn)
    return compute_weighted_mean(Ysig, wm, mean);
  if (!Ysig || !wm || !mean)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (mean->n_rows != Ysig->n_rows || mean->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;
  fn(Ysig, wm, mean, ctx);
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
 * @param Xdev Scratch for weighted state deviations (N x (2N+1)),
 *             contents destroyed
 * @param Ydev Scratch for measurement deviations (M x (2N+1)),
 *             contents destroyed
 * @param x_res State-space residual hook, or NULL for Euclidean
 * @param x_ctx Context for x_res
 * @param y_res Measurement-space residual hook, or NULL for Euclidean
 * @param y_ctx Context for y_res
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return compute_cross_covariance(
    const srukf_mat *Xsig, const srukf_mat *Ysig, const srukf_mat *x_mean,
    const srukf_mat *y_mean, const srukf_value *weights, srukf_mat *Pxz,
    srukf_mat *Xdev, srukf_mat *Ydev, srukf_residual_fn x_res, void *x_ctx,
    srukf_residual_fn y_res, void *y_ctx) {
  if (!Xsig || !Ysig || !x_mean || !y_mean || !weights || !Pxz)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (!Xdev || !Ydev)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index N = Xsig->n_rows;
  srukf_index M = Ysig->n_rows;
  srukf_index K = Xsig->n_cols;

  if (N != Pxz->n_rows || M != Pxz->n_cols)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (K != Ysig->n_cols)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Xdev->n_rows != N || Xdev->n_cols < K)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Ydev->n_rows != M || Ydev->n_cols < K)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Fold the weights into the state deviations, keep the measurement
   * deviations plain, then Pxz = Xdev * Ydev' is a single GEMM. */
  for (srukf_index k = 0; k < K; ++k) {
    srukf_value wk = weights[k];
    if (x_res) {
      srukf_mat sig_col, dev_col;
      srukf_mat_column_view(&sig_col, Xsig, k);
      srukf_mat_column_view(&dev_col, Xdev, k);
      x_res(&sig_col, x_mean, &dev_col, x_ctx);
      for (srukf_index i = 0; i < N; ++i)
        SRUKF_ENTRY(Xdev, i, k) *= wk;
    } else {
      for (srukf_index i = 0; i < N; ++i)
        SRUKF_ENTRY(Xdev, i, k) =
            wk * (SRUKF_ENTRY(Xsig, i, k) - SRUKF_ENTRY(x_mean, i, 0));
    }
    if (y_res) {
      srukf_mat sig_col, dev_col;
      srukf_mat_column_view(&sig_col, Ysig, k);
      srukf_mat_column_view(&dev_col, Ydev, k);
      y_res(&sig_col, y_mean, &dev_col, y_ctx);
    } else {
      for (srukf_index j = 0; j < M; ++j)
        SRUKF_ENTRY(Ydev, j, k) =
            SRUKF_ENTRY(Ysig, j, k) - SRUKF_ENTRY(y_mean, j, 0);
    }
  }

  SRUKF_GEMM(SRUKF_CBLAS_LAYOUT, CblasNoTrans, CblasTrans, (int)N, (int)M,
             (int)K, (srukf_value)1.0, Xdev->data, (int)SRUKF_LEADING_DIM(Xdev),
             Ydev->data, (int)SRUKF_LEADING_DIM(Ydev), (srukf_value)0.0,
             Pxz->data, (int)SRUKF_LEADING_DIM(Pxz));

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_srukf_core */
