/* correct.c -- the correct (measurement update) step
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

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
 * @param x_out Output corrected state (N x 1, must not alias x_in)
 * @param S_out Output corrected sqrt-covariance (N x N, must not alias S_in)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return srukf_correct_core(const srukf *ukf, const srukf_mat *x_in,
                                       const srukf_mat *S_in,
                                       const srukf_mat *z, srukf_model_fn h,
                                       void *user, srukf_mat *x_out,
                                       srukf_mat *S_out) {
  srukf_return ret = SRUKF_RETURN_OK;
  if (!ukf || !h || !z || !x_in || !S_in || !x_out || !S_out)
    return SRUKF_RETURN_PARAMETER_ERROR;
  /* Rsqrt->data is NULL until srukf_set_noise() supplies real values */
  if (!ukf->Rsqrt || !ukf->Rsqrt->data || !ukf->wm || !ukf->wc || !ukf->ws)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Whatever innovation the workspace holds is about to be overwritten;
   * it becomes readable again only if this step completes. */
  ukf->ws->correct_valid = false;

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
  ret = generate_sigma_points_from(ukf, x_in, S_in, Xsig);
  if (ret != SRUKF_RETURN_OK)
    return ret;
  ret = propagate_sigma_points(Xsig, Ysig, h, user);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* --- Validate callback output ----------------------------------- */
  if (!is_numeric_valid(Ysig)) {
    diag_report(ukf, "correct: callback h produced NaN or Inf");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* 2. Copy prior state mean for cross-covariance computation ------ */
  for (srukf_index i = 0; i < N; ++i)
    SRUKF_ENTRY(x_mean, i, 0) = SRUKF_ENTRY(x_in, i, 0);

  /* 3. Compute weighted mean for measurement (hook if registered) --- */
  ret = apply_weighted_mean(Ysig, ukf->wm, ukf->meas_mean_fn, ukf->meas_ops_ctx,
                            y_mean);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* 4. Compute weighted deviations for measurement ------------------- */
  ret = compute_weighted_deviations(
      Ysig, y_mean, ukf->wc, ukf->meas_residual_fn, ukf->meas_ops_ctx, Dev_M);
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

  ret = srukf_sqrt_from_deviations_ex(
      Dev_M, ukf->Rsqrt, Syy, qr_work, ws->tau_M, ws->downdate_work,
      ws->lapack_work, ws->lwork, wc0_negative, dev0_M_buf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  /* --- Reject a singular Syy ---------------------------------------- */
  /* The Kalman gain divides by Syy twice; a diagonal entry that vanishes
   * RELATIVE to the largest one means Pyy is numerically singular and
   * the gain is undefined. The test must be relative, not absolute: the
   * gain is scale-invariant in Syy, so a filter operating at 1e-15
   * scales with a well-conditioned Syy is perfectly healthy. */
  srukf_value Syy_max = 0.0;
  for (srukf_index i = 0; i < M; ++i) {
    srukf_value d = SRUKF_FABS(SRUKF_ENTRY(Syy, i, i));
    if (d > Syy_max)
      Syy_max = d;
  }
  bool Syy_singular = (Syy_max == 0.0);
  for (srukf_index i = 0; i < M && !Syy_singular; ++i)
    if (SRUKF_FABS(SRUKF_ENTRY(Syy, i, i)) < SRUKF_EPS * Syy_max)
      Syy_singular = true;
  if (Syy_singular) {
    diag_report(ukf, "correct: innovation covariance Syy is singular "
                     "(measurement model has no spread and no noise?)");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* 6. Cross‑covariance between state & measurement σ‑points -------- */
  /* Dev_N is predict-only and Dev_M is done serving the Syy QR above,
   * so both are free to serve as GEMM scratch here. */
  ret = compute_cross_covariance(Xsig, Ysig, x_mean, y_mean, ukf->wc, Pxz,
                                 ws->Dev_N, Dev_M, ukf->state_residual_fn,
                                 ukf->state_ops_ctx, ukf->meas_residual_fn,
                                 ukf->meas_ops_ctx);
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

  /* 8. Innovation (measurement-space residual if registered) ----------- */
  if (ukf->meas_residual_fn) {
    ukf->meas_residual_fn(z, y_mean, innov, ukf->meas_ops_ctx);
  } else {
    for (srukf_index i = 0; i < M; ++i)
      SRUKF_ENTRY(innov, i, 0) =
          SRUKF_ENTRY(z, i, 0) - SRUKF_ENTRY(y_mean, i, 0);
  }

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
      diag_report(ukf, "correct: Cholesky downdate failed, matrix not SPD");
      return ret;
    }
  }

  /* 11. Write state output --------------------------------------------- */
  memcpy(x_out->data, x_new->data, N * sizeof(srukf_value));

  /* Belt and braces: no numerical escape (BLAS overflow, downdate
   * pathology) may be reported as success. */
  if (!is_numeric_valid(x_out) || !is_numeric_valid(S_out)) {
    diag_report(ukf, "correct: non-finite result rejected");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* The measurement was incorporated: innov and Syy in the workspace
   * now describe this step and may be read via the accessors. */
  ukf->ws->correct_valid = true;

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_correct */

srukf_return srukf_correct_to(srukf *ukf, srukf_mat *x, srukf_mat *S,
                              const srukf_mat *z, srukf_model_fn h,
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

  /* Stage in the workspace, commit only on success: the caller's
   * buffers survive even a failure deep in the downdate loop. */
  srukf_mat *x_stage = ukf->ws->x_stage;
  srukf_mat *S_stage = ukf->ws->S_new;
  ret = srukf_correct_core(ukf, x, S, z, h, user, x_stage, S_stage);
  if (ret == SRUKF_RETURN_OK) {
    memcpy(x->data, x_stage->data, N * sizeof(srukf_value));
    memcpy(S->data, S_stage->data, N * N * sizeof(srukf_value));
  }
  return ret;
}

srukf_return srukf_correct(srukf *ukf, const srukf_mat *z, srukf_model_fn h,
                           void *user) {
  if (!ukf || !ukf->x || !ukf->S)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Ensure workspace is allocated */
  srukf_return ret = ensure_workspace(ukf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  srukf_index N = ukf->x->n_rows;

  /* Stage in the workspace, commit only on success. x_stage/S_new are
   * dedicated staging buffers the core never touches internally. */
  srukf_mat *x_stage = ukf->ws->x_stage;
  srukf_mat *S_stage = ukf->ws->S_new;
  ret = srukf_correct_core(ukf, ukf->x, ukf->S, z, h, user, x_stage, S_stage);
  if (ret == SRUKF_RETURN_OK) {
    memcpy(ukf->x->data, x_stage->data, N * sizeof(srukf_value));
    memcpy(ukf->S->data, S_stage->data, N * N * sizeof(srukf_value));
  }
  return ret;
}
