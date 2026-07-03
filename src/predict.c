/* predict.c -- the predict step
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

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
 * Outputs must not alias inputs; the public wrappers guarantee this by
 * staging results in dedicated workspace buffers and committing on
 * success.
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
 * @param x_out Output predicted state (N x 1, must not alias x_in)
 * @param S_out Output predicted sqrt-covariance (N x N, must not alias S_in)
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return srukf_predict_core(const srukf *ukf, const srukf_mat *x_in,
                                       const srukf_mat *S_in, srukf_model_fn f,
                                       void *user, srukf_mat *x_out,
                                       srukf_mat *S_out) {
  srukf_return ret = SRUKF_RETURN_OK;
  if (!ukf || !f || !x_in || !S_in || !x_out || !S_out)
    return SRUKF_RETURN_PARAMETER_ERROR;
  /* Qsqrt->data is NULL until srukf_set_noise() supplies real values
   * (srukf_create allocates the struct without a data buffer) */
  if (!ukf->Qsqrt || !ukf->Qsqrt->data || !ukf->wm || !ukf->wc || !ukf->ws)
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
  ret = generate_sigma_points_from(ukf, x_in, S_in, Xsig);
  if (ret != SRUKF_RETURN_OK) {
    diag_report(ukf, "predict: sigma point generation failed");
    return ret;
  }

  ret = propagate_sigma_points(Xsig, Ysig, f, user);
  if (ret != SRUKF_RETURN_OK) {
    diag_report(ukf, "predict: sigma point propagation failed");
    return ret;
  }

  /* --- Validate callback output ----------------------------------- */
  if (!is_numeric_valid(Ysig)) {
    diag_report(ukf, "predict: callback f produced NaN or Inf");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* --- Compute weighted mean (state-space hook if registered) ------ */
  ret = apply_weighted_mean(Ysig, ukf->wm, ukf->state_mean_fn,
                            ukf->state_ops_ctx, x_mean);
  if (ret != SRUKF_RETURN_OK) {
    diag_report(ukf, "predict: weighted mean failed");
    return ret;
  }

  /* --- Compute weighted deviations --------------------------------- */
  ret = compute_weighted_deviations(
      Ysig, x_mean, ukf->wc, ukf->state_residual_fn, ukf->state_ops_ctx, Dev);
  if (ret != SRUKF_RETURN_OK) {
    diag_report(ukf, "predict: compute_weighted_deviations failed");
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

  ret = srukf_sqrt_from_deviations_ex(
      Dev, ukf->Qsqrt, S_out, qr_work, ws->tau_N, ws->downdate_work,
      ws->lapack_work, ws->lwork, wc0_negative, dev0);
  if (ret != SRUKF_RETURN_OK) {
    diag_report(ukf, "predict: sqrt_from_deviations (QR/downdate) failed");
    return ret;
  }

  /* --- Write mean → x_out ---------------------------------------- */
  memcpy(x_out->data, x_mean->data, N * sizeof(srukf_value));

  /* Belt and braces: no numerical escape (BLAS overflow, downdate
   * pathology) may be reported as success -- the wrappers commit the
   * staged buffers on OK, and the contract is that committed state is
   * always usable. */
  if (!is_numeric_valid(x_out) || !is_numeric_valid(S_out)) {
    diag_report(ukf, "predict: non-finite result rejected");
    return SRUKF_RETURN_MATH_ERROR;
  }

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_predict */

srukf_return srukf_predict_to(srukf *ukf, srukf_mat *x, srukf_mat *S,
                              srukf_model_fn f, void *user) {
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

  /* Stage in the workspace, commit only on success: the caller's
   * buffers are never left half-updated by a mid-step failure. */
  srukf_mat *x_stage = ukf->ws->x_stage;
  srukf_mat *S_stage = ukf->ws->S_tmp;
  ret = srukf_predict_core(ukf, x, S, f, user, x_stage, S_stage);
  if (ret == SRUKF_RETURN_OK) {
    memcpy(x->data, x_stage->data, N * sizeof(srukf_value));
    memcpy(S->data, S_stage->data, N * N * sizeof(srukf_value));
  }
  return ret;
}

srukf_return srukf_predict(srukf *ukf, srukf_model_fn f, void *user) {
  if (!ukf || !ukf->x || !ukf->S)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* Ensure workspace is allocated */
  srukf_return ret = ensure_workspace(ukf);
  if (ret != SRUKF_RETURN_OK)
    return ret;

  srukf_index N = ukf->x->n_rows;

  /* Stage in the workspace, commit only on success. x_stage/S_tmp are
   * dedicated staging buffers the core never touches internally. */
  srukf_mat *x_stage = ukf->ws->x_stage;
  srukf_mat *S_stage = ukf->ws->S_tmp;
  ret = srukf_predict_core(ukf, ukf->x, ukf->S, f, user, x_stage, S_stage);
  if (ret == SRUKF_RETURN_OK) {
    memcpy(ukf->x->data, x_stage->data, N * sizeof(srukf_value));
    memcpy(ukf->S->data, S_stage->data, N * N * sizeof(srukf_value));
  }
  return ret;
}
