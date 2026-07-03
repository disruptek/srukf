/* accessors.c -- state, parameter, hook, and innovation accessors
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

/*-------------------- Custom space hooks ------------------------------*/

srukf_return srukf_set_state_ops(srukf *ukf, srukf_mean_fn mean_fn,
                                 srukf_residual_fn residual_fn, void *user) {
  if (!ukf)
    return SRUKF_RETURN_PARAMETER_ERROR;
  ukf->state_mean_fn = mean_fn;
  ukf->state_residual_fn = residual_fn;
  ukf->state_ops_ctx = user;
  return SRUKF_RETURN_OK;
}

srukf_return srukf_set_meas_ops(srukf *ukf, srukf_mean_fn mean_fn,
                                srukf_residual_fn residual_fn, void *user) {
  if (!ukf)
    return SRUKF_RETURN_PARAMETER_ERROR;
  ukf->meas_mean_fn = mean_fn;
  ukf->meas_residual_fn = residual_fn;
  ukf->meas_ops_ctx = user;
  return SRUKF_RETURN_OK;
}

/*-------------------- Parameter accessors ----------------------------*/
srukf_return srukf_get_scale(const srukf *ukf, srukf_value *alpha_out,
                             srukf_value *beta_out, srukf_value *kappa_out) {
  if (!ukf)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (alpha_out)
    *alpha_out = ukf->alpha;
  if (beta_out)
    *beta_out = ukf->beta;
  if (kappa_out)
    *kappa_out = ukf->kappa;
  return SRUKF_RETURN_OK;
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

/*-------------------- Innovation accessors ---------------------------*/

/* The innovation and Syy live in the workspace; correct_valid gates
 * access so callers can never read leftovers from an unrelated or
 * failed step. */

srukf_return srukf_get_innovation(const srukf *ukf, srukf_mat *innov_out) {
  if (!ukf || !innov_out || !ukf->ws || !ukf->ws->correct_valid)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index M = ukf->ws->M;
  if (innov_out->n_rows != M || innov_out->n_cols != 1)
    return SRUKF_RETURN_PARAMETER_ERROR;

  memcpy(innov_out->data, ukf->ws->innov->data, M * sizeof(srukf_value));
  return SRUKF_RETURN_OK;
}

srukf_return srukf_get_innovation_sqrt_cov(const srukf *ukf,
                                           srukf_mat *Syy_out) {
  if (!ukf || !Syy_out || !ukf->ws || !ukf->ws->correct_valid)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index M = ukf->ws->M;
  if (Syy_out->n_rows != M || Syy_out->n_cols != M)
    return SRUKF_RETURN_PARAMETER_ERROR;

  memcpy(Syy_out->data, ukf->ws->Syy->data, M * M * sizeof(srukf_value));
  return SRUKF_RETURN_OK;
}

srukf_return srukf_get_nis(const srukf *ukf, srukf_value *nis_out) {
  if (!ukf || !nis_out || !ukf->ws || !ukf->ws->correct_valid)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_workspace *ws = ukf->ws;
  srukf_index M = ws->M;

  /* NIS = innov' * (Syy Syy')^{-1} * innov = ||Syy^{-1} innov||^2,
   * one forward substitution against the lower-triangular Syy. */
  memcpy(ws->downdate_work, ws->innov->data, M * sizeof(srukf_value));
  SRUKF_TRSV(SRUKF_CBLAS_LAYOUT, CblasLower, CblasNoTrans, CblasNonUnit, (int)M,
             ws->Syy->data, (int)SRUKF_LEADING_DIM(ws->Syy), ws->downdate_work,
             1);

  srukf_value nis = 0.0;
  for (srukf_index i = 0; i < M; ++i)
    nis += ws->downdate_work[i] * ws->downdate_work[i];
  *nis_out = nis;
  return SRUKF_RETURN_OK;
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

  /* NaN survives the <= comparison; reject non-finite explicitly */
  if (!isfinite(init_std) || init_std <= 0.0)
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
