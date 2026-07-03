/* sigma.c -- sigma-point generation and propagation
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

/*============================================================================
 * @internal
 * @defgroup impl_sigma Sigma Point Generation
 * @brief Creating the 2N+1 sigma points that capture mean and covariance
 *
 * Sigma points are the core insight of the Unscented Transform. Instead of
 * linearizing a nonlinear function (like EKF does), we:
 *
 * 1. Choose 2N+1 carefully placed sample points around the mean
 * 2. Propagate each through the nonlinear function exactly
 * 3. Reconstruct statistics from the transformed samples
 *
 * The sigma points form a symmetric pattern: the mean, plus N points
 * offset by +gamma * column_of_S, plus N points offset by -gamma * column_of_S.
 * This pattern exactly matches the first two moments (mean and covariance)
 * of the original distribution.
 *
 * @{
 *============================================================================*/

/**
 * @brief Generate sigma points from state and sqrt-covariance
 *
 * Creates 2N+1 sigma points arranged symmetrically around the mean:
 *
 * @code
 * chi[0]   = x                        (the mean)
 * chi[i]   = x + gamma * S(:,i)       for i = 1..N
 * chi[i+N] = x - gamma * S(:,i)       for i = 1..N
 * @endcode
 *
 * where gamma = sqrt(N + lambda) is the spread factor.
 *
 * **Geometric interpretation:**
 * If P = S*S' is the covariance, then the sigma points lie on an ellipsoid
 * centered at x, scaled by gamma. The columns of S define the principal
 * axes of this ellipsoid.
 *
 * **Why use S instead of P?**
 * We need sqrt(P) to generate sigma points. In standard UKF, we'd compute
 * chol(P) every time. In SR-UKF, we already have S, saving a Cholesky
 * decomposition per step.
 *
 * @param ukf Filter (for lambda and diagnostic routing)
 * @param x State vector (N x 1)
 * @param S Sqrt-covariance (N x N, lower triangular)
 * @param Xsig Output: sigma points (N x (2N+1))
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return generate_sigma_points_from(const srukf *ukf,
                                               const srukf_mat *x,
                                               const srukf_mat *S,
                                               srukf_mat *Xsig) {
  if (!ukf || !x || !S || !Xsig)
    return SRUKF_RETURN_PARAMETER_ERROR;
  srukf_value lambda = ukf->lambda;

  srukf_index n = x->n_rows;       /* state dimension N */
  srukf_index n_sigma = 2 * n + 1; /* number of sigma points */
  if (Xsig->n_rows != n || Xsig->n_cols != n_sigma)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (S->n_rows != n || S->n_cols != n)
    return SRUKF_RETURN_PARAMETER_ERROR;

  /* scaling factor γ = sqrt( N + λ ); test positivity BEFORE the sqrt:
   * a NaN gamma would sail through a `gamma <= 0` comparison */
  srukf_value t = (srukf_value)n + lambda;
  if (!(t > 0.0)) {
    diag_report(ukf, "generate_sigma_points: N + lambda <= 0 or NaN");
    return SRUKF_RETURN_MATH_ERROR;
  }
  srukf_value gamma = SRUKF_SQRT(t);

  /* 1st column = mean state */
  for (srukf_index i = 0; i < n; ++i)
    SRUKF_ENTRY(Xsig, i, 0) = SRUKF_ENTRY(x, i, 0);

  /* Remaining columns */
  for (srukf_index k = 0; k < n; ++k) {
    for (srukf_index i = 0; i < n; ++i) {
      /* +γ * S(:,k)  */
      SRUKF_ENTRY(Xsig, i, k + 1) =
          SRUKF_ENTRY(x, i, 0) + gamma * SRUKF_ENTRY(S, i, k);
      /* -γ * S(:,k)  */
      SRUKF_ENTRY(Xsig, i, k + 1 + n) =
          SRUKF_ENTRY(x, i, 0) - gamma * SRUKF_ENTRY(S, i, k);
    }
  }
  return SRUKF_RETURN_OK;
}

/**
 * @brief Create a column view into a matrix (zero-copy)
 *
 * Sets up V to reference column k of M without copying data.
 * This is used extensively when propagating sigma points, where
 * we need to pass individual columns to the process/measurement model.
 *
 * @param V Output view (caller provides storage for the srukf_mat struct)
 * @param M Source matrix
 * @param k Column index
 */
static inline void srukf_mat_column_view(srukf_mat *V, const srukf_mat *M,
                                         srukf_index k) {
  V->n_rows = M->n_rows;
  V->n_cols = 1;
  V->inc_row = 1;         /* column vector */
  V->inc_col = M->n_rows; /* distance to next column in M */
  V->data = M->data + k * M->inc_col;
  V->type = 0;
  SRUKF_SET_TYPE(V, SRUKF_TYPE_COL_MAJOR);
}

/**
 * @brief Propagate sigma points through a function
 *
 * Applies a function (process model or measurement model) to each
 * sigma point individually. This is where the "unscented" magic happens:
 * we evaluate the nonlinear function exactly, rather than linearizing it.
 *
 * @param Xsig Input sigma points (N x (2N+1) for state, M x (2N+1) for meas)
 * @param Ysig Output propagated points (same size as Xsig or different for h)
 * @param func Function to apply: func(x_in, x_out, user)
 * @param user User data passed to func
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return propagate_sigma_points(const srukf_mat *Xsig,
                                           srukf_mat *Ysig, srukf_model_fn func,
                                           void *user) {
  /* Basic sanity checks */
  if (!Xsig || !func || !Ysig)
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (Ysig->data == NULL)
    return SRUKF_RETURN_PARAMETER_ERROR;
  if (Xsig->n_cols != Ysig->n_cols)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_index n_sigma = Xsig->n_cols; /* number of sigma points */

  /* Temporary matrix descriptors for a single column vector */
  srukf_mat col_in, col_out;

  /* Loop over all sigma points */
  for (srukf_index k = 0; k < n_sigma; ++k) {
    /* Point to the k‑th column of the input and output matrices */
    srukf_mat_column_view(&col_in, Xsig, k);
    srukf_mat_column_view(&col_out, Ysig, k);
    /* Call the user supplied model */
    func(&col_in, &col_out, user);
  }

  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_sigma */
