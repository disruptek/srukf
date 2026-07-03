/* weights.c -- sigma-point weights and scaling parameters
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

/*============================================================================
 * @internal
 * @defgroup impl_weights Weight Computation
 * @brief UKF sigma point weights
 *
 * The UKF uses weighted averages to reconstruct mean and covariance from
 * sigma points. There are two sets of weights:
 *
 * - **Mean weights (wm):** Used to compute weighted mean
 * - **Covariance weights (wc):** Used to compute weighted covariance
 *
 * The weights sum to 1 for the mean, but wc[0] can differ from wm[0]
 * to better capture higher-order moments of the distribution.
 *
 * @{
 *============================================================================*/

/**
 * @brief Compute sigma point weights
 *
 * Given the current scaling parameters (alpha, beta, kappa, lambda),
 * computes the 2N+1 weights for mean and covariance calculations.
 *
 * **Weight formulas:**
 * @code
 * wm[0] = lambda / (N + lambda)
 * wm[i] = 1 / (2 * (N + lambda))  for i = 1..2N
 *
 * wc[0] = wm[0] + (1 - alpha^2 + beta)
 * wc[i] = wm[i]  for i = 1..2N
 * @endcode
 *
 * **Note on negative wc[0]:**
 * For small alpha, lambda approaches -N, making wm[0] large and negative.
 * Adding (1 - alpha^2 + beta) may not compensate enough, leaving wc[0] < 0.
 * This is handled correctly in the QR-based covariance computation.
 *
 * @param ukf Filter with alpha/beta/kappa/lambda set and weight arrays
 *            allocated (guaranteed by srukf_set_scale, the only caller)
 * @param n State dimension N
 */
static void srukf_compute_weights(srukf *ukf, const srukf_index n) {
  srukf_index n_sigma = 2 * n + 1;

  /* Common denominator: n + λ = α²(n+κ), validated > 0 by set_scale */
  const srukf_value denom = (srukf_value)n + ukf->lambda;

  /* Mean weights: wm[0] = λ / (n+λ), wm[i>0] = 1/(2(n+λ)) */
  for (srukf_index i = 0; i < n_sigma; ++i)
    ukf->wm[i] = 1.0 / (2.0 * denom);
  ukf->wm[0] = ukf->lambda / denom;

  /* Covariance weights: wc[0] = wm[0] + (1-α²+β), wc[i>0] = wm[i] */
  for (srukf_index i = 0; i < n_sigma; ++i)
    ukf->wc[i] = ukf->wm[i];
  ukf->wc[0] += (1.0 - ukf->alpha * ukf->alpha + ukf->beta);
}

/** @} */ /* end impl_weights */

srukf_return srukf_set_scale(srukf *ukf, srukf_value alpha, srukf_value beta,
                             srukf_value kappa) {
  if (!ukf || !ukf->x)
    return SRUKF_RETURN_PARAMETER_ERROR; /* filter or state not yet allocated */

  /* NaN survives ordering comparisons, so reject non-finite inputs
   * explicitly or they poison every weight downstream. */
  if (!isfinite(alpha) || !isfinite(beta) || !isfinite(kappa))
    return SRUKF_RETURN_PARAMETER_ERROR;

  if (alpha <= 0.0)
    return SRUKF_RETURN_PARAMETER_ERROR; /* α must be positive */

  srukf_index n = ukf->x->n_rows; /* state dimension */

  /* n + κ must be positive: γ = sqrt(α²(n+κ)) must be real, and the
   * weight denominator n + λ = α²(n+κ) must be positive. */
  if ((srukf_value)n + kappa <= 0.0)
    return SRUKF_RETURN_PARAMETER_ERROR;

  srukf_value lambda =
      alpha * alpha * ((srukf_value)n + kappa) - (srukf_value)n;

  /* Mathematically positive, but reject underflow below the working
   * precision: the weights are O(1/(n+λ)) and would be garbage. */
  if ((srukf_value)n + lambda < SRUKF_EPS) {
    diag_report(ukf, "set_scale: alpha^2 * (n + kappa) underflows precision");
    return SRUKF_RETURN_MATH_ERROR;
  }

  /* Ensure weight storage exists before committing any parameter, so a
   * failed call never leaves parameters and weights out of sync. */
  srukf_index n_sigma = 2 * n + 1;
  if (!ukf->wm) {
    srukf_return rc = alloc_vector(&ukf->wm, n_sigma);
    if (rc != SRUKF_RETURN_OK)
      return rc;
  }
  if (!ukf->wc) {
    srukf_return rc = alloc_vector(&ukf->wc, n_sigma);
    if (rc != SRUKF_RETURN_OK)
      return rc;
  }

  ukf->alpha = alpha;
  ukf->beta = beta;
  ukf->kappa = kappa;
  ukf->lambda = lambda;
  srukf_compute_weights(ukf, n);
  return SRUKF_RETURN_OK;
}
