/* diag.c -- diagnostic message routing
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

/*============================================================================
 * @internal
 * @defgroup impl_diag Diagnostics
 * @brief Internal diagnostic support
 * @{
 *============================================================================*/

/** Global diagnostic callback (NULL = disabled) */
static srukf_diag_fn g_diag_callback = NULL;

void srukf_set_diag_callback(srukf_diag_fn fn) {
  g_diag_callback = fn;
}

/**
 * @brief Report a diagnostic message
 *
 * Routes to the instance's handler when one is set, else to the global
 * callback. Helpers that have no filter at hand pass NULL and reach the
 * global callback only.
 *
 * @param ukf Filter instance the message concerns (may be NULL)
 * @param msg Message to report
 */
static void diag_report(const srukf *ukf, const char *msg) {
  if (ukf && ukf->diag_fn) {
    ukf->diag_fn(msg, ukf->diag_ctx);
    return;
  }
  if (g_diag_callback)
    g_diag_callback(msg);
}

srukf_return srukf_set_diag(srukf *ukf, srukf_diag_handler fn, void *ctx) {
  if (!ukf)
    return SRUKF_RETURN_PARAMETER_ERROR;
  ukf->diag_fn = fn;
  ukf->diag_ctx = ctx;
  return SRUKF_RETURN_OK;
}

/** @} */ /* end impl_diag */
