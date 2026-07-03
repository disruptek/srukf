/* mat.c -- matrix allocation and small numeric helpers
 * Part of the srukf single translation unit; included by srukf.c.
 * Not compiled standalone. */

/*============================================================================
 * @internal
 * @defgroup impl_matrix Matrix Allocation
 * @brief Internal matrix utilities (derived from LAH by maj0e, MIT License)
 * @{
 *============================================================================*/

srukf_mat *srukf_mat_alloc(srukf_index rows, srukf_index cols, int alloc_data) {
  /* Zero dimensions are unrepresentable, and rows * cols must not wrap:
   * a wrapped product would produce a descriptor claiming billions of
   * rows over a (nearly) empty buffer. */
  if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols)
    return NULL;

  srukf_mat *mat = (srukf_mat *)calloc(1, sizeof(srukf_mat));
  if (!mat)
    return NULL;

  mat->n_rows = rows;
  mat->n_cols = cols;
  mat->inc_row = 1; /* column-major */
  mat->inc_col = rows;
  mat->type = SRUKF_TYPE_COL_MAJOR;

  if (cols == 1)
    SRUKF_SET_TYPE(mat, SRUKF_TYPE_VECTOR);
  if (rows == cols)
    SRUKF_SET_TYPE(mat, SRUKF_TYPE_SQUARE);

  if (alloc_data) {
    mat->data = (srukf_value *)calloc(rows * cols, sizeof(srukf_value));
    if (!mat->data) {
      free(mat);
      return NULL;
    }
  } else {
    mat->data = NULL;
    SRUKF_SET_TYPE(mat, SRUKF_TYPE_NO_DATA);
  }

  return mat;
}

void srukf_mat_free(srukf_mat *mat) {
  if (!mat)
    return;
  if (!SRUKF_IS_TYPE(mat, SRUKF_TYPE_NO_DATA))
    free(mat->data);
  free(mat);
}

/** @} */ /* end impl_matrix */

/*============================================================================
 * @internal
 * @defgroup impl_helpers Helper Functions
 * @brief Various utility functions
 * @{
 *============================================================================*/

/**
 * @brief Check if matrix contains any NaN or Inf values
 *
 * Used to validate callback outputs. If a process or measurement model
 * produces non-finite values, we detect it here and return an error
 * rather than letting garbage propagate through the filter.
 *
 * @param M Matrix to check
 * @return true if all values are finite, false if any NaN/Inf found
 */
static bool is_numeric_valid(const srukf_mat *M) {
  if (!M || !M->data)
    return false;
  for (srukf_index j = 0; j < M->n_cols; ++j)
    for (srukf_index i = 0; i < M->n_rows; ++i) {
      srukf_value v = SRUKF_ENTRY(M, i, j);
      if (isnan(v) || isinf(v))
        return false;
    }
  return true;
}

/**
 * @brief Allocate a vector of doubles
 * @param vec Output pointer
 * @param len Number of elements
 * @return SRUKF_RETURN_OK on success
 */
static srukf_return alloc_vector(srukf_value **vec, srukf_index len) {
  *vec = (srukf_value *)calloc(len, sizeof(srukf_value));
  return (*vec) ? SRUKF_RETURN_OK : SRUKF_RETURN_MEMORY_ERROR;
}

/** @} */ /* end impl_helpers */
