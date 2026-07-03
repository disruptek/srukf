/* --------------------------------------------------------------------
 * 49_zeroalloc.c - Verifies the real-time guarantee: once the workspace
 * is allocated, predict/correct perform ZERO heap allocations.
 *
 * The historical violation was LAPACKE's convenience wrapper
 * (LAPACKE_dgeqrf), which mallocs its work buffer on every call; the
 * library must use the _work variant with a workspace-owned buffer.
 *
 * Mechanism: this binary defines malloc(), so every malloc in the
 * process - including LAPACKE's - resolves here via symbol
 * interposition, and we forward to glibc's __libc_malloc. Requires
 * glibc (fine: CI and dev boxes); on other libcs the test is skipped
 * at compile time.
 *
 * This test includes srukf.c directly.
 * -------------------------------------------------------------------- */

#include <stdio.h>

/* Interposing malloc conflicts with sanitizer allocators; the test is
 * meaningless there anyway (ASan's allocator is not the hot path being
 * certified). Detect both GCC and Clang spellings. */
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define ZEROALLOC_SKIP 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define ZEROALLOC_SKIP 1
#endif
#endif

#if defined(__GLIBC__) && !defined(ZEROALLOC_SKIP)

#include <stddef.h>

extern void *__libc_malloc(size_t size);

static volatile int counting = 0;
static volatile long alloc_count = 0;

void *malloc(size_t size) {
  if (counting)
    ++alloc_count;
  return __libc_malloc(size);
}

#include "srukf.c"
#include "tests/test_helpers.h"

/* mildly nonlinear process model: rotation plus damping */
static void process(const srukf_mat *x, srukf_mat *x_out, void *user) {
  (void)user;
  srukf_index n = x->n_rows;
  for (srukf_index i = 0; i < n; ++i) {
    srukf_value a = SRUKF_ENTRY(x, i, 0);
    srukf_value b = SRUKF_ENTRY(x, (i + 1) % n, 0);
    SRUKF_ENTRY(x_out, i, 0) = 0.99 * a + 0.01 * b;
  }
}

/* observe the first M state components */
static void meas(const srukf_mat *x, srukf_mat *z, void *user) {
  (void)user;
  for (srukf_index i = 0; i < z->n_rows; ++i)
    SRUKF_ENTRY(z, i, 0) = SRUKF_ENTRY(x, i, 0);
}

int main(void) {
  enum { N = 6, M = 3, WARMUP = 5, STEPS = 100 };

  printf("Running zero-allocation tests...\n");

  srukf_mat *Q = SRUKF_MAT_ALLOC(N, N);
  srukf_mat *R = SRUKF_MAT_ALLOC(M, M);
  srukf_mat *z = SRUKF_MAT_ALLOC(M, 1);
  assert(Q && R && z);
  for (int i = 0; i < N; ++i)
    SRUKF_ENTRY(Q, i, i) = 0.1;
  for (int i = 0; i < M; ++i)
    SRUKF_ENTRY(R, i, i) = 0.2;
  SRUKF_ENTRY(z, 0, 0) = 0.3;

  srukf *ukf = srukf_create_from_noise(Q, R);
  assert(ukf);
  assert(srukf_reset(ukf, 1.0) == SRUKF_RETURN_OK);
  assert(srukf_alloc_workspace(ukf) == SRUKF_RETURN_OK);

  /* Warm up: first calls may trigger one-time allocations inside the
   * BLAS (thread pools, buffers) and stdio. Those are permitted. */
  for (int i = 0; i < WARMUP; ++i) {
    assert(srukf_predict(ukf, process, NULL) == SRUKF_RETURN_OK);
    assert(srukf_correct(ukf, z, meas, NULL) == SRUKF_RETURN_OK);
  }
  printf("  warmed up; counting allocations over %d steps\n", STEPS);

  counting = 1;
  for (int i = 0; i < STEPS; ++i) {
    assert(srukf_predict(ukf, process, NULL) == SRUKF_RETURN_OK);
    assert(srukf_correct(ukf, z, meas, NULL) == SRUKF_RETURN_OK);
  }
  counting = 0;

  if (alloc_count != 0) {
    printf("  FAILED: %ld heap allocations in %d predict/correct steps\n",
           alloc_count, STEPS);
    return 1;
  }
  printf("  test_zero_alloc      OK (0 allocations in %d steps)\n", STEPS);

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_mat_free(z);
  srukf_free(ukf);

  printf("zero-allocation tests passed.\n");
  return 0;
}

#else /* !__GLIBC__ or sanitizer build */

int main(void) {
  printf("Running zero-allocation tests...\n");
  printf("  test_zero_alloc      SKIPPED (requires glibc, no sanitizer)\n");
  printf("zero-allocation tests passed.\n");
  return 0;
}

#endif
