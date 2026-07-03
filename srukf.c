/**
 * @file srukf.c
 * @brief Square-Root Unscented Kalman Filter -- single translation unit
 *
 * The implementation lives in src/, split into focused parts that are
 * included here into ONE translation unit. This keeps internal linkage
 * (static helpers shared across parts), lets tests reach internals via
 * `#include "srukf.c"`, and leaves both build systems with a single
 * source entry. Include order is load-bearing: every static function is
 * defined before its first use -- hence the clang-format guard, which
 * prevents the include block from being alphabetized.
 */

/* clang-format off */
#include "src/prelude.c"
#include "src/mat.c"
#include "src/diag.c"
#include "src/workspace.c"
#include "src/weights.c"
#include "src/lifecycle.c"
#include "src/accessors.c"
#include "src/sigma.c"
#include "src/core.c"
#include "src/predict.c"
#include "src/correct.c"
/* clang-format on */
