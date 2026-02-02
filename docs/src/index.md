# srukf

[![CI](https://github.com/disruptek/srukf/actions/workflows/ci.yml/badge.svg)](https://github.com/disruptek/srukf/actions/workflows/ci.yml)

C implementation of the Square-Root Unscented Kalman Filter.

## When to Use SR-UKF

**The scenario:** You're building a safety system for an electric unicycle.
The rider leans forward to accelerate, backward to brake, and side-to-side to
turn—constantly reorienting the onboard IMU through extreme angles. GPS
provides position fixes, but cuts out unpredictably under bridges, in urban
canyons, and near buildings. You need to fuse accelerometer, gyroscope, and
intermittent GPS into a reliable position and velocity estimate that runs
continuously for hours of riding.

**Why not a standard Kalman Filter?**
The classic Kalman filter assumes linear dynamics: `x_next = A*x + B*u`. But
transforming accelerometer readings from the wheel's wildly tilting reference
frame to earth coordinates requires rotation matrices full of sines and
cosines. A linear approximation falls apart when the rider leans 45° into a
hard turn.

**Why not an Extended Kalman Filter (EKF)?**
The EKF linearizes via Jacobians, which you must derive analytically for your
specific model. For 3D rotations with quaternions or Euler angles, this is
tedious and error-prone. Worse, when the rider's pose changes rapidly—hopping
off a curb, dodging a pedestrian—the linearization becomes a poor
approximation of reality, and the filter can diverge.

**Why not a standard Unscented Kalman Filter (UKF)?**
The UKF handles nonlinearity elegantly via sigma points—no Jacobians needed.
But it maintains the full covariance matrix P, and here's the problem: when
GPS cuts out, the filter runs on IMU alone and uncertainty grows. When GPS
returns, the correction step subtracts a large outer product from P. Over
thousands of these cycles, numerical roundoff accumulates. Eventually P loses
positive-definiteness, and the filter crashes—possibly mid-ride.

**Why SR-UKF?**
The Square-Root UKF maintains S where P = SSᵀ. This has two key benefits:

1. **Guaranteed stability**: We never explicitly subtract matrices. Instead,
   covariance updates use QR decomposition (for the predict step) and
   Cholesky rank-1 downdates (for corrections). These operations preserve
   the triangular structure of S by construction.

2. **Double the precision**: Errors in S translate to *squared* errors in P.
   If S has 15 digits of precision, P effectively has 30. This matters when
   uncertainty swings wildly between GPS-available and GPS-denied conditions.

**Real-world impact:** A standard UKF running at 100Hz might fail after a few
hours due to accumulated roundoff during repeated GPS dropout/reacquisition
cycles. The SR-UKF handles this indefinitely. For a safety-critical system
on a vehicle that could hit 30 mph, "might crash after a few hours" isn't
acceptable.

## Overview

State estimation library for nonlinear dynamic systems. Provides predict and
correct operations for sensor fusion applications.

## Performance

![SR-UKF Benchmark](../benchmark/benchmark.svg)

Typical performance on modern hardware (times in microseconds per operation).
The `correct` step is more expensive than `predict` because it involves
computing the Kalman gain and performing multiple Cholesky downdates.

![SR-UKF Memory](../benchmark/memory.svg)

Memory usage scales with state and measurement dimensions. Green bars show
total allocated bytes (filter + workspace); gray bars show actual RSS, which
is often lower due to Linux lazy allocation (pages aren't mapped until touched).
The allocated value is authoritative for capacity planning.

Workspace memory can be pre-allocated via `srukf_alloc_workspace()` to avoid
malloc during filtering, or left to allocate on first use. Pre-allocation is
recommended for real-time systems; lazy allocation suits memory-constrained
environments where filters may not all be used.

Run `make bench` / `make bench-memory` to benchmark on your system, or
`make bench-chart` / `make bench-memory-chart` to regenerate the charts.

## Requirements

- C99 compiler (GCC or Clang)
- OpenBLAS
- LAPACKE

On Debian/Ubuntu:

    apt install libopenblas-dev liblapacke-dev

## Building

    make        # build library and run tests
    make lib    # build libsrukf.so only
    make test   # run tests
    make bench  # run benchmarks
    make docs   # generate API documentation (requires Doxygen)

## Installation

    make install                  # installs to /usr/local
    make install PREFIX=/opt/sr  # installs to /opt/sr

### Build Options

- `SRUKF_SINGLE` - Compile with single-precision floats instead of doubles:

        CFLAGS=-DSRUKF_SINGLE make lib

## Usage

### Basic Example

```c
#include "srukf.h"

// Create filter: 3 states, 2 measurements
srukf *ukf = srukf_create(3, 2);

// Set noise covariances (as square-roots)
srukf_set_noise(ukf, Qsqrt, Rsqrt);

// Predict with process model f(x) -> x'
srukf_predict(ukf, process_model, NULL);

// Correct with measurement z and model h(x) -> z
srukf_correct(ukf, z, measurement_model, NULL);

// Access state estimate
srukf_value x0 = SRUKF_ENTRY(ukf->x, 0, 0);

srukf_free(ukf);
```

### Safe State and Covariance Access

For production code, use the safe accessor functions instead of direct field access:

```c
// Get current state estimate
srukf_mat *x_out = srukf_mat_alloc(N, 1, 1);
srukf_get_state(ukf, x_out);

// Get current sqrt-covariance
srukf_mat *S_out = srukf_mat_alloc(N, N, 1);
srukf_get_sqrt_cov(ukf, S_out);

// Set state and covariance
srukf_set_state(ukf, x_init);
srukf_set_sqrt_cov(ukf, S_init);

// Reset to initial conditions
srukf_reset(ukf, 1.0);  // init_std = 1.0

srukf_mat_free(x_out);
srukf_mat_free(S_out);
```

### Tuning UKF Parameters

The UKF spread is controlled by three parameters:

```c
// Set UKF scaling parameters
srukf_set_scale(ukf,
                1e-3,  // alpha: spread (typically 1e-3)
                2.0,   // beta: prior knowledge (2.0 for Gaussian)
                0.0);  // kappa: secondary scaling
```

- `alpha` controls the spread of sigma points (typically 1e-3 to 1)
- `beta` incorporates prior knowledge of the distribution (2.0 is optimal for Gaussian)
- `kappa` provides secondary scaling (typically 0 or 3 - N)

### Transactional Operations

For advanced use cases like ensemble filtering or multi-hypothesis tracking:

```c
// Perform predict on a copy without modifying filter state
srukf_mat *x_candidate = srukf_mat_alloc(N, 1, 1);
srukf_mat *S_candidate = srukf_mat_alloc(N, N, 1);

srukf_get_state(ukf, x_candidate);
srukf_get_sqrt_cov(ukf, S_candidate);

srukf_predict_to(ukf, x_candidate, S_candidate, process_model, NULL);
// Filter's internal state is unchanged

srukf_mat_free(x_candidate);
srukf_mat_free(S_candidate);
```

### Error Handling

Functions return `srukf_return` codes:

```c
srukf_return status = srukf_predict(ukf, model, NULL);

if (status == SRUKF_RETURN_OK) {
  // success
} else if (status == SRUKF_RETURN_PARAMETER_ERROR) {
  // invalid parameters
} else if (status == SRUKF_RETURN_MATH_ERROR) {
  // numerical/mathematical error
}
```

### Workspace Management

For performance-critical applications, pre-allocate workspace to avoid allocations during filtering:

```c
// Pre-allocate internal workspace
srukf_alloc_workspace(ukf);

// Now predict/correct use pre-allocated memory
srukf_predict(ukf, model, NULL);
srukf_correct(ukf, z, h, NULL);

// Free workspace when done
srukf_free_workspace(ukf);
```

## API Reference

### Types and Enumerations

```c
typedef double srukf_value;      // Scalar type (float if SRUKF_SINGLE)
typedef size_t srukf_index;      // Index/size type

// Return codes
typedef enum {
  SRUKF_RETURN_OK = 0,           // Success
  SRUKF_RETURN_PARAMETER_ERROR,  // Invalid parameter
  SRUKF_RETURN_MATH_ERROR        // Numerical error
} srukf_return;

// Matrix type flags (bitwise)
typedef enum {
  SRUKF_TYPE_COL_MAJOR = 0x01,   // Column-major storage
  SRUKF_TYPE_NO_DATA = 0x02,     // Descriptor without data
  SRUKF_TYPE_VECTOR = 0x04,      // Single-column vector
  SRUKF_TYPE_SQUARE = 0x08       // Square matrix
} srukf_mat_type;

// Matrix structure
typedef struct {
  srukf_index n_cols, n_rows;    // Columns, rows
  srukf_index inc_row, inc_col;  // Strides
  srukf_value *data;             // Data pointer
  srukf_mat_type type;           // Type flags
} srukf_mat;

// Filter structure
typedef struct {
  srukf_mat *x;                  // State estimate (N x 1)
  srukf_mat *S;                  // Sqrt covariance (N x N)
  srukf_mat *Qsqrt;              // Process noise sqrt-cov
  srukf_mat *Rsqrt;              // Measurement noise sqrt-cov
  srukf_value alpha, beta, kappa; // UKF parameters
  srukf_value lambda;            // Computed scaling
  srukf_value *wm, *wc;          // Weights (2N+1 each)
  srukf_workspace *ws;           // Workspace (internal)
} srukf;
```

### Matrix Operations

```c
srukf_mat *srukf_mat_alloc(srukf_index rows, srukf_index cols, int alloc_data);
void srukf_mat_free(srukf_mat *mat);

// Convenience macros
#define SRUKF_MAT_ALLOC(rows, cols)         srukf_mat_alloc((rows), (cols), 1)
#define SRUKF_MAT_ALLOC_NO_DATA(rows, cols) srukf_mat_alloc((rows), (cols), 0)

// Element access (column-major)
#define SRUKF_ENTRY(A, i, j) ((A)->data[(i)*(A)->inc_row + (j)*(A)->inc_col])
```

### Filter Lifecycle

```c
// Create filter
srukf *srukf_create(int N, int M);
srukf *srukf_create_from_noise(const srukf_mat *Qsqrt, const srukf_mat *Rsqrt);

// Initialize/configure
srukf_return srukf_set_noise(srukf *ukf, const srukf_mat *Qsqrt, const srukf_mat *Rsqrt);
srukf_return srukf_set_scale(srukf *ukf, srukf_value alpha, srukf_value beta, srukf_value kappa);

// Cleanup
void srukf_free(srukf *ukf);
```

### State Access

```c
// Dimension queries
srukf_index srukf_state_dim(const srukf *ukf);
srukf_index srukf_meas_dim(const srukf *ukf);

// Safe state/covariance access
srukf_return srukf_get_state(const srukf *ukf, srukf_mat *x_out);
srukf_return srukf_set_state(srukf *ukf, const srukf_mat *x_in);
srukf_return srukf_get_sqrt_cov(const srukf *ukf, srukf_mat *S_out);
srukf_return srukf_set_sqrt_cov(srukf *ukf, const srukf_mat *S_in);

// Reset to initial conditions
srukf_return srukf_reset(srukf *ukf, srukf_value init_std);
```

### Core Operations

```c
// Atomic predict/correct (updates filter state in-place)
srukf_return srukf_predict(srukf *ukf,
                           void (*f)(const srukf_mat *, srukf_mat *, void *),
                           void *user);
srukf_return srukf_correct(srukf *ukf, srukf_mat *z,
                           void (*h)(const srukf_mat *, srukf_mat *, void *),
                           void *user);

// Transactional predict/correct (operates on user-provided buffers)
srukf_return srukf_predict_to(srukf *ukf, srukf_mat *x, srukf_mat *S,
                              void (*f)(const srukf_mat *, srukf_mat *, void *),
                              void *user);
srukf_return srukf_correct_to(srukf *ukf, srukf_mat *x, srukf_mat *S, srukf_mat *z,
                              void (*h)(const srukf_mat *, srukf_mat *, void *),
                              void *user);
```

### Workspace Management

```c
srukf_return srukf_alloc_workspace(srukf *ukf);
void srukf_free_workspace(srukf *ukf);
```

### Diagnostics

```c
typedef void (*srukf_diag_fn)(const char *msg);
void srukf_set_diag_callback(srukf_diag_fn fn);
```

## Limitations

- **Not thread-safe**. The diagnostic callback is global; each filter instance
  should be used from a single thread.
- **Noise matrices must be square-roots**. Provide S where P = S*S', not P directly.
- **Column-major layout**. All matrices must be in column-major (Fortran) order.

## Documentation

Full API documentation with algorithm explanations is available at:
**https://disruptek.github.io/srukf/**

The docs include:
- Intuitive explanations of the Unscented Transform and sigma points
- Why square-root formulation provides numerical stability
- Detailed algorithm walkthroughs for predict and correct steps
- Tuning parameter guide (α, β, κ)
- Complete API reference with examples

Generate locally with `make docs` (requires [Doxygen](https://www.doxygen.nl/)).

## Acknowledgments

Matrix utilities derived from [LAH](https://github.com/maj0e/linear-algebra-helpers) by maj0e (MIT License).

## License

MIT License. See LICENSE file.
