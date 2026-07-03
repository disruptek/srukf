# Contributing to SR-UKF

Thank you for considering contributing! This guide covers everything you need to get started.

## Development Setup

### Prerequisites

- C99 compiler (GCC or Clang)
- OpenBLAS and LAPACKE
- Python 3.7+ with NumPy (for Python bindings)
- clang-format (for code formatting)
- Doxygen (optional, for documentation)

**Ubuntu / Debian:**

```bash
sudo apt-get install build-essential libopenblas-dev liblapacke-dev \
    clang-format python3 python3-numpy
```

**macOS:**

```bash
brew install openblas lapack clang-format python numpy
```

### Building

```bash
git clone https://github.com/disruptek/srukf.git
cd srukf
make          # builds library and runs tests
make lib      # builds libsrukf.so and libsrukf.a
make test     # runs the test suite
make bench    # runs benchmarks
```

### The Two Build Systems

The project ships **two first-class build systems** serving different
audiences:

- **make** is the minimal path: it needs only a C compiler, binutils,
  and BLAS/LAPACKE. It builds, tests, and drives the sanitizer and
  coverage workflows. Users on stripped-down systems build with make
  alone — it must never grow a dependency on CMake.
- **CMake** is the integration surface: it provides `find_package(srukf)`
  exports, build options, and feeds the vcpkg/Conan packaging.

Neither wraps the other, and CI builds and tests both (including a job
that fails if the two systems disagree on the number of tests). To keep
them from drifting, **build knowledge lives in the source tree, not in
the build files**:

- Test classification is derived, not listed. A test that compiles the
  library internally does `#include "srukf.c"`; both build systems
  detect that include and compile it standalone. All other `tests/*.c`
  files link against the library. **Adding a test requires no
  build-system edits** — drop the file in `tests/` and follow the
  convention. (The one exception: `90_cpp_linkage.cpp` is registered
  explicitly in both systems because C++ availability is optional.)
- A test needing single precision defines `SRUKF_SINGLE` itself before
  including `srukf.c` (see `tests/47_single_precision.c`); there are no
  per-test compiler flags in the build files.
- The version number lives in `srukf.h` (`SRUKF_VERSION_*`); CMake and
  make both parse it from there. Do not version the C library anywhere
  else. (The Python package carries its own `version.py`.)
- Dependencies resolve through pkg-config in both systems with the same
  cblas → blas → openblas fallback chain; the Makefile keeps a static
  fallback and `LDFLAGS`/`CFLAGS` remain overridable for systems
  without pkg-config.

The invariant: **anything that changes what gets compiled must be
expressible as a file-level convention, never as a parallel edit to both
build files.** If you find yourself editing the Makefile and
CMakeLists.txt in the same commit, look for the convention you're
missing.

### Python Bindings

```bash
cd python
pip install -e ".[test]"
pytest
```

## Pull Request Process

1. Pull the latest `main` branch.
2. Create a new branch named after the feature or fix.
3. Make your changes.
4. Run the test suite: `make test`
5. If adding a feature, add an example that demonstrates every facet of the change.
6. Commit with a clear, descriptive message.
7. Pull `main` and rebase your branch onto it.
8. Run the test suite again: `make test`
9. Run the linter: `make format` (uses clang-format).
10. Run the test suite one more time: `make test`
11. Commit any formatting changes if necessary.
12. Push your branch and open a pull request against `main`.
13. Watch for CI failures or merge conflicts and address them promptly.

## Testing Requirements

All changes must pass the existing test suite. The tests cover:

| Test file | What it tests |
|---|---|
| `00_sigma.c` | Sigma point generation |
| `01_weights.c`, `02_weights.c` | UKF weight computation |
| `05_correct.c`, `06_predict.c` | Core predict/correct operations |
| `10_simple.c` | End-to-end linear tracking |
| `20_nonlinear.c` | Nonlinear system tracking |
| `30_errors.c` | Error handling and edge cases |
| `35_numerical.c` | Numerical stability |
| `40_stress.c` | Long-duration stress tests |
| `45_accessors.c` | Safe state/covariance access, version, innovation/NIS |
| `46_edge_cases.c` | Boundary conditions |
| `47_single_precision.c` | Single-precision (`float`) mode |
| `48_atomicity.c` | Transactional semantics of predict/correct on error |
| `49_zeroalloc.c` | Zero heap allocation in the steady-state hot path |
| `90_cpp_linkage.cpp` | Public header consumable from C++ |

**Run all tests:**

```bash
make test
```

**Run with verbose output:**

```bash
make test-verbose
```

**Run with sanitizers** (recommended before submitting):

```bash
# AddressSanitizer
make clean
make test CFLAGS="-Wall -Wextra -Wpedantic -O1 -g -fPIC -I. -DHAVE_LAPACK \
    -fsanitize=address -fno-omit-frame-pointer" \
    LDFLAGS="-lm -llapacke -lblas -lopenblas -fsanitize=address"

# UndefinedBehaviorSanitizer
make clean
make test CFLAGS="-Wall -Wextra -Wpedantic -O1 -g -fPIC -I. -DHAVE_LAPACK \
    -fsanitize=undefined -fno-omit-frame-pointer" \
    LDFLAGS="-lm -llapacke -lblas -lopenblas -fsanitize=undefined"
```

**Python tests:**

```bash
cd python
pytest
```

### Coverage

```bash
make coverage
# Output: srukf.c.gcov
```

## Code Style

This project uses **clang-format** with the configuration in `.clang-format`:

- Based on LLVM style
- 2-space indentation
- 80-column limit
- Pointer alignment: right (`char *p`, not `char* p`)

**Format all source files:**

```bash
make format
```

### Naming Conventions

- Public API: `srukf_` prefix (e.g., `srukf_create`, `srukf_predict`)
- Public types: `srukf_` prefix (e.g., `srukf_mat`, `srukf_value`)
- Public macros: `SRUKF_` prefix (e.g., `SRUKF_ENTRY`, `SRUKF_MAT_ALLOC`)
- Internal functions: `static`, no prefix required
- Test files: numbered prefix for ordering (e.g., `10_simple.c`)

### General Guidelines

- C99 standard; no compiler-specific extensions in the public API.
- All matrices are column-major (Fortran order) for BLAS/LAPACK compatibility.
- Noise matrices are always in square-root form.
- Functions return `srukf_return` codes; never `abort()` or `exit()` from library code.
- No heap allocation in the predict/correct hot path when workspace is pre-allocated.

## Reporting Issues

### Bug Reports

Please include:

1. **System info** — OS, compiler version, BLAS/LAPACK implementation.
2. **Minimal reproducer** — Smallest code that triggers the bug.
3. **Expected vs. actual behavior** — What you expected and what happened.
4. **Filter dimensions** — State dimension N, measurement dimension M.
5. **Sanitizer output** — If applicable, run with ASan/UBSan and include the output.

### Feature Requests

Please describe:

1. **Use case** — What problem are you solving?
2. **Proposed API** — How would you like to call it?
3. **Alternatives considered** — What workarounds exist today?

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
