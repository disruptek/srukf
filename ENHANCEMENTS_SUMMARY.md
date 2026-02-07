# SR-UKF Enhancement Summary

## 1. Executive Summary

### What Was Implemented

The SR-UKF project received a comprehensive modernization across four areas:

1. **CMake build system** with pkg-config, `find_package()` support, and full parity with the existing Makefile
2. **Package manager integration** for Conan 2.x and vcpkg, enabling one-command installation
3. **Python bindings** via ctypes with a Pythonic API, test suite, examples, and Jupyter tutorial
4. **CI/CD pipeline** with code coverage, sanitizer testing, CMake verification, Python matrix testing, and automated benchmark tracking

### Why These Enhancements Matter

The core SR-UKF library is production-quality C99 with excellent numerical properties. However, adoption was limited by:

- **No standard build system** &mdash; users of CMake-based projects (the vast majority of C/C++ projects) had to manually wire include paths and link flags
- **No package manager presence** &mdash; every consumer had to clone, build, and install manually
- **No Python interface** &mdash; the fastest-growing segment of the sensor fusion community (robotics researchers, data scientists, rapid prototypers) was excluded
- **No automated quality gates** &mdash; no coverage tracking, no sanitizer runs, no regression detection

These enhancements remove every major barrier to adoption without changing a single line of the core library.

### Impact on Adoption and Usability

| Before | After |
|--------|-------|
| Manual `gcc` invocation or Makefile only | CMake, Makefile, Conan, vcpkg |
| Copy header + source into your project | `find_package(srukf)` or `pip install srukf` |
| C users only | C, C++, and Python users |
| No coverage data | Codecov integration with per-PR reports |
| No regression detection | Automated benchmark tracking with alerts |
| No sanitizer testing | ASan + UBSan on every push |

---

## 2. Completed Enhancements

### 2.1 Build System Modernization (CMake + pkg-config)

**Branch:** `pkg-config-and-package-managers`

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Full CMake build system (355 lines) |
| `srukf.pc.in` | pkg-config template |
| `cmake/srukfConfig.cmake.in` | CMake package config for `find_package()` |
| `docs/building-with-cmake.md` | User-facing CMake documentation |

**Key capabilities:**

- Shared and static library targets (`libsrukf.so`, `libsrukf.a`)
- All 13 tests registered with CTest (internal, public API, and single-precision)
- All 3 examples built as CMake targets
- Both benchmarks with custom chart-generation targets
- Proper `install()` with export sets, versioned SONAME, and CMake package config
- pkg-config `.pc` file generated at configure time
- Single-precision mode via `-DSRUKF_SINGLE_PRECISION=ON`
- Dependencies resolved via `pkg_check_modules()` (works on CMake 3.15 through 4.x)

### 2.2 Package Manager Integration (Conan + vcpkg)

**Branch:** `pkg-config-and-package-managers`

| File | Purpose |
|------|---------|
| `conanfile.py` | Conan 2.x recipe |
| `vcpkg.json` | vcpkg manifest |
| `docs/vcpkg-port-guide.md` | Step-by-step vcpkg submission guide |

**Conan recipe features:**

- Configurable options: `shared`, `single_precision`, `build_tests`, `build_examples`
- System requirements advisory for Linux, macOS
- Proper `cmake_layout()` integration
- License file installation
- Component-level `package_info()`

**vcpkg manifest features:**

- Declares dependencies on `openblas` and `lapack` from the vcpkg registry
- Uses `vcpkg-cmake` and `vcpkg-cmake-config` build helpers
- Includes a ready-to-submit `portfile.cmake` template in the docs

### 2.3 Python Bindings

**Branch:** `ci-coverage-cmake-python`

| File | Purpose |
|------|---------|
| `python/srukf/__init__.py` | Package entry point with public API |
| `python/srukf/core.py` | `UnscentedKalmanFilter` class (482 lines) |
| `python/srukf/_bindings.py` | ctypes declarations and library loading (321 lines) |
| `python/srukf/utils.py` | numpy/C matrix conversion utilities (211 lines) |
| `python/srukf/version.py` | Version metadata |
| `python/setup.py` | Build script that auto-compiles `libsrukf.so` |
| `python/pyproject.toml` | PEP 621 project metadata |
| `python/MANIFEST.in` | Source distribution manifest |
| `python/build_library.py` | Standalone library build helper |
| `python/README.md` | Python-specific documentation |
| `python/tests/test_basic.py` | 20 tests: creation, state, covariance, predict/update |
| `python/tests/test_matrix.py` | 12 tests: numpy/C round-trip, validation |
| `python/tests/test_pendulum.py` | 7 tests: full nonlinear simulation, numerics |
| `python/examples/simple.py` | Minimal 1D tracking example |
| `python/examples/pendulum.py` | Nonlinear pendulum with optional matplotlib |
| `python/examples/notebooks/tutorial.ipynb` | Interactive Jupyter tutorial |

**Architecture decisions:**

- **ctypes over CFFI/pybind11** &mdash; zero additional build dependencies; works with any Python 3.7+ without a compiler at import time
- **Library discovery chain** &mdash; bundled `.so` > repository root > `SRUKF_LIB_PATH` env var > system paths
- **Data copying at the boundary** &mdash; numpy arrays are copied into C-allocated `srukf_mat` structs to avoid GC/lifetime issues; a zero-copy `srukf_mat_read_into()` is available for hot loops
- **Callback safety** &mdash; Python callbacks are wrapped in `CFUNCTYPE` closures with explicit reference retention to prevent GC during C calls
- **Exception hierarchy** &mdash; `SrukfError` > `SrukfParameterError` (also a `ValueError`) and `SrukfMathError`, mapped from C return codes
- **Method chaining** &mdash; all mutating methods return `self`

### 2.4 Documentation Improvements

**Branch:** `ci-coverage-cmake-python`

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | 5-minute getting started guide (C and Python) |
| `CONTRIBUTING.md` | Contributor guide with PR process, testing, style |
| `CITATION.cff` | GitHub citation metadata |
| `.zenodo.json` | Zenodo DOI metadata |
| `README.md` | Updated with badges (CI, coverage, PyPI, docs) |

### 2.5 CI/CD Enhancements

**Branch:** `ci-coverage-cmake-python`

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | 8-job CI pipeline (311 lines) |
| `.github/workflows/benchmark.yml` | Automated benchmark tracking (84 lines) |
| `.github/pull_request_template.md` | PR checklist template |

**CI jobs:**

| Job | What it does | Blocks PRs? |
|-----|-------------|-------------|
| `build` | `make lib && make test && make bench` | Yes |
| `examples` | Build and run all 3 examples with output validation | Yes |
| `sanitize-address` | ASan + LeakSanitizer on all tests | Advisory |
| `sanitize-undefined` | UBSan on all tests | Advisory |
| `examples-sanitize` | ASan on all examples | Advisory |
| `coverage` | gcov + lcov, upload to Codecov, HTML artifact | Yes |
| `cmake-build` | CMake Release + Debug, CTest, install, downstream `find_package()` smoke test | Yes |
| `python-package` | Python 3.8&ndash;3.12 matrix: install, pytest, run examples | Yes |

**Benchmark tracking:**

- Converts benchmark output to `benchmark-action/github-action-benchmark` JSON format
- Stores historical data on `gh-pages` branch
- Alerts on 20%+ regressions via PR comments
- Auto-pushes results on main branch merges

---

## 3. Technical Details

### 3.1 CMake Build System

**What was added:** A 355-line `CMakeLists.txt` that mirrors the Makefile exactly.

**Key design decisions:**

- Uses `pkg_check_modules()` instead of `FindBLAS`/`FindLAPACK` because CMake 4.x removed the built-in Find modules. pkg-config works across all CMake versions 3.15+.
- Validates that `cblas.h` and `lapacke.h` are actually usable via `check_include_file()`.
- Internal tests (which `#include "srukf.c"` directly) are compiled standalone with the correct include paths and link flags, not linked against the library.
- Test targets unconditionally add `-UNDEBUG` to ensure `assert()` works in Release builds.
- The single-precision test is always compiled with `-DSRUKF_SINGLE` regardless of the global `SRUKF_SINGLE_PRECISION` option.
- Benchmark targets enable C extensions (`C_EXTENSIONS ON`) because they use `clock_gettime()`.

**How to use it:**

```bash
# Basic build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

# Install
cmake --install build --prefix /usr/local

# Use from downstream project
find_package(srukf REQUIRED)
target_link_libraries(my_app PRIVATE srukf::srukf)
```

**Verification performed:**

- CI runs CMake in both Release and Debug configurations
- CI installs to a temporary prefix and builds a downstream smoke test that calls `srukf_create()` / `srukf_free()` via `find_package(srukf)`
- All 13 CTest tests pass in both configurations

### 3.2 Package Manager Integration

**Conan &mdash; what was added:** An 86-line `conanfile.py` using the Conan 2.x API.

**How to use it:**

```bash
# Local development
conan create . --build=missing

# In a consumer project's conanfile.txt:
[requires]
srukf/1.0.0

[generators]
CMakeDeps
CMakeToolchain
```

**vcpkg &mdash; what was added:** A `vcpkg.json` manifest and a `docs/vcpkg-port-guide.md` with a complete `portfile.cmake` template.

**How to use it (after port submission):**

```bash
vcpkg install srukf
```

Or in a CMake project with vcpkg toolchain:

```cmake
find_package(srukf REQUIRED)
target_link_libraries(my_app PRIVATE srukf::srukf)
```

### 3.3 Python Bindings

**What was added:** A complete Python package in `python/` with 1,200+ lines of code across 6 modules.

**Key design decisions:**

- **ctypes** was chosen over CFFI, pybind11, or Cython because it requires zero compilation at import time and has no dependencies beyond the standard library. The C library is loaded as a pre-built `.so`.
- The `_bindings.py` module declares every public C function's `argtypes` and `restype` for type safety and crash prevention.
- Matrix conversion copies data at the Python/C boundary. This is safe and simple. For hot loops, `srukf_mat_read_into()` provides a pre-allocated-buffer path.
- The `UnscentedKalmanFilter` class owns the C filter pointer and frees it in `__del__`. Callback closures are pinned in `_live_callbacks` during C calls.

**How to use it:**

```python
import numpy as np
from srukf import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(
    state_dim=2, meas_dim=1,
    process_noise_sqrt=np.diag([0.1, 0.01]),
    meas_noise_sqrt=np.array([[0.5]]),
)

ukf.predict(lambda x: np.array([x[0] + 0.1 * x[1], x[1]]))
ukf.update(np.array([1.5]), lambda x: np.array([x[0]]))

print(ukf.x)    # state estimate
print(ukf.P)    # full covariance
```

**Testing performed:**

- 39 pytest tests across 3 test files
- Tests cover: creation, destruction, state get/set, covariance get/set, reset, predict, update, method chaining, kwargs forwarding, error handling, matrix round-trips, validation, nonlinear pendulum tracking, numerical properties
- CI runs tests against Python 3.8, 3.9, 3.10, 3.11, and 3.12

### 3.4 CI/CD Pipeline

**What was added:** Two GitHub Actions workflows totaling 395 lines.

**Key design decisions:**

- Sanitizer jobs use `continue-on-error: true` so they report issues without blocking merges. This is intentional: OpenBLAS occasionally triggers ASan false positives in its internal threading.
- The coverage job uploads both to Codecov (for badge/PR comments) and as a GitHub Actions artifact (for offline review).
- The CMake CI job tests the full install-and-consume cycle, not just "does it compile."
- The benchmark workflow uses `benchmark-action/github-action-benchmark` for historical tracking with configurable alert thresholds.

---

## 4. Quick Start for New Features

### How to Use CMake

```bash
# Prerequisites (Ubuntu/Debian)
sudo apt-get install libopenblas-dev liblapacke-dev cmake

# Build
git clone https://github.com/disruptek/srukf.git
cd srukf
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Test
ctest --test-dir build --output-on-failure

# Install
sudo cmake --install build

# Use in your CMakeLists.txt
# find_package(srukf REQUIRED)
# target_link_libraries(my_app PRIVATE srukf::srukf)
```

### How to Install via Package Managers

**Conan:**

```bash
# From the srukf repository
conan create . --build=missing

# Or add to your conanfile.txt
# [requires]
# srukf/1.0.0
```

**vcpkg (after port submission):**

```bash
vcpkg install srukf
```

### How to Use Python Bindings

```bash
# Prerequisites
sudo apt-get install libopenblas-dev liblapacke-dev

# Install
git clone https://github.com/disruptek/srukf.git
cd srukf
make lib
cd python
pip install -e '.[test]'

# Verify
pytest -v

# Run examples
python examples/simple.py
python examples/pendulum.py --plot
```

### How to Check Coverage Reports

**Locally:**

```bash
make coverage
# Output: srukf.c.gcov

# For HTML report:
sudo apt-get install lcov
make coverage
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
genhtml coverage.info --output-directory coverage-report
xdg-open coverage-report/index.html
```

**On CI:**

- Codecov badge in README links to per-file coverage
- HTML report available as a downloadable artifact on every CI run

---

## 5. Branch Organization

| Branch | Base | Content | Status |
|--------|------|---------|--------|
| `main` | &mdash; | Core library, Makefile, tests, examples, docs | Stable |
| `pkg-config-and-package-managers` | `main` | CMake, pkg-config, Conan, vcpkg | Ready for PR |
| `ci-coverage-cmake-python` | `main` | CI/CD, coverage, Python bindings, docs, badges | Ready for PR |

### Dependency Between Branches

The two enhancement branches are **independent** &mdash; they both branch from `main` and do not depend on each other. They can be merged in any order.

However, the CI workflow in `ci-coverage-cmake-python` includes a `cmake-build` job that expects `CMakeLists.txt` to exist. **Recommended merge order:**

1. Merge `pkg-config-and-package-managers` first (adds CMake)
2. Rebase `ci-coverage-cmake-python` onto the updated `main`
3. Merge `ci-coverage-cmake-python`

Alternatively, merge both simultaneously and resolve the trivial `.gitignore` conflict.

### How to Merge

```bash
# Option A: Sequential merge
git checkout main && git pull

# First PR
git checkout pkg-config-and-package-managers
git rebase main
git push -u origin pkg-config-and-package-managers
gh pr create --title "Add CMake build system, pkg-config, Conan, and vcpkg support" \
  --body "Adds a CMake build system with full parity to the Makefile, plus pkg-config and package manager integration."

# After first PR merges:
git checkout main && git pull
git checkout ci-coverage-cmake-python
git rebase main
git push -u origin ci-coverage-cmake-python
gh pr create --title "Add CI/CD pipeline, coverage, Python bindings, and documentation" \
  --body "Adds comprehensive CI, code coverage, Python bindings, and contributor documentation."
```

---

## 6. Next Steps

### Immediate Actions

- [ ] Create PRs for both branches (see merge order above)
- [ ] Set up Codecov integration (add `CODECOV_TOKEN` to repository secrets)
- [ ] Enable GitHub Pages deployment (Settings > Pages > GitHub Actions)
- [ ] Verify CI passes on both PRs
- [ ] Review and merge

### Short-Term (1&ndash;4 weeks)

- [ ] **PyPI publishing** &mdash; add a `publish.yml` workflow triggered on GitHub releases that runs `python -m build && twine upload`
- [ ] **Conan Center submission** &mdash; fork `conan-center-index`, add the recipe, submit PR
- [ ] **vcpkg submission** &mdash; fork `microsoft/vcpkg`, follow `docs/vcpkg-port-guide.md`, submit PR
- [ ] **Zenodo DOI** &mdash; link the GitHub repository to Zenodo for automatic DOI minting on releases
- [ ] **Tag v1.0.0** &mdash; create a Git tag and GitHub release

### Long-Term (1&ndash;6 months)

- [ ] **ROS 2 wrapper** &mdash; a `srukf_ros` package that wraps the filter as a ROS 2 node with `sensor_msgs` input and `nav_msgs/Odometry` output
- [ ] **WebAssembly build** &mdash; compile with Emscripten for browser-based demos (the web explainer could use the real library instead of a JS reimplementation)
- [ ] **MATLAB/Simulink MEX interface** &mdash; for the controls engineering community
- [ ] **Adaptive noise estimation** &mdash; online Q/R tuning via innovation-based methods
- [ ] **Multi-rate measurement support** &mdash; first-class API for sensors at different frequencies (currently handled manually in user code)

---

## 7. Migration Guide

### For Existing Users: Nothing Breaks

The enhancements are purely additive:

- The Makefile is unchanged. `make lib && make test` works exactly as before.
- The C API (`srukf.h`) is unchanged. No symbols were added, removed, or renamed.
- The library ABI is unchanged. Existing binaries linked against `libsrukf.so` continue to work.
- The `.gitignore` has minor additions for build artifacts.

**You can ignore every enhancement and continue using the project exactly as you did before.**

### For New Users: Recommended Workflow

**C/C++ users:**

```bash
# Clone and build with CMake
git clone https://github.com/disruptek/srukf.git
cd srukf
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build

# In your project's CMakeLists.txt:
find_package(srukf REQUIRED)
target_link_libraries(my_app PRIVATE srukf::srukf)
```

**Python users:**

```bash
git clone https://github.com/disruptek/srukf.git
cd srukf && make lib
cd python && pip install -e .

# Then:
from srukf import UnscentedKalmanFilter
```

**Learning path:**

1. Read `QUICKSTART.md` (5 minutes)
2. Open the interactive web explainer in `examples/web_explainer/` (10 minutes)
3. Work through the pendulum example in `examples/01_pendulum/` (20 minutes)
4. Try the Python tutorial notebook in `python/examples/notebooks/tutorial.ipynb`

### For Contributors: New Testing Requirements

All PRs should now pass:

| Check | Command | Required? |
|-------|---------|-----------|
| Makefile tests | `make test` | Yes |
| CMake tests | `cmake -B build && cmake --build build && ctest --test-dir build` | Yes |
| Python tests | `cd python && pytest` | Yes (if touching Python code) |
| Sanitizers | See `CONTRIBUTING.md` | Recommended |
| Formatting | `make format` | Yes |
| Examples | Build and run all 3 examples | Yes (if touching core library) |

The PR template (`.github/pull_request_template.md`) includes a checklist covering all of these.

**Code style:**

- C: clang-format with LLVM base, 2-space indent, 80-column limit (see `.clang-format`)
- Python: standard Python conventions; numpy-style docstrings
- CMake: 2-space indent, lowercase commands

**Naming conventions:**

- Public C API: `srukf_` prefix
- Public C macros: `SRUKF_` prefix
- Python: `UnscentedKalmanFilter` class, `SrukfError` exceptions
- Test files: numbered prefix for ordering (`00_sigma.c`, `01_weights.c`, ...)
