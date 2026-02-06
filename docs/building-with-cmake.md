# Building with CMake

The SR-UKF library provides a CMake build system alongside the existing
Makefile.  Both produce identical outputs; use whichever fits your workflow.

## Prerequisites

- CMake 3.15+
- A C99 compiler (GCC, Clang)
- LAPACK and BLAS development libraries (e.g. `libopenblas-dev` on
  Debian/Ubuntu, `sci-libs/openblas` on Gentoo)

## Quick start

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
ctest --output-on-failure
```

## Build options

| Option                  | Default | Description                          |
|-------------------------|---------|--------------------------------------|
| `SRUKF_BUILD_SHARED`   | `ON`    | Build `libsrukf.so`                 |
| `SRUKF_BUILD_STATIC`   | `ON`    | Build `libsrukf.a`                  |
| `SRUKF_BUILD_TESTS`    | `ON`    | Build and register test binaries     |
| `SRUKF_BUILD_EXAMPLES` | `ON`    | Build example programs               |
| `SRUKF_BUILD_BENCHMARKS`| `ON`   | Build benchmark programs             |
| `SRUKF_SINGLE_PRECISION`| `OFF`  | Use `float` instead of `double`     |

Example — static library only, no examples:

```bash
cmake .. -DSRUKF_BUILD_SHARED=OFF -DSRUKF_BUILD_EXAMPLES=OFF
```

## Installing

```bash
cmake --install . --prefix /usr/local
```

This installs:

```
<prefix>/
  include/srukf.h
  lib/libsrukf.so       (shared, if enabled)
  lib/libsrukf.so.1
  lib/libsrukf.so.1.0.0
  lib/libsrukf.a         (static, if enabled)
  lib/cmake/srukf/
    srukfConfig.cmake
    srukfConfigVersion.cmake
    srukfTargets.cmake
```

## Using from a downstream CMake project

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_app C)

find_package(srukf REQUIRED)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE srukf::srukf)
```

Configure with:

```bash
cmake . -DCMAKE_PREFIX_PATH=/path/to/srukf/install
```

The `srukf::srukf` target brings in include paths and link dependencies
(LAPACK, BLAS, libm) automatically.

## Running benchmarks

```bash
cmake --build . --target benchmark
./benchmark

# Generate SVG charts (requires Python 3)
cmake --build . --target bench-chart
cmake --build . --target bench-memory-chart
```
