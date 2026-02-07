cmake_minimum_required(VERSION 3.15)

project(srukf
  VERSION 1.0.0
  DESCRIPTION "Square-Root Unscented Kalman Filter"
  LANGUAGES C
)

# ── Options ──────────────────────────────────────────────────────────
option(SRUKF_BUILD_SHARED     "Build shared library"   ON)
option(SRUKF_BUILD_STATIC     "Build static library"   ON)
option(SRUKF_BUILD_TESTS      "Build tests"            ON)
option(SRUKF_BUILD_EXAMPLES   "Build examples"         ON)
option(SRUKF_BUILD_BENCHMARKS "Build benchmarks"       ON)
option(SRUKF_SINGLE_PRECISION "Use single precision (float instead of double)" OFF)

if(NOT SRUKF_BUILD_SHARED AND NOT SRUKF_BUILD_STATIC)
  message(FATAL_ERROR "At least one of SRUKF_BUILD_SHARED or SRUKF_BUILD_STATIC must be ON")
endif()

# ── Dependencies ─────────────────────────────────────────────────────
# The Makefile links: -lm -llapacke -lblas -lopenblas
# We use pkg-config to locate these portably across CMake 3.15+ and 4.x
# (CMake 4 removed the built-in FindBLAS/FindLAPACK modules).
find_package(PkgConfig MODULE REQUIRED)
pkg_check_modules(LAPACKE REQUIRED IMPORTED_TARGET lapacke)

# Try to find CBLAS - systems name it differently (cblas, blas, or openblas)
pkg_check_modules(CBLAS IMPORTED_TARGET cblas)
if(NOT CBLAS_FOUND)
  pkg_check_modules(CBLAS IMPORTED_TARGET blas)
endif()
if(NOT CBLAS_FOUND)
  pkg_check_modules(CBLAS REQUIRED IMPORTED_TARGET openblas)
endif()

# Verify the C headers are actually usable.
include(CheckIncludeFile)
set(CMAKE_REQUIRED_INCLUDES ${LAPACKE_INCLUDE_DIRS} ${CBLAS_INCLUDE_DIRS})
check_include_file(cblas.h   HAVE_CBLAS_H)
check_include_file(lapacke.h HAVE_LAPACKE_H)
if(NOT HAVE_CBLAS_H OR NOT HAVE_LAPACKE_H)
  message(FATAL_ERROR
    "cblas.h and lapacke.h are required.  "
    "Install LAPACK/BLAS development packages (e.g. libopenblas-dev)."
  )
endif()

# Aggregate link libraries.  PkgConfig::LAPACKE and PkgConfig::CBLAS
# carry include dirs, compile flags, and link flags automatically.
set(SRUKF_LINK_LIBS PkgConfig::LAPACKE PkgConfig::CBLAS m)

# ── Compile flags matching the Makefile ──────────────────────────────
set(SRUKF_PRIVATE_COMPILE_OPTIONS
  -Wall -Wextra -Wpedantic
)
set(SRUKF_PRIVATE_COMPILE_DEFINITIONS
  HAVE_LAPACK
)
if(SRUKF_SINGLE_PRECISION)
  list(APPEND SRUKF_PRIVATE_COMPILE_DEFINITIONS SRUKF_SINGLE)
endif()

# ── Helper: configure one library target ─────────────────────────────
function(srukf_configure_target target)
  target_sources(${target} PRIVATE srukf.c)

  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  target_compile_options(${target} PRIVATE ${SRUKF_PRIVATE_COMPILE_OPTIONS})
  target_compile_definitions(${target} PRIVATE ${SRUKF_PRIVATE_COMPILE_DEFINITIONS})
  target_link_libraries(${target} PUBLIC ${SRUKF_LINK_LIBS})

  set_target_properties(${target} PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        OFF
  )
endfunction()

# ── Library targets ──────────────────────────────────────────────────
include(GNUInstallDirs)

if(SRUKF_BUILD_SHARED)
  add_library(srukf_shared SHARED)
  srukf_configure_target(srukf_shared)
  set_target_properties(srukf_shared PROPERTIES
    OUTPUT_NAME   srukf
    VERSION       ${PROJECT_VERSION}
    SOVERSION     ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
  )
  add_library(srukf::srukf ALIAS srukf_shared)
endif()

if(SRUKF_BUILD_STATIC)
  add_library(srukf_static STATIC)
  srukf_configure_target(srukf_static)
  set_target_properties(srukf_static PROPERTIES
    OUTPUT_NAME   srukf
    POSITION_INDEPENDENT_CODE ON
  )
  if(NOT SRUKF_BUILD_SHARED)
    add_library(srukf::srukf ALIAS srukf_static)
  endif()
endif()

# ── Tests ────────────────────────────────────────────────────────────
if(SRUKF_BUILD_TESTS)
  enable_testing()

  # Internal tests: compile srukf.c directly (the test #includes srukf.c)
  set(INTERNAL_TESTS
    00_sigma
    06_predict
    10_simple
    20_nonlinear
    30_errors
    35_numerical
    40_stress
    46_edge_cases
  )

  # Public API tests: link against the library
  set(PUBLIC_TESTS
    01_weights
    02_weights
    05_correct
    45_accessors
  )

  # Preferred link target for public API tests and examples
  if(SRUKF_BUILD_SHARED)
    set(SRUKF_LINK_TARGET srukf_shared)
  else()
    set(SRUKF_LINK_TARGET srukf_static)
  endif()

  # Tests use assert() for validation.  CMake's Release/RelWithDebInfo
  # build types add -DNDEBUG which disables assert().  We must ensure
  # NDEBUG is never defined for test targets.
  #
  # The portable approach: add -UNDEBUG after any -DNDEBUG that CMake
  # injects via CMAKE_C_FLAGS_<CONFIG>.  This works on GCC, Clang, and
  # MSVC (/UNDEBUG is not needed — MSVC uses /D, but the principle holds).
  foreach(test_name IN LISTS INTERNAL_TESTS)
    add_executable(test_${test_name} tests/${test_name}.c)
    # These tests #include "srukf.c" themselves, so they compile the
    # library source directly.  We only need the include path and deps.
    target_include_directories(test_${test_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(test_${test_name} PRIVATE
      ${SRUKF_PRIVATE_COMPILE_OPTIONS} -UNDEBUG
    )
    target_compile_definitions(test_${test_name} PRIVATE ${SRUKF_PRIVATE_COMPILE_DEFINITIONS})
    target_link_libraries(test_${test_name} PRIVATE ${SRUKF_LINK_LIBS})
    set_target_properties(test_${test_name} PROPERTIES
      C_STANDARD          99
      C_STANDARD_REQUIRED ON
      C_EXTENSIONS        OFF
    )
    add_test(NAME ${test_name} COMMAND test_${test_name})
  endforeach()

  foreach(test_name IN LISTS PUBLIC_TESTS)
    add_executable(test_${test_name} tests/${test_name}.c)
    target_link_libraries(test_${test_name} PRIVATE ${SRUKF_LINK_TARGET})
    target_compile_options(test_${test_name} PRIVATE
      ${SRUKF_PRIVATE_COMPILE_OPTIONS} -UNDEBUG
    )
    set_target_properties(test_${test_name} PROPERTIES
      C_STANDARD          99
      C_STANDARD_REQUIRED ON
      C_EXTENSIONS        OFF
    )
    add_test(NAME ${test_name} COMMAND test_${test_name})
  endforeach()

  # Single-precision test (always compiled with -DSRUKF_SINGLE regardless
  # of the global SRUKF_SINGLE_PRECISION option)
  add_executable(test_47_single_precision tests/47_single_precision.c)
  target_include_directories(test_47_single_precision PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_compile_options(test_47_single_precision PRIVATE
    ${SRUKF_PRIVATE_COMPILE_OPTIONS} -UNDEBUG
  )
  target_compile_definitions(test_47_single_precision PRIVATE HAVE_LAPACK SRUKF_SINGLE)
  target_link_libraries(test_47_single_precision PRIVATE ${SRUKF_LINK_LIBS})
  set_target_properties(test_47_single_precision PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        OFF
  )
  add_test(NAME 47_single_precision COMMAND test_47_single_precision)
endif()

# ── Examples ─────────────────────────────────────────────────────────
if(SRUKF_BUILD_EXAMPLES)
  if(NOT SRUKF_BUILD_SHARED AND NOT SRUKF_BUILD_STATIC)
    message(FATAL_ERROR "Examples require at least one library variant")
  endif()

  # Preferred link target
  if(NOT DEFINED SRUKF_LINK_TARGET)
    if(SRUKF_BUILD_SHARED)
      set(SRUKF_LINK_TARGET srukf_shared)
    else()
      set(SRUKF_LINK_TARGET srukf_static)
    endif()
  endif()

  # Common plot utilities (static helper library, not installed)
  add_library(srukf_plot_utils STATIC examples/common/plot_utils.c)
  target_include_directories(srukf_plot_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
  )
  target_compile_options(srukf_plot_utils PRIVATE ${SRUKF_PRIVATE_COMPILE_OPTIONS})
  set_target_properties(srukf_plot_utils PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        OFF
  )

  # 01_pendulum
  add_executable(example_pendulum examples/01_pendulum/pendulum.c)
  target_link_libraries(example_pendulum PRIVATE ${SRUKF_LINK_TARGET} srukf_plot_utils m)
  target_compile_options(example_pendulum PRIVATE ${SRUKF_PRIVATE_COMPILE_OPTIONS})
  set_target_properties(example_pendulum PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        OFF
  )

  # 02_gps_imu_1d
  add_executable(example_gps_imu examples/02_gps_imu_1d/gps_imu.c)
  target_link_libraries(example_gps_imu PRIVATE ${SRUKF_LINK_TARGET} srukf_plot_utils m)
  target_compile_options(example_gps_imu PRIVATE ${SRUKF_PRIVATE_COMPILE_OPTIONS})
  set_target_properties(example_gps_imu PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        OFF
  )

  # 03_stability_test
  add_executable(example_stability examples/03_stability_test/stability.c)
  target_link_libraries(example_stability PRIVATE ${SRUKF_LINK_TARGET} srukf_plot_utils m)
  target_include_directories(example_stability PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/common
  )
  target_compile_options(example_stability PRIVATE ${SRUKF_PRIVATE_COMPILE_OPTIONS})
  set_target_properties(example_stability PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        OFF
  )
endif()

# ── Benchmarks ───────────────────────────────────────────────────────
if(SRUKF_BUILD_BENCHMARKS)
  if(NOT DEFINED SRUKF_LINK_TARGET)
    if(SRUKF_BUILD_SHARED)
      set(SRUKF_LINK_TARGET srukf_shared)
    else()
      set(SRUKF_LINK_TARGET srukf_static)
    endif()
  endif()

  # Benchmarks use clock_gettime (POSIX), so GNU extensions are required.
  # The root Makefile does not restrict the C standard, matching this.
  add_executable(benchmark benchmark/benchmark.c)
  target_link_libraries(benchmark PRIVATE ${SRUKF_LINK_TARGET})
  target_compile_options(benchmark PRIVATE ${SRUKF_PRIVATE_COMPILE_OPTIONS})
  set_target_properties(benchmark PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        ON
  )

  add_executable(memory_bench benchmark/memory_bench.c)
  target_link_libraries(memory_bench PRIVATE ${SRUKF_LINK_TARGET})
  target_compile_options(memory_bench PRIVATE ${SRUKF_PRIVATE_COMPILE_OPTIONS})
  set_target_properties(memory_bench PROPERTIES
    C_STANDARD          99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS        ON
  )

  # Convenience targets for generating charts
  add_custom_target(bench-chart
    COMMAND benchmark | python3 ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/generate_chart.py
            > ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/benchmark.svg
    DEPENDS benchmark
    COMMENT "Running benchmark and generating benchmark/benchmark.svg"
  )

  add_custom_target(bench-memory-chart
    COMMAND memory_bench | python3 ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/generate_memory_chart.py
            > ${CMAKE_CURRENT_SOURCE_DIR}/benchmark/memory.svg
    DEPENDS memory_bench
    COMMENT "Running memory benchmark and generating benchmark/memory.svg"
  )
endif()

# ── pkg-config ────────────────────────────────────────────────────────
configure_file(srukf.pc.in srukf.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/srukf.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ── Installation ─────────────────────────────────────────────────────
# Collect targets to install
set(SRUKF_INSTALL_TARGETS "")
if(SRUKF_BUILD_SHARED)
  list(APPEND SRUKF_INSTALL_TARGETS srukf_shared)
endif()
if(SRUKF_BUILD_STATIC)
  list(APPEND SRUKF_INSTALL_TARGETS srukf_static)
endif()

install(TARGETS ${SRUKF_INSTALL_TARGETS}
  EXPORT srukfTargets
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES srukf.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ── CMake package config for find_package(srukf) ────────────────────
include(CMakePackageConfigHelpers)

install(EXPORT srukfTargets
  FILE        srukfTargets.cmake
  NAMESPACE   srukf::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/srukf
)

configure_package_config_file(
  cmake/srukfConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/srukfConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/srukf
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/srukfConfigVersion.cmake
  VERSION       ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/srukfConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/srukfConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/srukf
)
