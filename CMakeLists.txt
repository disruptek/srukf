cmake_minimum_required(VERSION 3.15)

# The version is owned by srukf.h (see CONTRIBUTING.md); parse it.
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/srukf.h _srukf_version_line
     REGEX "^#define SRUKF_VERSION \"")
if(NOT _srukf_version_line MATCHES "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"")
  message(FATAL_ERROR "Cannot parse SRUKF_VERSION from srukf.h")
endif()

project(srukf
  VERSION ${CMAKE_MATCH_1}
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

  # Preferred link target for public API tests and examples
  if(SRUKF_BUILD_SHARED)
    set(SRUKF_LINK_TARGET srukf_shared)
  else()
    set(SRUKF_LINK_TARGET srukf_static)
  endif()

  # Tests are discovered and classified by convention (CONTRIBUTING.md):
  # a test that does `#include "srukf.c"` compiles the library source
  # directly (internal) and only needs the include path and deps; any
  # other tests/*.c links the library (public). Per-test flags such as
  # SRUKF_SINGLE are defined by the test source itself. Adding a test
  # therefore requires no build-system edits.
  #
  # Tests use assert() for validation.  CMake's Release/RelWithDebInfo
  # build types add -DNDEBUG which disables assert().  We must ensure
  # NDEBUG is never defined for test targets, so -UNDEBUG follows any
  # -DNDEBUG that CMake injects via CMAKE_C_FLAGS_<CONFIG>.
  file(GLOB SRUKF_TEST_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c)
  foreach(test_src IN LISTS SRUKF_TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)
    file(STRINGS ${test_src} _includes_impl REGEX "#include \"srukf\\.c\"")

    add_executable(test_${test_name} ${test_src})
    target_compile_options(test_${test_name} PRIVATE
      ${SRUKF_PRIVATE_COMPILE_OPTIONS} -UNDEBUG
    )
    set_target_properties(test_${test_name} PROPERTIES
      C_STANDARD          99
      C_STANDARD_REQUIRED ON
      C_EXTENSIONS        OFF
    )
    if(_includes_impl)
      target_include_directories(test_${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
      )
      target_compile_definitions(test_${test_name} PRIVATE
        ${SRUKF_PRIVATE_COMPILE_DEFINITIONS}
      )
      target_link_libraries(test_${test_name} PRIVATE ${SRUKF_LINK_LIBS})
    else()
      target_link_libraries(test_${test_name} PRIVATE ${SRUKF_LINK_TARGET})
    endif()
    add_test(NAME ${test_name} COMMAND test_${test_name})
  endforeach()

  # C++ linkage test: verifies the extern "C" guards in the public
  # header. Only built when a C++ compiler is available.
  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    add_executable(test_90_cpp_linkage tests/90_cpp_linkage.cpp)
    target_link_libraries(test_90_cpp_linkage PRIVATE ${SRUKF_LINK_TARGET})
    target_compile_options(test_90_cpp_linkage PRIVATE -Wall -Wextra -UNDEBUG)
    set_target_properties(test_90_cpp_linkage PROPERTIES
      CXX_STANDARD          11
      CXX_STANDARD_REQUIRED ON
    )
    add_test(NAME 90_cpp_linkage COMMAND test_90_cpp_linkage)
  endif()
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
