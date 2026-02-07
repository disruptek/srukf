"""Low-level ctypes bindings for the SR-UKF C library.

This module handles:
- Library discovery and loading
- C type declarations (srukf_mat, srukf, enums)
- Function prototype declarations with proper argtypes/restype
- Callback wrapper types for process/measurement models

Users should prefer the high-level API in ``srukf.core`` instead of
using this module directly.
"""

import ctypes
import ctypes.util
import os
import platform
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_double,
    c_int,
    c_size_t,
    c_void_p,
)

# ---------------------------------------------------------------------------
# Scalar typedefs matching the C header
# ---------------------------------------------------------------------------

#: Scalar value type (double precision; single precision not exposed here).
srukf_value = c_double

#: Index type used for matrix dimensions.
srukf_index = c_size_t

# ---------------------------------------------------------------------------
# Return code enum
# ---------------------------------------------------------------------------

SRUKF_RETURN_OK = 0
SRUKF_RETURN_PARAMETER_ERROR = 1
SRUKF_RETURN_MATH_ERROR = 2

# ---------------------------------------------------------------------------
# Matrix type flags
# ---------------------------------------------------------------------------

SRUKF_TYPE_COL_MAJOR = 0x01
SRUKF_TYPE_NO_DATA = 0x02
SRUKF_TYPE_VECTOR = 0x04
SRUKF_TYPE_SQUARE = 0x08


# ---------------------------------------------------------------------------
# C structures
# ---------------------------------------------------------------------------

class SrukfMat(Structure):
    """ctypes mirror of the C ``srukf_mat`` structure.

    Fields
    ------
    n_cols : size_t
    n_rows : size_t
    inc_row : size_t
    inc_col : size_t
    data : POINTER(c_double)
    type : c_int
    """

    _fields_ = [
        ("n_cols", srukf_index),
        ("n_rows", srukf_index),
        ("inc_row", srukf_index),
        ("inc_col", srukf_index),
        ("data", POINTER(srukf_value)),
        ("type", c_int),
    ]


class SrukfWorkspace(Structure):
    """Opaque workspace — we never inspect its fields from Python."""

    _fields_: list = []


class SrukfFilter(Structure):
    """ctypes mirror of the C ``srukf`` structure.

    We declare only the fields we need to read; the workspace pointer
    is opaque.
    """

    _fields_ = [
        ("x", POINTER(SrukfMat)),
        ("S", POINTER(SrukfMat)),
        ("Qsqrt", POINTER(SrukfMat)),
        ("Rsqrt", POINTER(SrukfMat)),
        ("alpha", srukf_value),
        ("beta", srukf_value),
        ("kappa", srukf_value),
        ("lambda_", srukf_value),  # 'lambda' is a Python keyword
        ("wm", POINTER(srukf_value)),
        ("wc", POINTER(srukf_value)),
        ("ws", POINTER(SrukfWorkspace)),
    ]


# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

#: C callback: void (*)(const srukf_mat*, srukf_mat*, void*)
ProcessModelFunc = CFUNCTYPE(None, POINTER(SrukfMat), POINTER(SrukfMat), c_void_p)
MeasModelFunc = CFUNCTYPE(None, POINTER(SrukfMat), POINTER(SrukfMat), c_void_p)

#: Diagnostic callback: void (*)(const char*)
DiagFunc = CFUNCTYPE(None, ctypes.c_char_p)


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

def _lib_name() -> str:
    """Return the platform-specific shared library filename."""
    system = platform.system()
    if system == "Windows":
        return "srukf.dll"
    elif system == "Darwin":
        return "libsrukf.dylib"
    return "libsrukf.so"


def _find_library() -> str:
    """Locate ``libsrukf`` using several search strategies.

    Search order:
    1. Bundled alongside this Python file (``srukf/libsrukf.so``)
    2. Repository root (one level above ``python/``)
    3. ``SRUKF_LIB_PATH`` environment variable
    4. System library paths via ``ctypes.util.find_library``

    Returns
    -------
    str
        Absolute path to the shared library.

    Raises
    ------
    OSError
        If the library cannot be found anywhere.
    """
    name = _lib_name()

    # 1. Bundled next to this file
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(here, name)
    if os.path.isfile(bundled):
        return bundled

    # 2. Repository root (../.. from python/srukf/)
    repo_root = os.path.normpath(os.path.join(here, os.pardir, os.pardir))
    repo_lib = os.path.join(repo_root, name)
    if os.path.isfile(repo_lib):
        return repo_lib

    # 3. Environment variable
    env_path = os.environ.get("SRUKF_LIB_PATH")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        candidate = os.path.join(env_path, name)
        if os.path.isfile(candidate):
            return candidate

    # 4. System paths
    found = ctypes.util.find_library("srukf")
    if found:
        return found

    raise OSError(
        f"Cannot find {name}. Searched:\n"
        f"  1. {bundled}\n"
        f"  2. {repo_lib}\n"
        f"  3. SRUKF_LIB_PATH={env_path!r}\n"
        f"  4. System library paths\n"
        f"Build with 'make lib' in the repository root, or set SRUKF_LIB_PATH."
    )


def _load_library() -> ctypes.CDLL:
    """Load the shared library and declare all function prototypes."""
    path = _find_library()
    lib = ctypes.CDLL(path)
    _declare_functions(lib)
    return lib


# ---------------------------------------------------------------------------
# Function prototype declarations
# ---------------------------------------------------------------------------

def _declare_functions(lib: ctypes.CDLL) -> None:
    """Attach argtypes / restype to every public C function."""

    # -- Diagnostics --------------------------------------------------------
    lib.srukf_set_diag_callback.argtypes = [DiagFunc]
    lib.srukf_set_diag_callback.restype = None

    # -- Matrix allocation --------------------------------------------------
    lib.srukf_mat_alloc.argtypes = [srukf_index, srukf_index, c_int]
    lib.srukf_mat_alloc.restype = POINTER(SrukfMat)

    lib.srukf_mat_free.argtypes = [POINTER(SrukfMat)]
    lib.srukf_mat_free.restype = None

    # -- Filter lifecycle ---------------------------------------------------
    lib.srukf_create.argtypes = [c_int, c_int]
    lib.srukf_create.restype = POINTER(SrukfFilter)

    lib.srukf_create_from_noise.argtypes = [
        POINTER(SrukfMat),
        POINTER(SrukfMat),
    ]
    lib.srukf_create_from_noise.restype = POINTER(SrukfFilter)

    lib.srukf_free.argtypes = [POINTER(SrukfFilter)]
    lib.srukf_free.restype = None

    # -- Initialization -----------------------------------------------------
    lib.srukf_set_noise.argtypes = [
        POINTER(SrukfFilter),
        POINTER(SrukfMat),
        POINTER(SrukfMat),
    ]
    lib.srukf_set_noise.restype = c_int

    lib.srukf_set_scale.argtypes = [
        POINTER(SrukfFilter),
        srukf_value,
        srukf_value,
        srukf_value,
    ]
    lib.srukf_set_scale.restype = c_int

    # -- Accessors ----------------------------------------------------------
    lib.srukf_state_dim.argtypes = [POINTER(SrukfFilter)]
    lib.srukf_state_dim.restype = srukf_index

    lib.srukf_meas_dim.argtypes = [POINTER(SrukfFilter)]
    lib.srukf_meas_dim.restype = srukf_index

    lib.srukf_get_state.argtypes = [POINTER(SrukfFilter), POINTER(SrukfMat)]
    lib.srukf_get_state.restype = c_int

    lib.srukf_set_state.argtypes = [POINTER(SrukfFilter), POINTER(SrukfMat)]
    lib.srukf_set_state.restype = c_int

    lib.srukf_get_sqrt_cov.argtypes = [POINTER(SrukfFilter), POINTER(SrukfMat)]
    lib.srukf_get_sqrt_cov.restype = c_int

    lib.srukf_set_sqrt_cov.argtypes = [POINTER(SrukfFilter), POINTER(SrukfMat)]
    lib.srukf_set_sqrt_cov.restype = c_int

    lib.srukf_reset.argtypes = [POINTER(SrukfFilter), srukf_value]
    lib.srukf_reset.restype = c_int

    # -- Core operations ----------------------------------------------------
    lib.srukf_predict.argtypes = [
        POINTER(SrukfFilter),
        ProcessModelFunc,
        c_void_p,
    ]
    lib.srukf_predict.restype = c_int

    lib.srukf_correct.argtypes = [
        POINTER(SrukfFilter),
        POINTER(SrukfMat),
        MeasModelFunc,
        c_void_p,
    ]
    lib.srukf_correct.restype = c_int

    # -- Transactional operations -------------------------------------------
    lib.srukf_predict_to.argtypes = [
        POINTER(SrukfFilter),
        POINTER(SrukfMat),
        POINTER(SrukfMat),
        ProcessModelFunc,
        c_void_p,
    ]
    lib.srukf_predict_to.restype = c_int

    lib.srukf_correct_to.argtypes = [
        POINTER(SrukfFilter),
        POINTER(SrukfMat),
        POINTER(SrukfMat),
        POINTER(SrukfMat),
        MeasModelFunc,
        c_void_p,
    ]
    lib.srukf_correct_to.restype = c_int

    # -- Workspace ----------------------------------------------------------
    lib.srukf_alloc_workspace.argtypes = [POINTER(SrukfFilter)]
    lib.srukf_alloc_workspace.restype = c_int

    lib.srukf_free_workspace.argtypes = [POINTER(SrukfFilter)]
    lib.srukf_free_workspace.restype = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: The loaded C library handle.  Imported as ``from srukf._bindings import lib``.
lib = _load_library()
