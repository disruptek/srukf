"""Matrix conversion utilities for the SR-UKF Python bindings.

Provides zero-copy (where possible) conversions between numpy arrays
and the C ``srukf_mat`` structure used by the library.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER

import numpy as np

from ._bindings import SrukfMat, lib, srukf_value

# ---------------------------------------------------------------------------
# numpy <-> SrukfMat conversions
# ---------------------------------------------------------------------------


def numpy_to_srukf_mat(arr: np.ndarray) -> POINTER(SrukfMat):
    """Convert a numpy array to a C-allocated ``srukf_mat``.

    The data is **copied** into a freshly allocated ``srukf_mat`` because
    the C library may reallocate or free the matrix independently of
    Python's garbage collector.

    Parameters
    ----------
    arr : numpy.ndarray
        1-D or 2-D array of float64 values.

    Returns
    -------
    ctypes.POINTER(SrukfMat)
        Pointer to a heap-allocated ``srukf_mat`` with copied data.
        The caller is responsible for calling ``lib.srukf_mat_free()``
        when done.

    Raises
    ------
    ValueError
        If *arr* is not 1-D or 2-D, or not convertible to float64.

    Examples
    --------
    >>> import numpy as np
    >>> mat_p = numpy_to_srukf_mat(np.eye(3))
    >>> lib.srukf_mat_free(mat_p)
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got {arr.ndim}-D")

    rows, cols = arr.shape
    mat_p = lib.srukf_mat_alloc(rows, cols, 1)
    if not mat_p:
        raise MemoryError("srukf_mat_alloc returned NULL")

    # Copy data in column-major order (Fortran order)
    col_major = np.asfortranarray(arr)
    ctypes.memmove(
        mat_p.contents.data,
        col_major.ctypes.data,
        rows * cols * ctypes.sizeof(srukf_value),
    )
    return mat_p


def srukf_mat_to_numpy(mat_p: POINTER(SrukfMat), copy: bool = True) -> np.ndarray:
    """Convert a C ``srukf_mat`` pointer to a numpy array.

    Parameters
    ----------
    mat_p : ctypes.POINTER(SrukfMat)
        Pointer to a valid ``srukf_mat``.
    copy : bool, optional
        If *True* (default), return an independent copy.  If *False*,
        return a **view** into the C memory — the array becomes invalid
        once the C matrix is freed.

    Returns
    -------
    numpy.ndarray
        2-D array of shape ``(n_rows, n_cols)`` with dtype float64.

    Raises
    ------
    ValueError
        If *mat_p* is NULL or has no data.

    Examples
    --------
    >>> mat_p = numpy_to_srukf_mat(np.array([[1.0, 2.0], [3.0, 4.0]]))
    >>> a = srukf_mat_to_numpy(mat_p)
    >>> a
    array([[1., 2.],
           [3., 4.]])
    >>> lib.srukf_mat_free(mat_p)
    """
    if not mat_p:
        raise ValueError("NULL srukf_mat pointer")

    mat = mat_p.contents
    rows = mat.n_rows
    cols = mat.n_cols

    if not mat.data:
        raise ValueError("srukf_mat has no data (SRUKF_TYPE_NO_DATA)")

    # Build a numpy array that shares the C buffer (column-major)
    buf = (srukf_value * (rows * cols)).from_address(
        ctypes.addressof(mat.data.contents)
    )
    view = np.frombuffer(buf, dtype=np.float64).reshape((rows, cols), order="F")

    return view.copy() if copy else view


def srukf_mat_read_into(mat_p: POINTER(SrukfMat), out: np.ndarray) -> np.ndarray:
    """Copy data from a C ``srukf_mat`` into an existing numpy array.

    This avoids allocation when reading state/covariance in a hot loop.

    Parameters
    ----------
    mat_p : ctypes.POINTER(SrukfMat)
        Source matrix.
    out : numpy.ndarray
        Destination array; must have compatible shape and float64 dtype.

    Returns
    -------
    numpy.ndarray
        *out*, for convenience.
    """
    mat = mat_p.contents
    rows, cols = mat.n_rows, mat.n_cols
    nbytes = rows * cols * ctypes.sizeof(srukf_value)
    # Read column-major data into a temporary, then reshape
    buf = (ctypes.c_char * nbytes)()
    ctypes.memmove(buf, mat.data, nbytes)
    tmp = np.frombuffer(buf, dtype=np.float64).reshape((rows, cols), order="F")
    np.copyto(out, tmp)
    return out


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_square(arr: np.ndarray, name: str = "matrix") -> np.ndarray:
    """Ensure *arr* is a square 2-D float64 array.

    Parameters
    ----------
    arr : array_like
        Input to validate.
    name : str
        Human-readable name for error messages.

    Returns
    -------
    numpy.ndarray
        Validated array (may be a new object if dtype conversion occurred).

    Raises
    ------
    ValueError
        If the array is not 2-D or not square.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a square 2-D array, got shape {arr.shape}"
        )
    return arr


def validate_vector(arr: np.ndarray, length: int, name: str = "vector") -> np.ndarray:
    """Ensure *arr* is a 1-D float64 array of the given length.

    Parameters
    ----------
    arr : array_like
        Input to validate.
    length : int
        Expected number of elements.
    name : str
        Human-readable name for error messages.

    Returns
    -------
    numpy.ndarray
        Validated 1-D array.

    Raises
    ------
    ValueError
        If shape does not match.
    """
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.shape[0] != length:
        raise ValueError(
            f"{name} must have {length} elements, got {arr.shape[0]}"
        )
    return arr
