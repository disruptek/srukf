"""High-level Pythonic interface to the Square-Root Unscented Kalman Filter.

Example
-------
>>> import numpy as np
>>> from srukf import UnscentedKalmanFilter
>>>
>>> ukf = UnscentedKalmanFilter(
...     state_dim=2, meas_dim=1,
...     process_noise_sqrt=np.diag([0.1, 0.01]),
...     meas_noise_sqrt=np.array([[0.5]]),
... )
>>> ukf.reset(1.0)
>>> ukf.x = np.array([0.0, 1.0])
>>> ukf.predict(lambda x: np.array([x[0] + 0.1 * x[1], x[1]]))
>>> ukf.update(np.array([0.12]), lambda x: np.array([x[0]]))
>>> print(ukf.x)
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER
from typing import Any, Callable, Optional

import numpy as np

from ._bindings import (
    MeasModelFunc,
    ProcessModelFunc,
    SrukfFilter,
    SrukfMat,
    SRUKF_RETURN_MATH_ERROR,
    SRUKF_RETURN_OK,
    SRUKF_RETURN_PARAMETER_ERROR,
    lib,
    srukf_value,
)
from .utils import (
    numpy_to_srukf_mat,
    srukf_mat_to_numpy,
    validate_square,
    validate_vector,
)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class SrukfError(RuntimeError):
    """Base exception for SR-UKF errors."""


class SrukfParameterError(SrukfError, ValueError):
    """Raised when the C library returns ``SRUKF_RETURN_PARAMETER_ERROR``."""


class SrukfMathError(SrukfError):
    """Raised when the C library returns ``SRUKF_RETURN_MATH_ERROR``."""


_RETURN_EXCEPTIONS = {
    SRUKF_RETURN_PARAMETER_ERROR: SrukfParameterError,
    SRUKF_RETURN_MATH_ERROR: SrukfMathError,
}


def _check(rc: int, context: str = "") -> None:
    """Translate a C return code into a Python exception."""
    if rc == SRUKF_RETURN_OK:
        return
    exc_cls = _RETURN_EXCEPTIONS.get(rc, SrukfError)
    msg = f"SR-UKF error (code {rc})"
    if context:
        msg = f"{context}: {msg}"
    raise exc_cls(msg)


# ---------------------------------------------------------------------------
# Callback wrappers
# ---------------------------------------------------------------------------


def _make_process_callback(
    py_func: Callable[..., np.ndarray],
    state_dim: int,
    kwargs: dict[str, Any],
) -> ProcessModelFunc:
    """Wrap a Python ``f(x, **kw) -> x_next`` into a C callback."""

    def _cb(x_in_p: POINTER(SrukfMat), x_out_p: POINTER(SrukfMat), _user: Any) -> None:
        # Read input state
        x_in = srukf_mat_to_numpy(x_in_p, copy=True).ravel()
        # Call Python function
        x_out = np.asarray(py_func(x_in, **kwargs), dtype=np.float64).ravel()
        # Write output
        mat_out = x_out_p.contents
        for i in range(state_dim):
            mat_out.data[i] = x_out[i]

    return ProcessModelFunc(_cb)


def _make_meas_callback(
    py_func: Callable[..., np.ndarray],
    meas_dim: int,
    kwargs: dict[str, Any],
) -> MeasModelFunc:
    """Wrap a Python ``h(x, **kw) -> z`` into a C callback."""

    def _cb(x_in_p: POINTER(SrukfMat), z_out_p: POINTER(SrukfMat), _user: Any) -> None:
        x_in = srukf_mat_to_numpy(x_in_p, copy=True).ravel()
        z_out = np.asarray(py_func(x_in, **kwargs), dtype=np.float64).ravel()
        mat_out = z_out_p.contents
        for i in range(meas_dim):
            mat_out.data[i] = z_out[i]

    return MeasModelFunc(_cb)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class UnscentedKalmanFilter:
    """Square-Root Unscented Kalman Filter.

    Parameters
    ----------
    state_dim : int
        Dimension of the state vector (*N*).
    meas_dim : int
        Dimension of the measurement vector (*M*).
    process_noise_sqrt : array_like
        Square-root of the process noise covariance (*N* x *N*).
        This is the Cholesky factor ``Q_sqrt`` such that ``Q = Q_sqrt @ Q_sqrt.T``.
    meas_noise_sqrt : array_like
        Square-root of the measurement noise covariance (*M* x *M*).
    alpha : float, optional
        Sigma-point spread parameter (default ``1e-3``).
    beta : float, optional
        Prior distribution parameter; 2.0 is optimal for Gaussian (default ``2.0``).
    kappa : float, optional
        Secondary scaling parameter (default ``0.0``).

    Raises
    ------
    SrukfParameterError
        If dimensions are invalid or noise matrices have wrong shape.
    MemoryError
        If the C library fails to allocate.

    Examples
    --------
    >>> ukf = UnscentedKalmanFilter(
    ...     state_dim=2, meas_dim=1,
    ...     process_noise_sqrt=np.diag([0.1, 0.01]),
    ...     meas_noise_sqrt=np.array([[0.5]]),
    ... )
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        process_noise_sqrt: np.ndarray,
        meas_noise_sqrt: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        if state_dim <= 0 or meas_dim <= 0:
            raise SrukfParameterError(
                f"Dimensions must be positive: state_dim={state_dim}, meas_dim={meas_dim}"
            )

        process_noise_sqrt = validate_square(
            process_noise_sqrt, "process_noise_sqrt"
        )
        meas_noise_sqrt = validate_square(meas_noise_sqrt, "meas_noise_sqrt")

        if process_noise_sqrt.shape[0] != state_dim:
            raise SrukfParameterError(
                f"process_noise_sqrt shape {process_noise_sqrt.shape} "
                f"does not match state_dim={state_dim}"
            )
        if meas_noise_sqrt.shape[0] != meas_dim:
            raise SrukfParameterError(
                f"meas_noise_sqrt shape {meas_noise_sqrt.shape} "
                f"does not match meas_dim={meas_dim}"
            )

        self._state_dim = state_dim
        self._meas_dim = meas_dim

        # Convert noise matrices to C
        q_mat = numpy_to_srukf_mat(process_noise_sqrt)
        r_mat = numpy_to_srukf_mat(meas_noise_sqrt)

        try:
            # Create filter from noise matrices
            self._ptr = lib.srukf_create_from_noise(q_mat, r_mat)
            if not self._ptr:
                raise MemoryError("srukf_create_from_noise returned NULL")
        finally:
            lib.srukf_mat_free(q_mat)
            lib.srukf_mat_free(r_mat)

        # Set scaling parameters
        _check(
            lib.srukf_set_scale(self._ptr, alpha, beta, kappa),
            "set_scale",
        )

        # Keep references to prevent GC of live callbacks
        self._live_callbacks: list = []

    # -- Destructor ---------------------------------------------------------

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            lib.srukf_free(ptr)
            self._ptr = None

    # -- Properties ---------------------------------------------------------

    @property
    def state_dim(self) -> int:
        """State vector dimension *N*."""
        return self._state_dim

    @property
    def meas_dim(self) -> int:
        """Measurement vector dimension *M*."""
        return self._meas_dim

    @property
    def x(self) -> np.ndarray:
        """Current state estimate as a 1-D numpy array of length *N*.

        Examples
        --------
        >>> ukf.x
        array([0., 0.])
        >>> ukf.x = np.array([1.0, 2.0])
        """
        n = self._state_dim
        out_mat = lib.srukf_mat_alloc(n, 1, 1)
        if not out_mat:
            raise MemoryError("Failed to allocate state output buffer")
        try:
            _check(lib.srukf_get_state(self._ptr, out_mat), "get_state")
            return srukf_mat_to_numpy(out_mat, copy=True).ravel()
        finally:
            lib.srukf_mat_free(out_mat)

    @x.setter
    def x(self, value: np.ndarray) -> None:
        value = validate_vector(value, self._state_dim, "state")
        mat = numpy_to_srukf_mat(value.reshape(-1, 1))
        try:
            _check(lib.srukf_set_state(self._ptr, mat), "set_state")
        finally:
            lib.srukf_mat_free(mat)

    @property
    def S(self) -> np.ndarray:
        """Square-root covariance matrix (*N* x *N*).

        This is the lower-triangular Cholesky factor such that
        ``P = S @ S.T``.
        """
        n = self._state_dim
        out_mat = lib.srukf_mat_alloc(n, n, 1)
        if not out_mat:
            raise MemoryError("Failed to allocate covariance output buffer")
        try:
            _check(lib.srukf_get_sqrt_cov(self._ptr, out_mat), "get_sqrt_cov")
            return srukf_mat_to_numpy(out_mat, copy=True)
        finally:
            lib.srukf_mat_free(out_mat)

    @S.setter
    def S(self, value: np.ndarray) -> None:
        value = validate_square(value, "S")
        if value.shape[0] != self._state_dim:
            raise SrukfParameterError(
                f"S shape {value.shape} does not match state_dim={self._state_dim}"
            )
        mat = numpy_to_srukf_mat(value)
        try:
            _check(lib.srukf_set_sqrt_cov(self._ptr, mat), "set_sqrt_cov")
        finally:
            lib.srukf_mat_free(mat)

    @property
    def P(self) -> np.ndarray:
        """Full covariance matrix ``S @ S.T`` (*N* x *N*).

        This is a convenience property; the library stores only *S*.
        """
        s = self.S
        return s @ s.T

    # -- Methods ------------------------------------------------------------

    def predict(
        self,
        process_model: Callable[..., np.ndarray],
        **kwargs: Any,
    ) -> "UnscentedKalmanFilter":
        """Run the prediction step.

        Propagates the state estimate and covariance forward through
        the process model.

        Parameters
        ----------
        process_model : callable
            Function ``f(x, **kwargs) -> x_next`` where *x* and *x_next*
            are 1-D numpy arrays of length *N*.
        **kwargs
            Extra keyword arguments forwarded to *process_model*.

        Returns
        -------
        UnscentedKalmanFilter
            *self*, for method chaining.

        Raises
        ------
        SrukfMathError
            If the process model produces NaN/Inf or the QR step fails.

        Examples
        --------
        >>> def f(x, dt=0.1):
        ...     return np.array([x[0] + dt * x[1], x[1]])
        >>> ukf.predict(f, dt=0.05)
        """
        cb = _make_process_callback(process_model, self._state_dim, kwargs)
        # prevent GC of the callback during the C call
        self._live_callbacks.append(cb)
        try:
            _check(lib.srukf_predict(self._ptr, cb, None), "predict")
        finally:
            self._live_callbacks.pop()
        return self

    def update(
        self,
        measurement: np.ndarray,
        meas_model: Callable[..., np.ndarray],
        **kwargs: Any,
    ) -> "UnscentedKalmanFilter":
        """Run the correction (update) step.

        Incorporates a measurement to refine the state estimate.

        Parameters
        ----------
        measurement : array_like
            Measurement vector of length *M*.
        meas_model : callable
            Function ``h(x, **kwargs) -> z_predicted`` where *x* is a
            1-D array of length *N* and *z_predicted* has length *M*.
        **kwargs
            Extra keyword arguments forwarded to *meas_model*.

        Returns
        -------
        UnscentedKalmanFilter
            *self*, for method chaining.

        Raises
        ------
        SrukfMathError
            If the measurement model produces NaN/Inf or the Cholesky
            downdate fails.

        Examples
        --------
        >>> ukf.update(np.array([1.5]), lambda x: np.array([x[0]]))
        """
        measurement = validate_vector(measurement, self._meas_dim, "measurement")
        z_mat = numpy_to_srukf_mat(measurement.reshape(-1, 1))
        cb = _make_meas_callback(meas_model, self._meas_dim, kwargs)
        self._live_callbacks.append(cb)
        try:
            _check(lib.srukf_correct(self._ptr, z_mat, cb, None), "correct")
        finally:
            self._live_callbacks.pop()
            lib.srukf_mat_free(z_mat)
        return self

    def reset(self, init_std: float = 1.0) -> "UnscentedKalmanFilter":
        """Reset state to zero and covariance to ``init_std * I``.

        Parameters
        ----------
        init_std : float
            Initial standard deviation (must be > 0).

        Returns
        -------
        UnscentedKalmanFilter
            *self*, for method chaining.

        Examples
        --------
        >>> ukf.reset(0.5)
        """
        _check(lib.srukf_reset(self._ptr, init_std), "reset")
        return self

    def set_noise(
        self,
        process_noise_sqrt: np.ndarray,
        meas_noise_sqrt: np.ndarray,
    ) -> "UnscentedKalmanFilter":
        """Update the noise covariance matrices.

        Parameters
        ----------
        process_noise_sqrt : array_like
            New process noise sqrt-covariance (*N* x *N*).
        meas_noise_sqrt : array_like
            New measurement noise sqrt-covariance (*M* x *M*).

        Returns
        -------
        UnscentedKalmanFilter
            *self*, for method chaining.
        """
        process_noise_sqrt = validate_square(process_noise_sqrt, "process_noise_sqrt")
        meas_noise_sqrt = validate_square(meas_noise_sqrt, "meas_noise_sqrt")

        q_mat = numpy_to_srukf_mat(process_noise_sqrt)
        r_mat = numpy_to_srukf_mat(meas_noise_sqrt)
        try:
            _check(lib.srukf_set_noise(self._ptr, q_mat, r_mat), "set_noise")
        finally:
            lib.srukf_mat_free(q_mat)
            lib.srukf_mat_free(r_mat)
        return self

    def set_scale(
        self,
        alpha: float,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> "UnscentedKalmanFilter":
        """Update the UKF scaling parameters.

        Parameters
        ----------
        alpha : float
            Sigma-point spread (must be > 0).
        beta : float
            Prior distribution parameter (2.0 for Gaussian).
        kappa : float
            Secondary scaling parameter.

        Returns
        -------
        UnscentedKalmanFilter
            *self*, for method chaining.
        """
        _check(lib.srukf_set_scale(self._ptr, alpha, beta, kappa), "set_scale")
        return self

    # -- Representation -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UnscentedKalmanFilter(state_dim={self._state_dim}, "
            f"meas_dim={self._meas_dim})"
        )
