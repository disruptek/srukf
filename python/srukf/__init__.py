"""Square-Root Unscented Kalman Filter — Python bindings.

Quick start::

    import numpy as np
    from srukf import UnscentedKalmanFilter

    ukf = UnscentedKalmanFilter(
        state_dim=2, meas_dim=1,
        process_noise_sqrt=np.diag([0.1, 0.01]),
        meas_noise_sqrt=np.array([[0.5]]),
    )
    ukf.predict(lambda x: np.array([x[0] + 0.1 * x[1], x[1]]))
    ukf.update(np.array([0.12]), lambda x: np.array([x[0]]))
"""

from .core import (
    SrukfError,
    SrukfMathError,
    SrukfParameterError,
    UnscentedKalmanFilter,
)
from .version import __version__, __version_info__

__all__ = [
    "UnscentedKalmanFilter",
    "SrukfError",
    "SrukfParameterError",
    "SrukfMathError",
    "__version__",
    "__version_info__",
]
