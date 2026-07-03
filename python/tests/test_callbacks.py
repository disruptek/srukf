"""Callback error-handling tests for the SR-UKF Python bindings.

A Python exception raised inside a process/measurement model must not be
swallowed: ctypes cannot propagate exceptions through the C boundary, so
without explicit handling the C library would read a stale output buffer
(finite garbage) and silently corrupt the estimate. The bindings must
surface the exception to the caller and leave the filter state unchanged.
"""

import numpy as np
import pytest

from srukf import UnscentedKalmanFilter


@pytest.fixture
def ukf2():
    ukf = UnscentedKalmanFilter(
        state_dim=2,
        meas_dim=1,
        process_noise_sqrt=np.diag([0.1, 0.1]),
        meas_noise_sqrt=np.array([[0.2]]),
    )
    ukf.reset(1.0)
    ukf.x = np.array([1.0, -2.0])
    return ukf


class TestCallbackExceptions:
    def test_predict_callback_exception_raises(self, ukf2):
        def bad_f(x):
            raise ValueError("boom")

        with pytest.raises(Exception) as excinfo:
            ukf2.predict(bad_f)
        # the original exception must be visible, either directly or as
        # the __cause__ of the bindings' error
        exc = excinfo.value
        assert isinstance(exc, ValueError) or isinstance(
            exc.__cause__, ValueError
        )

    def test_predict_callback_exception_preserves_state(self, ukf2):
        x_before = ukf2.x.copy()
        S_before = ukf2.S.copy()

        def bad_f(x):
            raise ValueError("boom")

        with pytest.raises(Exception):
            ukf2.predict(bad_f)

        np.testing.assert_allclose(ukf2.x, x_before)
        np.testing.assert_allclose(ukf2.S, S_before)

    def test_update_callback_exception_raises(self, ukf2):
        def bad_h(x):
            raise RuntimeError("sensor model exploded")

        with pytest.raises(Exception) as excinfo:
            ukf2.update(np.array([0.5]), bad_h)
        exc = excinfo.value
        assert isinstance(exc, RuntimeError) or isinstance(
            exc.__cause__, RuntimeError
        )

    def test_update_callback_exception_preserves_state(self, ukf2):
        x_before = ukf2.x.copy()
        S_before = ukf2.S.copy()

        def bad_h(x):
            raise RuntimeError("boom")

        with pytest.raises(Exception):
            ukf2.update(np.array([0.5]), bad_h)

        np.testing.assert_allclose(ukf2.x, x_before)
        np.testing.assert_allclose(ukf2.S, S_before)

    def test_filter_usable_after_callback_exception(self, ukf2):
        def bad_f(x):
            raise ValueError("boom")

        with pytest.raises(Exception):
            ukf2.predict(bad_f)

        # a good model afterwards must work normally
        ukf2.predict(lambda x: x.copy())
        ukf2.update(np.array([0.9]), lambda x: np.array([x[0]]))
        assert np.all(np.isfinite(ukf2.x))
        assert np.all(np.isfinite(ukf2.S))

    def test_wrong_output_shape_raises(self, ukf2):
        # a model returning the wrong number of elements must error,
        # not silently write partial output
        with pytest.raises(Exception):
            ukf2.predict(lambda x: np.array([1.0]))
