"""Basic functionality tests for the SR-UKF Python bindings."""

import numpy as np
import pytest

from srukf import (
    SrukfError,
    SrukfMathError,
    SrukfParameterError,
    UnscentedKalmanFilter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ukf2():
    """A 2-state, 1-measurement filter with sensible defaults."""
    return UnscentedKalmanFilter(
        state_dim=2,
        meas_dim=1,
        process_noise_sqrt=np.diag([0.1, 0.1]),
        meas_noise_sqrt=np.array([[0.2]]),
    )


# ---------------------------------------------------------------------------
# Creation / destruction
# ---------------------------------------------------------------------------


class TestCreation:
    def test_basic_creation(self, ukf2):
        assert ukf2.state_dim == 2
        assert ukf2.meas_dim == 1

    def test_repr(self, ukf2):
        r = repr(ukf2)
        assert "state_dim=2" in r
        assert "meas_dim=1" in r

    def test_larger_dimensions(self):
        ukf = UnscentedKalmanFilter(
            state_dim=7,
            meas_dim=5,
            process_noise_sqrt=0.1 * np.eye(7),
            meas_noise_sqrt=0.2 * np.eye(5),
        )
        assert ukf.state_dim == 7
        assert ukf.meas_dim == 5

    def test_invalid_dimensions(self):
        with pytest.raises(SrukfParameterError):
            UnscentedKalmanFilter(
                state_dim=0,
                meas_dim=1,
                process_noise_sqrt=np.eye(1),
                meas_noise_sqrt=np.eye(1),
            )

    def test_mismatched_noise_shape(self):
        with pytest.raises(SrukfParameterError):
            UnscentedKalmanFilter(
                state_dim=2,
                meas_dim=1,
                process_noise_sqrt=np.eye(3),  # wrong: should be 2x2
                meas_noise_sqrt=np.eye(1),
            )

    def test_non_square_noise(self):
        with pytest.raises(ValueError):
            UnscentedKalmanFilter(
                state_dim=2,
                meas_dim=1,
                process_noise_sqrt=np.ones((2, 3)),
                meas_noise_sqrt=np.eye(1),
            )


# ---------------------------------------------------------------------------
# State get / set
# ---------------------------------------------------------------------------


class TestState:
    def test_initial_state_near_zero(self, ukf2):
        x = ukf2.x
        assert x.shape == (2,)
        # Default state from srukf_create_from_noise is zero
        np.testing.assert_allclose(x, [0.0, 0.0], atol=1e-10)

    def test_set_state(self, ukf2):
        ukf2.x = np.array([1.5, -0.3])
        np.testing.assert_allclose(ukf2.x, [1.5, -0.3], atol=1e-12)

    def test_set_state_wrong_length(self, ukf2):
        with pytest.raises(ValueError):
            ukf2.x = np.array([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Covariance get / set
# ---------------------------------------------------------------------------


class TestCovariance:
    def test_initial_S_shape(self, ukf2):
        S = ukf2.S
        assert S.shape == (2, 2)

    def test_set_S(self, ukf2):
        new_S = np.array([[0.5, 0.0], [0.1, 0.4]])
        ukf2.S = new_S
        np.testing.assert_allclose(ukf2.S, new_S, atol=1e-12)

    def test_P_equals_S_ST(self, ukf2):
        S = ukf2.S
        P = ukf2.P
        np.testing.assert_allclose(P, S @ S.T, atol=1e-12)

    def test_set_S_wrong_shape(self, ukf2):
        with pytest.raises((ValueError, SrukfParameterError)):
            ukf2.S = np.eye(3)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_zeros_state(self, ukf2):
        ukf2.x = np.array([5.0, 3.0])
        ukf2.reset(1.0)
        np.testing.assert_allclose(ukf2.x, [0.0, 0.0], atol=1e-12)

    def test_reset_sets_identity_covariance(self, ukf2):
        ukf2.reset(2.5)
        S = ukf2.S
        np.testing.assert_allclose(S, 2.5 * np.eye(2), atol=1e-12)

    def test_reset_invalid_std(self, ukf2):
        with pytest.raises(SrukfError):
            ukf2.reset(-1.0)

    def test_reset_returns_self(self, ukf2):
        result = ukf2.reset(1.0)
        assert result is ukf2


# ---------------------------------------------------------------------------
# Predict / Update
# ---------------------------------------------------------------------------


class TestPredictUpdate:
    def test_identity_predict_preserves_state(self, ukf2):
        ukf2.x = np.array([1.0, 2.0])
        ukf2.predict(lambda x: x.copy())
        # State should remain close to [1, 2]
        np.testing.assert_allclose(ukf2.x, [1.0, 2.0], atol=1e-6)

    def test_predict_increases_uncertainty(self, ukf2):
        ukf2.reset(0.1)
        trace_before = np.trace(ukf2.P)
        ukf2.predict(lambda x: x.copy())
        trace_after = np.trace(ukf2.P)
        assert trace_after > trace_before

    def test_update_moves_state_toward_measurement(self, ukf2):
        ukf2.reset(1.0)
        ukf2.predict(lambda x: x.copy())
        ukf2.update(np.array([5.0]), lambda x: np.array([x[0]]))
        # State should move toward 5.0
        assert ukf2.x[0] > 0.0

    def test_update_decreases_uncertainty(self, ukf2):
        ukf2.reset(1.0)
        ukf2.predict(lambda x: x.copy())
        trace_before = np.trace(ukf2.P)
        ukf2.update(np.array([0.0]), lambda x: np.array([x[0]]))
        trace_after = np.trace(ukf2.P)
        assert trace_after < trace_before

    def test_method_chaining(self, ukf2):
        result = (
            ukf2.reset(1.0)
            .predict(lambda x: x.copy())
            .update(np.array([1.0]), lambda x: np.array([x[0]]))
        )
        assert result is ukf2

    def test_predict_with_kwargs(self, ukf2):
        def process(x, dt=0.1):
            return np.array([x[0] + dt * x[1], x[1]])

        ukf2.x = np.array([0.0, 1.0])
        ukf2.predict(process, dt=0.5)
        # Position should advance by ~0.5
        assert ukf2.x[0] > 0.3

    def test_update_wrong_measurement_dim(self, ukf2):
        ukf2.predict(lambda x: x.copy())
        with pytest.raises(ValueError):
            ukf2.update(np.array([1.0, 2.0]), lambda x: np.array([x[0]]))


# ---------------------------------------------------------------------------
# Noise / scale updates
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_set_noise(self, ukf2):
        ukf2.set_noise(
            process_noise_sqrt=0.5 * np.eye(2),
            meas_noise_sqrt=np.array([[1.0]]),
        )
        # Should not raise; verify filter still works
        ukf2.predict(lambda x: x.copy())

    def test_set_scale(self, ukf2):
        ukf2.set_scale(alpha=0.5, beta=2.0, kappa=1.0)
        ukf2.predict(lambda x: x.copy())

    def test_set_scale_invalid_alpha(self, ukf2):
        with pytest.raises(SrukfError):
            ukf2.set_scale(alpha=-1.0)
