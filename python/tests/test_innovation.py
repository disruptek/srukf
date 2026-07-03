"""Tests for library version reporting and innovation access."""

import numpy as np
import pytest

from srukf import (
    SrukfParameterError,
    UnscentedKalmanFilter,
    lib_version,
)


class TestVersion:
    def test_lib_version_string(self):
        v = lib_version()
        assert isinstance(v, str)
        assert v.count(".") == 2
        major = int(v.split(".")[0])
        assert major >= 1


class TestInnovation:
    """Analytic scalar case, mirroring the C test in 45_accessors.c:
    N = M = 1, identity h, x = 0, S = 1, R = 1, alpha = 1 (lambda = 0).
    With z = 2: innovation = 2, Syy = sqrt(2), NIS = 2.
    """

    @pytest.fixture
    def ukf1(self):
        ukf = UnscentedKalmanFilter(
            state_dim=1,
            meas_dim=1,
            process_noise_sqrt=np.array([[0.1]]),
            meas_noise_sqrt=np.array([[1.0]]),
            alpha=1.0,
            beta=2.0,
            kappa=0.0,
        )
        ukf.reset(1.0)
        return ukf

    def test_innovation_before_correct_raises(self, ukf1):
        with pytest.raises(SrukfParameterError):
            _ = ukf1.innovation
        with pytest.raises(SrukfParameterError):
            _ = ukf1.innovation_sqrt_cov
        with pytest.raises(SrukfParameterError):
            _ = ukf1.nis

    def test_innovation_values(self, ukf1):
        ukf1.update(np.array([2.0]), lambda x: x.copy())

        np.testing.assert_allclose(ukf1.innovation, [2.0], atol=1e-9)
        np.testing.assert_allclose(
            ukf1.innovation_sqrt_cov, [[np.sqrt(2.0)]], atol=1e-9
        )
        assert ukf1.nis == pytest.approx(2.0, abs=1e-9)

    def test_nis_gating_workflow(self, ukf1):
        """NIS should grow with the size of the measurement surprise."""
        ukf1.update(np.array([0.1]), lambda x: x.copy())
        small = ukf1.nis
        ukf1.update(np.array([50.0]), lambda x: x.copy())
        large = ukf1.nis
        assert large > small
