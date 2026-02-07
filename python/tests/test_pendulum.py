"""Pendulum tracking test — port of the C 01_pendulum example.

This test verifies that the Python bindings produce numerically
consistent results when tracking a nonlinear pendulum.
"""

import math

import numpy as np
import pytest

from srukf import UnscentedKalmanFilter


# ---------------------------------------------------------------------------
# Pendulum physics (matching the C example)
# ---------------------------------------------------------------------------

G = 9.81  # gravitational acceleration (m/s^2)
L = 1.0   # pendulum length (m)
B = 0.1   # damping coefficient


def pendulum_derivatives(theta: float, omega: float):
    """Compute dtheta/dt and domega/dt."""
    dtheta = omega
    domega = -(G / L) * math.sin(theta) - B * omega
    return dtheta, domega


def pendulum_rk4(theta: float, omega: float, dt: float):
    """RK4 integration step for the pendulum ODE."""
    k1t, k1o = pendulum_derivatives(theta, omega)
    k2t, k2o = pendulum_derivatives(theta + k1t * dt / 2, omega + k1o * dt / 2)
    k3t, k3o = pendulum_derivatives(theta + k2t * dt / 2, omega + k2o * dt / 2)
    k4t, k4o = pendulum_derivatives(theta + k3t * dt, omega + k3o * dt)

    theta_next = theta + (dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t)
    omega_next = omega + (dt / 6) * (k1o + 2 * k2o + 2 * k3o + k4o)
    return theta_next, omega_next


# ---------------------------------------------------------------------------
# UKF models
# ---------------------------------------------------------------------------


def process_model(x, dt=0.01):
    """Process model: propagate pendulum state by dt."""
    theta_next, omega_next = pendulum_rk4(x[0], x[1], dt)
    return np.array([theta_next, omega_next])


def measurement_model(x):
    """Measurement model: observe angle only."""
    return np.array([x[0]])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPendulumTracking:
    """Run a full pendulum simulation and verify filter performance."""

    @pytest.fixture
    def simulation(self):
        """Run the pendulum simulation and return results."""
        rng = np.random.default_rng(12345)

        dt = 0.01
        duration = 5.0
        n_steps = int(duration / dt)
        meas_period = 1.0 / 20.0  # 20 Hz measurements
        meas_noise_std = 0.1
        process_noise_std = 0.01

        # Create filter
        ukf = UnscentedKalmanFilter(
            state_dim=2,
            meas_dim=1,
            process_noise_sqrt=process_noise_std * np.eye(2),
            meas_noise_sqrt=np.array([[meas_noise_std]]),
        )

        # Initial conditions
        initial_angle = math.pi / 4  # 45 degrees
        initial_velocity = 0.0

        ukf.x = np.array([initial_angle + 0.05, initial_velocity + 0.05])
        ukf.S = 0.2 * np.eye(2)

        # Simulate
        theta_true = initial_angle
        omega_true = initial_velocity
        time_since_meas = 0.0

        true_angles = []
        est_angles = []
        est_velocities = []
        errors = []

        for step in range(n_steps):
            # True dynamics (with small perturbation)
            theta_noisy = theta_true + rng.normal(0, process_noise_std * dt)
            omega_noisy = omega_true + rng.normal(0, process_noise_std * dt)
            theta_true, omega_true = pendulum_rk4(theta_noisy, omega_noisy, dt)

            true_angles.append(theta_true)

            # Predict
            ukf.predict(process_model, dt=dt)

            # Measurement at 20 Hz
            time_since_meas += dt
            if time_since_meas >= meas_period:
                z = np.array([theta_true + rng.normal(0, meas_noise_std)])
                ukf.update(z, measurement_model)
                time_since_meas = 0.0

            est_angles.append(ukf.x[0])
            est_velocities.append(ukf.x[1])
            errors.append(abs(ukf.x[0] - theta_true))

        return {
            "true_angles": np.array(true_angles),
            "est_angles": np.array(est_angles),
            "est_velocities": np.array(est_velocities),
            "errors": np.array(errors),
            "ukf": ukf,
        }

    def test_filter_converges(self, simulation):
        """After initial transient, error should be small."""
        # Skip first 50 steps (convergence period)
        errors = simulation["errors"][50:]
        mean_error = np.mean(errors)
        assert mean_error < 0.05, f"Mean tracking error too large: {mean_error:.4f}"

    def test_error_bounded(self, simulation):
        """Maximum error should stay bounded."""
        max_error = np.max(simulation["errors"][50:])
        assert max_error < 0.3, f"Max tracking error too large: {max_error:.4f}"

    def test_velocity_estimated(self, simulation):
        """Filter should produce reasonable velocity estimates."""
        # Velocity should not be stuck at zero
        vel_std = np.std(simulation["est_velocities"])
        assert vel_std > 0.1, f"Velocity estimates too static: std={vel_std:.4f}"

    def test_covariance_stays_spd(self, simulation):
        """Covariance should remain symmetric positive definite."""
        S = simulation["ukf"].S
        P = S @ S.T
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues > 0), f"Non-positive eigenvalues: {eigenvalues}"

    def test_uncertainty_reasonable(self, simulation):
        """Uncertainty should be small after convergence."""
        P = simulation["ukf"].P
        trace = np.trace(P)
        assert trace < 1.0, f"Covariance trace too large: {trace:.4f}"


class TestPendulumNumerics:
    """Verify numerical properties of the pendulum filter."""

    def test_rk4_energy_conservation(self):
        """RK4 should approximately conserve energy for undamped pendulum."""
        # Use undamped pendulum (B=0) for this test
        theta, omega = math.pi / 6, 0.0
        dt = 0.001

        def energy(th, om):
            return 0.5 * om**2 + (G / L) * (1 - math.cos(th))

        e0 = energy(theta, omega)
        for _ in range(10000):
            # Undamped: override B temporarily
            dtheta = omega
            domega = -(G / L) * math.sin(theta)

            k1t, k1o = omega, -(G / L) * math.sin(theta)
            k2t, k2o = omega + k1o * dt / 2, -(G / L) * math.sin(theta + k1t * dt / 2)
            k3t, k3o = omega + k2o * dt / 2, -(G / L) * math.sin(theta + k2t * dt / 2)
            k4t, k4o = omega + k3o * dt, -(G / L) * math.sin(theta + k3t * dt)

            theta += (dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t)
            omega += (dt / 6) * (k1o + 2 * k2o + 2 * k3o + k4o)

        e_final = energy(theta, omega)
        # Energy should be conserved to within ~1e-8 for RK4
        assert abs(e_final - e0) / e0 < 1e-6

    def test_multiple_predict_without_update(self):
        """Predicting without updates should increase uncertainty."""
        ukf = UnscentedKalmanFilter(
            state_dim=2,
            meas_dim=1,
            process_noise_sqrt=0.01 * np.eye(2),
            meas_noise_sqrt=np.array([[0.1]]),
        )
        ukf.reset(0.1)
        ukf.x = np.array([0.5, 0.0])

        traces = []
        for _ in range(50):
            ukf.predict(process_model, dt=0.01)
            traces.append(np.trace(ukf.P))

        # Trace should be monotonically increasing (or at least non-decreasing)
        for i in range(1, len(traces)):
            assert traces[i] >= traces[i - 1] - 1e-10
