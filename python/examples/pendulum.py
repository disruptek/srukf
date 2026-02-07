#!/usr/bin/env python3
"""Nonlinear pendulum tracking with SR-UKF.

Port of the C 01_pendulum example.  Optionally generates a matplotlib
plot if matplotlib is installed.

Usage:
    python pendulum.py              # text output only
    python pendulum.py --plot       # with matplotlib visualization
    python pendulum.py --duration 20
"""

import argparse
import math
import sys

import numpy as np

from srukf import UnscentedKalmanFilter

# ---------------------------------------------------------------------------
# Pendulum physics
# ---------------------------------------------------------------------------

G = 9.81   # gravitational acceleration (m/s^2)
L = 1.0    # pendulum length (m)
B = 0.1    # damping coefficient


def pendulum_rk4(theta, omega, dt):
    """RK4 integration of the damped pendulum ODE."""

    def deriv(th, om):
        return om, -(G / L) * math.sin(th) - B * om

    k1t, k1o = deriv(theta, omega)
    k2t, k2o = deriv(theta + k1t * dt / 2, omega + k1o * dt / 2)
    k3t, k3o = deriv(theta + k2t * dt / 2, omega + k2o * dt / 2)
    k4t, k4o = deriv(theta + k3t * dt, omega + k3o * dt)

    return (
        theta + (dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t),
        omega + (dt / 6) * (k1o + 2 * k2o + 2 * k3o + k4o),
    )


# ---------------------------------------------------------------------------
# UKF models
# ---------------------------------------------------------------------------


def process_model(x, dt=0.01):
    th, om = pendulum_rk4(x[0], x[1], dt)
    return np.array([th, om])


def measurement_model(x):
    return np.array([x[0]])


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run(duration=10.0, dt=0.01, meas_rate=20.0, meas_noise=0.1,
        process_noise=0.01, initial_angle=math.pi / 4, plot=False):
    rng = np.random.default_rng(42)
    n_steps = int(duration / dt)
    meas_period = 1.0 / meas_rate

    # Create filter
    ukf = UnscentedKalmanFilter(
        state_dim=2,
        meas_dim=1,
        process_noise_sqrt=process_noise * np.eye(2),
        meas_noise_sqrt=np.array([[meas_noise]]),
    )
    ukf.x = np.array([initial_angle + rng.normal(0, 0.1), rng.normal(0, 0.1)])
    ukf.S = 0.2 * np.eye(2)

    # Storage
    time_arr = np.zeros(n_steps)
    true_angle = np.zeros(n_steps)
    meas_arr = np.full(n_steps, np.nan)
    est_angle = np.zeros(n_steps)
    est_velocity = np.zeros(n_steps)
    uncertainty = np.zeros(n_steps)

    theta_true = initial_angle
    omega_true = 0.0
    time_since_meas = 0.0

    for step in range(n_steps):
        t = step * dt
        time_arr[step] = t

        # True dynamics with process noise
        theta_noisy = theta_true + rng.normal(0, process_noise * dt)
        omega_noisy = omega_true + rng.normal(0, process_noise * dt)
        theta_true, omega_true = pendulum_rk4(theta_noisy, omega_noisy, dt)
        true_angle[step] = theta_true

        # Predict
        ukf.predict(process_model, dt=dt)

        # Measurement
        time_since_meas += dt
        if time_since_meas >= meas_period:
            z = np.array([theta_true + rng.normal(0, meas_noise)])
            meas_arr[step] = z[0]
            ukf.update(z, measurement_model)
            time_since_meas = 0.0

        est_angle[step] = ukf.x[0]
        est_velocity[step] = ukf.x[1]
        uncertainty[step] = ukf.S[0, 0]

        if step % (n_steps // 10) == 0:
            print(f"  {step * 100 // n_steps:3d}%  "
                  f"t={t:.2f}  true={theta_true:.4f}  "
                  f"est={ukf.x[0]:.4f}  err={abs(ukf.x[0] - theta_true):.4f}")

    # Final statistics
    errors = np.abs(est_angle[50:] - true_angle[50:])
    print(f"\nResults (after convergence):")
    print(f"  Mean error:  {np.mean(errors):.4f} rad ({np.degrees(np.mean(errors)):.2f} deg)")
    print(f"  Max error:   {np.max(errors):.4f} rad ({np.degrees(np.max(errors)):.2f} deg)")
    print(f"  Final P trace: {np.trace(ukf.P):.6f}")

    if plot:
        _plot(time_arr, true_angle, meas_arr, est_angle, uncertainty)


def _plot(time_arr, true_angle, meas_arr, est_angle, uncertainty):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nInstall matplotlib for plotting: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Angle plot
    ax1.plot(time_arr, np.degrees(true_angle), "g-", alpha=0.8, label="True angle")
    meas_mask = ~np.isnan(meas_arr)
    ax1.scatter(
        time_arr[meas_mask],
        np.degrees(meas_arr[meas_mask]),
        c="red", s=8, alpha=0.5, label="Measurements", zorder=5,
    )
    ax1.plot(time_arr, np.degrees(est_angle), "b-", lw=2, label="SR-UKF estimate")
    ax1.fill_between(
        time_arr,
        np.degrees(est_angle - uncertainty),
        np.degrees(est_angle + uncertainty),
        alpha=0.2, color="blue", label=r"$\pm 1\sigma$",
    )
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title("Pendulum Tracking with SR-UKF")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Error plot
    error = np.degrees(np.abs(est_angle - true_angle))
    ax2.plot(time_arr, error, "r-", alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Absolute Error (degrees)")
    ax2.set_title("Tracking Error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pendulum_python.svg", dpi=150)
    print("\nSaved: pendulum_python.svg")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pendulum tracking with SR-UKF")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print("Pendulum Tracking with SR-UKF (Python)")
    print("=" * 45)
    run(duration=args.duration, meas_noise=args.noise, plot=args.plot)
