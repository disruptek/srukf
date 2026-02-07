#!/usr/bin/env python3
"""Minimal SR-UKF example: 1D constant-velocity tracking."""

import numpy as np

from srukf import UnscentedKalmanFilter

# Create filter: 2 states [position, velocity], 1 measurement [position]
ukf = UnscentedKalmanFilter(
    state_dim=2,
    meas_dim=1,
    process_noise_sqrt=np.diag([0.1, 0.01]),
    meas_noise_sqrt=np.array([[0.5]]),
)


# Process model: constant velocity  x_next = [pos + dt*vel, vel]
def process_model(x, dt=0.1):
    return np.array([x[0] + dt * x[1], x[1]])


# Measurement model: observe position only
def meas_model(x):
    return np.array([x[0]])


# Simulate a target moving at constant velocity with noisy measurements
rng = np.random.default_rng(42)
true_velocity = 1.0
dt = 0.1

for t in range(100):
    true_pos = t * dt * true_velocity
    measurement = np.array([true_pos]) + rng.normal(0, 0.5, size=1)

    ukf.predict(process_model, dt=dt)
    ukf.update(measurement, meas_model)

    print(
        f"t={t * dt:5.1f}  "
        f"true={true_pos:7.3f}  "
        f"meas={measurement[0]:7.3f}  "
        f"est_pos={ukf.x[0]:7.3f}  "
        f"est_vel={ukf.x[1]:7.3f}"
    )
