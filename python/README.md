# SR-UKF Python Bindings

Python interface to the [Square-Root Unscented Kalman Filter](https://github.com/disruptek/srukf) C library via ctypes.

## Installation

### From source (recommended)

```bash
# Clone the repository
git clone https://github.com/disruptek/srukf.git
cd srukf

# Build the C library
make lib

# Install the Python package
cd python
pip install -e .
```

### With system library

If `libsrukf.so` is installed system-wide (e.g., via `make install`):

```bash
cd python
pip install .
```

### Dependencies

**Build-time:** C compiler, CBLAS, LAPACKE (e.g., OpenBLAS)

**Runtime:** numpy >= 1.19.0

**Optional:** matplotlib (for visualization), pytest (for testing)

## Quick Start

```python
import numpy as np
from srukf import UnscentedKalmanFilter

# 1D constant-velocity tracking
ukf = UnscentedKalmanFilter(
    state_dim=2,           # [position, velocity]
    meas_dim=1,            # [position]
    process_noise_sqrt=np.diag([0.1, 0.01]),
    meas_noise_sqrt=np.array([[0.5]]),
)

def process_model(x, dt=0.1):
    return np.array([x[0] + dt * x[1], x[1]])

def meas_model(x):
    return np.array([x[0]])

for t in range(100):
    ukf.predict(process_model, dt=0.1)
    z = np.array([t * 0.1]) + np.random.randn(1) * 0.5
    ukf.update(z, meas_model)
    print(f"pos={ukf.x[0]:.3f}, vel={ukf.x[1]:.3f}")
```

## API Reference

### `UnscentedKalmanFilter`

**Constructor:**
```python
UnscentedKalmanFilter(
    state_dim,              # int: state vector dimension N
    meas_dim,               # int: measurement vector dimension M
    process_noise_sqrt,     # (N, N) array: sqrt of process noise covariance
    meas_noise_sqrt,        # (M, M) array: sqrt of measurement noise covariance
    alpha=1e-3,             # float: sigma point spread
    beta=2.0,               # float: prior distribution (2.0 for Gaussian)
    kappa=0.0,              # float: secondary scaling
)
```

**Properties:**
- `ukf.x` — state estimate (1-D numpy array, get/set)
- `ukf.S` — sqrt-covariance (2-D numpy array, get/set)
- `ukf.P` — full covariance `S @ S.T` (computed, read-only)
- `ukf.state_dim` — state dimension N
- `ukf.meas_dim` — measurement dimension M

**Methods:**
- `ukf.predict(process_model, **kwargs)` — prediction step
- `ukf.update(measurement, meas_model, **kwargs)` — correction step
- `ukf.reset(init_std=1.0)` — reset to zero state, identity covariance
- `ukf.set_noise(Q_sqrt, R_sqrt)` — update noise matrices
- `ukf.set_scale(alpha, beta, kappa)` — update UKF parameters

All methods return `self` for chaining:
```python
ukf.reset(1.0).predict(f).update(z, h)
```

### Exceptions

- `SrukfError` — base exception
- `SrukfParameterError` — invalid parameters (also a `ValueError`)
- `SrukfMathError` — numerical failure (non-SPD matrix, NaN, etc.)

## Differences from C API

| C API | Python API |
|-------|-----------|
| `srukf_create(N, M)` | `UnscentedKalmanFilter(N, M, Q, R)` |
| `srukf_predict(ukf, f, user)` | `ukf.predict(f, **kwargs)` |
| `srukf_correct(ukf, z, h, user)` | `ukf.update(z, h, **kwargs)` |
| `srukf_get_state(ukf, out)` | `ukf.x` |
| `srukf_set_state(ukf, x)` | `ukf.x = x` |
| Return codes | Exceptions |
| `void*` user data | `**kwargs` |
| Manual memory management | Automatic via `__del__` |

## Examples

```bash
python examples/simple.py           # Minimal tracking example
python examples/pendulum.py         # Nonlinear pendulum
python examples/pendulum.py --plot  # With matplotlib visualization
```

## Testing

```bash
pip install -e .[test]
pytest -v
```

## License

MIT — same as the C library.
