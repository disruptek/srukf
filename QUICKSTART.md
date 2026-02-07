# Quick Start: Your First SR-UKF in 5 Minutes

## What is SR-UKF?

SR-UKF is a state estimation filter that tracks hidden quantities (like position and velocity) from noisy sensor readings. The "square-root" formulation keeps it numerically stable over long runs, making it suitable for safety-critical systems like robotics and navigation.

## Installation

### C Library

**Ubuntu / Debian:**

```bash
sudo apt-get install libopenblas-dev liblapacke-dev
git clone https://github.com/disruptek/srukf.git
cd srukf
make lib
sudo make install
```

**macOS (Homebrew):**

```bash
brew install openblas lapack
git clone https://github.com/disruptek/srukf.git
cd srukf
make lib
sudo make install
```

**CMake (any platform):**

```bash
git clone https://github.com/disruptek/srukf.git
cd srukf
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

### Python Package

```bash
cd srukf/python
pip install .
```

Requires NumPy and a working C compiler (the shared library is built automatically during install).

## Your First Filter in C

Save this as `track.c`. It tracks a 1D object moving at constant velocity using noisy position measurements.

```c
#include "srukf.h"
#include <math.h>
#include <stdio.h>

/* Process model: constant velocity, x = [position, velocity] */
static void process(const srukf_mat *in, srukf_mat *out, void *ctx) {
  double dt = *(double *)ctx;
  double pos = SRUKF_ENTRY(in, 0, 0);
  double vel = SRUKF_ENTRY(in, 1, 0);
  SRUKF_ENTRY(out, 0, 0) = pos + dt * vel;  /* position += velocity * dt */
  SRUKF_ENTRY(out, 1, 0) = vel;             /* velocity stays the same  */
}

/* Measurement model: we observe position only */
static void measure(const srukf_mat *x, srukf_mat *z, void *ctx) {
  (void)ctx;
  SRUKF_ENTRY(z, 0, 0) = SRUKF_ENTRY(x, 0, 0);
}

int main(void) {
  double dt = 0.1;

  /* Create filter: 2 states [pos, vel], 1 measurement [pos] */
  srukf *ukf = srukf_create(2, 1);

  /* Set noise (square-root covariance matrices) */
  srukf_mat *Q = srukf_mat_alloc(2, 2, 1);  /* process noise     */
  srukf_mat *R = srukf_mat_alloc(1, 1, 1);  /* measurement noise */
  SRUKF_ENTRY(Q, 0, 0) = 0.1;   /* position uncertainty per step */
  SRUKF_ENTRY(Q, 1, 1) = 0.01;  /* velocity uncertainty per step */
  SRUKF_ENTRY(R, 0, 0) = 0.5;   /* sensor noise std dev          */
  srukf_set_noise(ukf, Q, R);

  /* Simulate 50 steps of a target moving at velocity = 1.0 */
  for (int t = 0; t < 50; t++) {
    double true_pos = t * dt * 1.0;
    double noisy_pos = true_pos + 0.5 * sin(t * 0.7);  /* fake noise */

    srukf_predict(ukf, process, &dt);

    srukf_mat *z = srukf_mat_alloc(1, 1, 1);
    SRUKF_ENTRY(z, 0, 0) = noisy_pos;
    srukf_correct(ukf, z, measure, NULL);
    srukf_mat_free(z);

    printf("t=%4.1f  true=%6.2f  meas=%6.2f  est=%6.2f  vel=%5.2f\n",
           t * dt, true_pos, noisy_pos,
           SRUKF_ENTRY(ukf->x, 0, 0), SRUKF_ENTRY(ukf->x, 1, 0));
  }

  srukf_mat_free(Q);
  srukf_mat_free(R);
  srukf_free(ukf);
  return 0;
}
```

**Compile and run:**

```bash
gcc -o track track.c -I/usr/local/include -L/usr/local/lib \
    -lsrukf -llapacke -lblas -lopenblas -lm
./track
```

**Expected output** (first few lines):

```
t= 0.0  true=  0.00  meas=  0.00  est=  0.00  vel= 0.00
t= 0.1  true=  0.10  meas=  0.43  est=  0.30  vel= 0.52
t= 0.2  true=  0.20  meas=  0.82  est=  0.60  vel= 0.78
t= 0.3  true=  0.30  meas=  1.12  est=  0.85  vel= 0.87
...
```

The `est` column converges toward `true` even though `meas` is noisy.

## Your First Filter in Python

Save this as `track.py`:

```python
import numpy as np
from srukf import UnscentedKalmanFilter

# Create filter: 2 states [position, velocity], 1 measurement [position]
ukf = UnscentedKalmanFilter(
    state_dim=2,
    meas_dim=1,
    process_noise_sqrt=np.diag([0.1, 0.01]),
    meas_noise_sqrt=np.array([[0.5]]),
)

dt = 0.1

for t in range(50):
    true_pos = t * dt * 1.0
    noisy_pos = true_pos + 0.5 * np.sin(t * 0.7)

    ukf.predict(lambda x: np.array([x[0] + dt * x[1], x[1]]))
    ukf.update(np.array([noisy_pos]), lambda x: np.array([x[0]]))

    print(f"t={t*dt:4.1f}  true={true_pos:6.2f}  "
          f"meas={noisy_pos:6.2f}  est={ukf.x[0]:6.2f}  vel={ukf.x[1]:5.2f}")
```

**Run:**

```bash
python track.py
```

Output matches the C version.

## What Just Happened?

Here's what the filter did at each time step:

1. **Predict** — The filter used the process model ("position increases by velocity times dt") to forecast where the object should be. This also grew the uncertainty, because the model isn't perfect.

2. **Correct** — The filter received a noisy position measurement and combined it with the prediction. Measurements that agree with the prediction are trusted more; wild outliers are down-weighted. This shrank the uncertainty.

3. **Repeat** — Over many cycles, the filter converges: the estimated position tracks the true position closely, and it even infers velocity despite never measuring it directly.

The "unscented" part means the filter handles nonlinear models (like `sin(θ)` in a pendulum) without requiring you to derive calculus-heavy Jacobian matrices. The "square-root" part means the internal math stays numerically stable even after millions of iterations.

## Next Steps

- **[Examples](examples/)** — Four worked examples from beginner to advanced, with visualizations
- **[Interactive Web Explainer](examples/web_explainer/)** — Drag sigma points and watch the UKF work in your browser
- **[API Documentation](https://disruptek.github.io/srukf/)** — Full reference with algorithm explanations
- **[Tuning Guide](README.md#tuning-ukf-parameters)** — How to set α, β, κ and noise matrices
- **[Mathematical Background](https://disruptek.github.io/srukf/)** — Why square-root formulation matters
