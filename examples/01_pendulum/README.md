# Pendulum Tracking with SR-UKF

This example demonstrates the Square-Root Unscented Kalman Filter tracking a nonlinear pendulum system.

## What This Example Shows

1. **Nonlinear Dynamics**: The pendulum's equations of motion involve `sin(θ)`, making it inherently nonlinear
2. **State Estimation**: We estimate both angle and angular velocity, even though we only measure angle
3. **Uncertainty Tracking**: The filter maintains and propagates uncertainty estimates
4. **Sensor Fusion**: Combines noisy measurements with a physics-based model

## The Physics

A damped pendulum follows these equations:

```
dθ/dt = ω
dω/dt = -(g/L)sin(θ) - b*ω
```

Where:
- θ = angle from vertical (radians)
- ω = angular velocity (rad/s)
- g = 9.81 m/s² (gravity)
- L = 1.0 m (pendulum length)
- b = 0.1 (damping coefficient)

The `sin(θ)` term is what makes this nonlinear and interesting for the UKF!

## Building and Running

### Prerequisites

- SR-UKF library built (`make lib` in the root directory)
- (Optional) gnuplot for SVG generation: `apt install gnuplot`

### Build

```bash
make
```

### Run

Default run (10 seconds, SVG output):
```bash
./pendulum
```

Longer simulation with auto-open:
```bash
./pendulum --duration=20 --open
```

Adjust noise levels:
```bash
./pendulum --noise=0.2
```

Generate all output formats:
```bash
./pendulum --format=all
```

### Command-Line Options

- `--duration=SECONDS` - Simulation duration (default: 10.0)
- `--noise=LEVEL` - Measurement noise std dev in radians (default: 0.1)
- `--format=FORMAT` - Output format: `svg`, `csv`, `json`, or `all` (default: `svg`)
- `--open` - Automatically open SVG after generation
- `--help` - Show usage information

## Output Files

### SVG (requires gnuplot)

`pendulum.svg` - Visualization showing:
- True pendulum motion (green line)
- Noisy measurements (red points)
- SR-UKF estimate (blue line)

Dark mode is enabled by default for better readability.

![Pendulum Tracking Results](pendulum.svg)

### CSV

`pendulum.csv` - Tab-delimited data with columns:
- `timestamp` - Time in seconds
- `True Angle` - Ground truth angle
- `Measurements` - Noisy sensor readings (NaN when no measurement)
- `SR-UKF Estimate` - Filtered angle estimate

Multiple CSV files are created if different time axes are used (e.g., different measurement rates).

### JSON

`pendulum.json` - Structured data format suitable for web visualization or further analysis.

## What to Observe

### In the Plots

1. **Measurement Noise**: Notice the scatter in the red measurement points
2. **Smooth Estimates**: The blue SR-UKF line is smooth despite noisy inputs
3. **Tracking Accuracy**: The estimate (blue) closely follows the truth (green)
4. **Phase Lag**: Small delay is normal - the filter balances responsiveness vs noise rejection

### In the Code

The heavily commented source code (`pendulum.c`) explains:

- **RK4 Integration**: How we propagate the nonlinear ODE accurately
- **Process Model**: How the UKF predicts state evolution
- **Measurement Model**: What we actually observe vs what we estimate
- **Noise Tuning**: How to set Q (process noise) and R (measurement noise)
- **Filter Loop**: Predict → Measure → Correct cycle

## Educational Value

This example is ideal for learning because:

1. **Simple Physics**: Everyone understands a swinging pendulum
2. **Clear Nonlinearity**: The `sin(θ)` makes linear approximations fail visibly
3. **Visual Feedback**: Plots immediately show if the filter works
4. **Complete Implementation**: Every step is explained in comments

## Extending This Example

Try modifying:

- **Initial conditions**: Start with larger angles (e.g., θ₀ = π/2)
- **Damping**: Increase/decrease `b` to see energy dissipation effects
- **Measurement rate**: Change `measurement_rate` to simulate slower sensors
- **Noise levels**: Crank up `measurement_noise` to stress-test the filter
- **Process model**: Add external forcing (e.g., wind gusts)

## Troubleshooting

### "gnuplot not found"

SVG generation requires gnuplot. Install it:
```bash
sudo apt install gnuplot  # Debian/Ubuntu
```

Or use CSV/JSON output and plot with your preferred tool.

A pre-generated SVG is included in the repository, so you can see expected results even without gnuplot.

### Build errors

Make sure the SR-UKF library is built first:
```bash
cd ../..
make lib
cd examples/01_pendulum
make
```

### Strange filter behavior

Check your noise parameters. If measurements are very noisy but `measurement_noise` is set too low, the filter will "trust" bad data. Conversely, setting it too high makes the filter ignore measurements.

Good rule of thumb: Set `measurement_noise` to match your actual sensor's noise level.

## Next Steps

After understanding this example, try:

1. **GPS+IMU Fusion** (`examples/02_gps_imu_1d`) - Intermittent measurements
2. **Stability Test** (`examples/03_stability_test`) - Long-duration numerical stability
3. **Web Explainer** (TBD) - Interactive visualization of UKF concepts

## References

- [Wikipedia: Unscented Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter)
- Van der Merwe & Wan (2001): "The Square-Root Unscented Kalman Filter for State and Parameter-Estimation"
- Main SR-UKF library: `../../srukf.h`
