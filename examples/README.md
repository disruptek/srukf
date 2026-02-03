# SR-UKF Examples

This directory contains examples demonstrating the Square-Root Unscented Kalman Filter in action. Each example is self-contained with detailed documentation, build instructions, and pre-generated visualizations.

## Available Examples

### 1. **Pendulum Tracking** (`01_pendulum/`)

Track a nonlinear pendulum with noisy angle measurements.

**Difficulty**: Beginner  
**Concepts**: Nonlinear dynamics, state estimation, RK4 integration  
**Dimensions**: 2D state (angle, velocity), 1D measurement

**Why this example?**
- Easy to understand physically
- Clear nonlinearity (`sin(θ)` in dynamics)
- Beautiful phase-space visualization
- Perfect introduction to UKF concepts

![Pendulum Example](01_pendulum/pendulum.svg)

[→ Run this example](01_pendulum/)

---

### 2. **GPS + IMU Fusion** (`02_gps_imu_1d/`) *(Coming Soon)*

Fuse continuous IMU measurements with intermittent GPS updates in 1D motion.

**Difficulty**: Intermediate  
**Concepts**: Sensor fusion, intermittent measurements, uncertainty growth  
**Dimensions**: 3D state (position, velocity, acceleration), 2 measurement types

**Why this example?**
- Real-world application (matches README motivation)
- Shows uncertainty growing/shrinking
- Demonstrates handling measurement dropouts
- Foundation for full 3D navigation

---

### 3. **Long-Duration Stability** (`03_stability_test/`) *(Coming Soon)*

Demonstrate SR-UKF's numerical stability over 100,000+ timesteps.

**Difficulty**: Intermediate  
**Concepts**: Numerical stability, covariance conditioning, long-term filtering  
**Dimensions**: Configurable

**Why this example?**
- Proves the core SR-UKF value proposition
- Shows robustness to numerical issues
- Educational for understanding square-root formulation benefits
- Comparison point when other filters are implemented

---

## Building Examples

### Prerequisites

1. **SR-UKF Library**: Build the main library first
   ```bash
   cd ..
   make lib
   ```

2. **Dependencies**: 
   - Standard build tools (gcc, make)
   - LAPACK/BLAS (liblapacke-dev, libopenblas-dev)
   - **Optional**: gnuplot for SVG generation

3. **Install gnuplot** (optional but recommended):
   ```bash
   sudo apt install gnuplot      # Debian/Ubuntu
   brew install gnuplot          # macOS
   ```

### Build All Examples

```bash
make examples        # From root directory
```

Or build individually:

```bash
cd examples/01_pendulum
make
./pendulum --help
```

## Output Formats

All examples support multiple output formats via `--format` flag:

### SVG (Default)

Professional-quality vector graphics generated via gnuplot:
- Dark mode enabled for better readability
- High resolution (1200x800)
- Embedded fonts and styling
- Can be opened directly or included in documentation

If gnuplot is unavailable, pre-generated SVGs are included in the repository.

### CSV

Comma-separated values for analysis in external tools:
- Multiple CSV files if different time axes exist
- ISO8601-style timestamps (elapsed seconds with microsecond precision)
- One file per unique time axis
- Easy to import into Python, R, MATLAB, Excel, etc.

Example:
```csv
timestamp,True Angle,Measurements,SR-UKF Estimate
0.000000,0.785398,NaN,0.792156
0.010000,0.784234,NaN,0.791023
0.020000,0.782897,0.821435,0.789654
...
```

### JSON

Structured data for web visualization:
- Simple schema: array of series, each with name and data points
- Suitable for D3.js, Chart.js, etc.
- Includes metadata in comments (can be extended)

Example:
```json
{
  "series": [
    {
      "name": "True Angle",
      "data": [
        {"t": 0.000000, "y": 0.785398},
        {"t": 0.010000, "y": 0.784234},
        ...
      ]
    },
    ...
  ]
}
```

## Common Utilities

The `common/` directory provides shared functionality:

- **`plot_utils.h/.c`**: Multi-format output, gnuplot integration
- **`example_helpers.h/.c`** *(TBD)*: Filter setup helpers, noise generation

These utilities handle the grunt work so example code can focus on the filtering problem.

## Design Philosophy

Each example follows these principles:

1. **Heavy Documentation**: Every function, every parameter explained
2. **Physical Intuition**: Real-world context, not just math
3. **Self-Contained**: Can be understood without reading other examples
4. **Pedagogical**: Designed for learning, not just demonstration
5. **Production-Quality Code**: Real comments, error handling, clean structure

## Running Examples

### Quick Start

```bash
cd examples/01_pendulum
make
./pendulum --open
```

### Common Options

Most examples support:
- `--duration=SECONDS` - How long to simulate
- `--noise=LEVEL` - Measurement noise standard deviation
- `--format=FORMAT` - Output format (svg, csv, json, all)
- `--open` - Auto-open SVG after generation
- `--help` - Show usage information

### Tips

1. **Start with default settings** to see expected behavior
2. **Increase noise** to stress-test the filter
3. **Generate all formats** (`--format=all`) for analysis
4. **Use `--open`** for immediate visual feedback

## Understanding the Output

### What to Look For

1. **Measurement Scatter**: Raw sensor data is noisy
2. **Smooth Estimates**: Filter output is smooth despite noise
3. **Tracking Accuracy**: Estimate follows true state closely
4. **Uncertainty**: Confidence bounds show filter's self-awareness
5. **Phase Behavior**: Small lag is normal and tunable

### Red Flags

- **Divergence**: Estimate drifts away from truth → tune noise parameters
- **Jerky Estimates**: Too much noise rejection → increase process noise
- **Ignoring Measurements**: Filter too conservative → decrease measurement noise

## Contributing New Examples

Wanted examples:
- 2D object tracking (bearing-only measurements)
- Attitude estimation (quaternions)
- Target tracking with clutter
- Battery state-of-charge estimation
- Any sensor fusion application!

See existing examples for structure and documentation style.

## Troubleshooting

### "srukf.h not found"

Build the library first:
```bash
cd ../..
make lib
```

### "gnuplot not found"

Install gnuplot or use `--format=csv`:
```bash
sudo apt install gnuplot
# OR
./pendulum --format=csv
```

### Example crashes or gives bad results

1. Check that library is up to date: `make clean && make lib` in root
2. Try default parameters first before customizing
3. Check noise parameters are reasonable for your problem scale
4. Look at CSV output to debug numeric issues

### Build errors with LAPACK

Make sure LAPACK/BLAS are installed:
```bash
sudo apt install libopenblas-dev liblapacke-dev
```

## Next Steps

1. **Start with pendulum example** - easiest to understand
2. **Read the source code** - it's heavily documented for learning
3. **Experiment with parameters** - break things and see what happens!
4. **Try implementing your own example** - best way to learn

## Web-Based Interactive Explainer

Coming soon: A web application with:
- Interactive parameter tuning
- Live sigma point visualization
- Algorithm step-through
- Side-by-side filter comparison

Will be hosted at: `https://disruptek.github.io/srukf/explainer/`

## Questions?

- See individual example READMEs for specific guidance
- Check the main [SR-UKF documentation](../docs/)
- Review the library API reference
- Look at test cases in `../tests/` for more usage patterns
