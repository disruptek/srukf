# Long-Duration Numerical Stability Test

This example stress-tests the SR-UKF implementation for numerical stability over extended operation (thousands of filter iterations). It demonstrates why the square-root formulation is superior to standard Kalman filtering for long-running systems.

## What This Example Shows

1. **Numerical Robustness** - Square-root formulation prevents covariance corruption
2. **Long-Duration Performance** - Filter operates correctly for extended periods
3. **Health Monitoring** - Track numerical metrics (condition numbers, positive definiteness)
4. **Filter Consistency** - NEES and NIS tests validate statistical properties
5. **Multiple Scenarios** - Test different challenging conditions

## The Problem: Numerical Instability in Standard Filters

Standard Kalman filters suffer from several numerical issues:

- **Covariance symmetry loss** - Rounding errors can make P non-symmetric
- **Positive definiteness violation** - P can become negative definite (impossible!)
- **Accumulated roundoff** - Errors compound over thousands of iterations
- **Ill-conditioning** - Small eigenvalues lead to numerical instability

**The SR-UKF solution:** By working with the Cholesky factor S (where P = SS'), the filter:
- Guarantees positive definiteness (by construction)
- Maintains symmetry automatically
- Provides better numerical conditioning
- Is more robust to roundoff errors

This example validates these claims empirically.

## Test Scenarios

### 1. Baseline (default)
- **Goal:** Establish performance baseline
- **Conditions:** Moderate noise, steady motion
- **Duration:** 1000 seconds (100,000 timesteps)
- **Expected:** Stable error, consistent NEES ≈ 4.0

### 2. High Dynamics
- **Goal:** Test under aggressive maneuvering
- **Conditions:** Sinusoidal forcing, high accelerations
- **Duration:** 1000 seconds
- **Expected:** Bounded error despite high dynamics

### 3. Poor Observability
- **Goal:** Test with infrequent measurements
- **Conditions:** 1 Hz GPS (vs 10 Hz baseline)
- **Duration:** 1000 seconds
- **Expected:** Higher uncertainty but no divergence

### 4. Extreme Noise
- **Goal:** Test with very noisy sensors
- **Conditions:** 10x measurement noise
- **Duration:** 1000 seconds
- **Expected:** Larger errors but stable operation

## Building and Running

### Prerequisites

- SR-UKF library built (`make lib` in root directory)
- (Optional) gnuplot for SVG generation: `apt install gnuplot`

### Build

```bash
make
```

### Run

Default scenario (baseline, 1000 seconds):
```bash
./stability
```

Run specific scenario with verbose output:
```bash
./stability --scenario=baseline --verbose
```

Test all scenarios:
```bash
for scenario in baseline dynamics observability noise; do
  ./stability --scenario=$scenario --format=all
done
```

Ultra-long stress test (10,000 seconds ≈ 2.8 hours simulated):
```bash
./stability --scenario=baseline --duration=10000 --verbose
```

### Command-Line Options

- `--scenario=NAME` - Test scenario (default: `baseline`)
  - Options: `baseline`, `dynamics`, `observability`, `noise`
- `--duration=SECONDS` - Simulation duration (default: 1000.0)
- `--format=FORMAT` - Output format: `svg`, `csv`, `json`, `all` (default: `svg`)
- `--verbose` - Print detailed progress during simulation
- `--open` - Automatically open SVG plots after generation
- `--help` - Show usage information

## Output Files

### SVG Plots (requires gnuplot)

**1. `stability_error.svg`** - Position error over time:
- Shows whether estimation error remains bounded
- Ideally: error stays within a band (no divergence)

**2. `stability_nees.svg`** - NEES consistency test:
- **NEES** (Normalized Estimation Error Squared) = error' * P⁻¹ * error
- Expected value ≈ 4.0 for 4D state (chi-squared distribution)
- Too low: filter overconfident (P too small)
- Too high: filter underconfident (P too large)
- Consistent filter: NEES fluctuates around 4.0

**3. `stability_trace.svg`** - Covariance trace evolution:
- Trace(P) = total uncertainty
- Should stabilize (not grow unbounded or collapse to zero)
- Reflects filter's confidence in its estimates

### CSV

`stability_results.csv` - Tab-delimited data with columns:
- `time` - Timestamp
- `position_error` - Position RMSE (m)
- `velocity_error` - Velocity RMSE (m/s)
- `cov_trace` - Covariance trace
- `nees` - NEES value
- `condition` - Condition number of covariance

## What to Observe

### Successful Test (Expected)

```
=============================================================
  Test Results
=============================================================
Completed: 100000 timesteps in 2.34 seconds (42735 steps/sec)
Measurements: 10000 (10.0 Hz average)

Error Statistics:
  Max position error: 8.23 m
  Final position error: 3.45 m
  Final velocity error: 1.12 m/s

Consistency Metrics:
  Mean NEES: 4.12 (expect ≈4.0 for consistent filter)
  Final covariance trace: 12.34
  Final condition number: 2.45e+01

Numerical Health:
  Health failures: 0 (0.00%)
  Status: ✓ EXCELLENT - No numerical issues detected
=============================================================
```

**Key indicators:**
- ✅ No health failures (NaN, Inf, or non-positive-definite covariance)
- ✅ NEES near 4.0 (filter is statistically consistent)
- ✅ Bounded position error (no divergence)
- ✅ Condition number < 10⁶ (numerically well-conditioned)
- ✅ Covariance trace stable (uncertainty neither exploding nor collapsing)

### Warning Signs

- ⚠️ NEES >> 4.0: Filter underconfident (tune measurement noise down)
- ⚠️ NEES << 4.0: Filter overconfident (increase process noise)
- ⚠️ Growing position error: Filter diverging (check noise tuning)
- ⚠️ Condition number > 10⁸: Numerical issues (but SR-UKF handles this!)
- ⚠️ Health failures > 0: Critical numerical problems

## Metrics Explained

### NEES (Normalized Estimation Error Squared)

NEES tests whether the filter's uncertainty estimates match actual errors:

```
NEES = (x_true - x_est)' * P⁻¹ * (x_true - x_est)
```

- **Chi-squared distributed** with n degrees of freedom (here n=4)
- **Expected value:** E[NEES] = n = 4.0
- **95% confidence interval:** [0.71, 9.49] for n=4
- Persistent values outside this range indicate filter inconsistency

### Condition Number

Measures numerical stability of covariance matrix:

```
κ(P) = λ_max / λ_min
```

- **κ ≈ 1:** Perfect conditioning (rare in practice)
- **κ < 10³:** Excellent (typical for SR-UKF)
- **κ < 10⁶:** Good (still reliable)
- **κ > 10⁸:** Poor (standard KF might fail here, SR-UKF is more robust)
- **κ → ∞:** Singular (filter has collapsed)

The SR-UKF's square-root formulation typically maintains better conditioning than standard implementations.

### Covariance Trace

Total uncertainty across all states:

```
Trace(P) = Σ P_ii = σ_x² + σ_y² + σ_vx² + σ_vy²
```

- Should reach steady-state after initial transient
- Growing trace → filter losing confidence (divergence)
- Collapsing trace → filter overconfident (covariance corruption)
- Stable trace → healthy operation

## Comparison with Standard UKF

While we don't implement a standard UKF here for comparison, typical issues with non-square-root formulations over long runs include:

| Issue | Standard UKF | SR-UKF |
|-------|--------------|--------|
| **Covariance symmetry** | Can be lost | Guaranteed |
| **Positive definiteness** | Can be violated | Guaranteed |
| **Numerical conditioning** | Degrades | Better maintained |
| **Long-duration stability** | May diverge | Robust |
| **Computational cost** | Slightly lower | Slightly higher (worth it!) |

The SR-UKF pays a small computational penalty (Cholesky updates vs direct covariance) for significant numerical benefits.

## Educational Value

This example teaches:

1. **Numerical analysis** - How to monitor filter health
2. **Consistency testing** - NEES/NIS validation techniques
3. **Long-term behavior** - What "stability" means for state estimation
4. **Robustness validation** - Stress-testing under challenging conditions
5. **Square-root benefits** - Why the SR formulation matters in practice

## Extending This Example

Try modifying:

- **Longer duration:** Test 100,000+ seconds (but be patient!)
- **Single precision:** Change to `float` instead of `double` to exacerbate numerical issues
- **Outlier injection:** Add occasional bad measurements to test robustness
- **Dimension scaling:** Increase state dimension (10D, 20D) to stress linear algebra
- **Comparison:** Implement standard UKF and compare health metrics

## Performance Notes

On a typical modern CPU:
- **100,000 timesteps:** ~2 seconds (baseline scenario)
- **1,000,000 timesteps:** ~20 seconds (10x longer simulation)
- **Memory usage:** Minimal (<10 MB for million timesteps)

The SR-UKF is efficient enough for real-time embedded applications.

## Troubleshooting

### Build errors

Make sure SR-UKF library is built:
```bash
cd ../..
make lib
cd examples/03_stability_test
make
```

### Slow execution

For faster testing during development:
```bash
./stability --duration=100  # 100 seconds instead of 1000
```

### Health failures detected

If you see health failures:
1. Check if using correct compiler optimization (`-O2`)
2. Verify LAPACK linkage (needed for condition number computation)
3. Try different scenario (some are intentionally challenging)
4. This is still informative! The test is detecting numerical issues.

### NEES far from 4.0

This indicates noise tuning issues, not necessarily a bug:
- **NEES >> 4.0:** Reduce measurement noise parameter
- **NEES << 4.0:** Increase process noise parameter

## Next Steps

After running this test:

1. **Experiment with scenarios** - Try all four, compare results
2. **Inspect CSV output** - Plot condition numbers over time
3. **Modify parameters** - See how noise levels affect consistency
4. **Read the code** - Understand health metric computations

## References

- Van der Merwe & Wan (2001): "The Square-Root Unscented Kalman Filter"
- Bar-Shalom et al. (2001): "Estimation with Applications to Tracking and Navigation" (NEES/NIS theory)
- Higham (2002): "Accuracy and Stability of Numerical Algorithms" (numerical conditioning)
- Main SR-UKF library: `../../srukf.h`
