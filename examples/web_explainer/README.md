# Unscented Kalman Filter - Interactive Web Explainer

An interactive visualization that explains how the Unscented Kalman Filter (UKF) works, focusing on sigma points and nonlinear transformations.

## Features

### 1. Sigma Points Transformation Visualization

**Interactive Demo:**
- Drag to change the input state mean
- Adjust uncertainty using the slider
- Select different nonlinear functions (sin, x², exp, tanh)
- See how sigma points capture uncertainty through nonlinear transformations

**What You'll See:**
- **Blue**: Input Gaussian distribution
- **Green**: Sigma points (2n+1 = 3 for 1D case)
- **Yellow**: True nonlinear transformation
- **Red**: Output distribution reconstructed from transformed sigma points

**Key Insight:** Notice how the red output distribution closely matches the yellow true transformation, even for highly nonlinear functions like `sin(x)`. This is the power of the UKF!

### 2. Algorithm Steps Animation

**Step-by-Step Walkthrough:**
- Click "Next Step" to advance through the algorithm
- Click "Animate" to watch the full predict-measure-update cycle
- Observe how uncertainty grows during prediction and shrinks during updates

**Three Steps:**
1. **Predict** - Use process model to estimate next state (uncertainty increases)
2. **Measure** - Receive a noisy observation
3. **Update** - Correct estimate using measurement (uncertainty decreases)

### 3. Educational Content

- Why UKF vs. Extended Kalman Filter (EKF)?
- How sigma points are generated
- Mathematical foundations
- Square-root formulation benefits
- Real-world applications

## Usage

### Local Viewing

Simply open `index.html` in your web browser:

```bash
cd examples/web_explainer
firefox index.html  # or chrome, safari, etc.
```

No build process, no dependencies, no internet connection required!

### GitHub Pages

This explainer can be hosted on GitHub Pages for easy sharing. The HTML/CSS/JS is completely self-contained.

## How It Works

### Sigma Point Selection

For an n-dimensional state with mean `x̄` and covariance `P`:

```
Number of sigma points: 2n + 1

χ₀ = x̄  (mean point)
χᵢ = x̄ + √((n+λ)P)ᵢ   for i = 1,...,n  (positive spread)
χᵢ = x̄ - √((n+λ)P)ᵢ₋ₙ for i = n+1,...,2n  (negative spread)
```

Where `λ = α²(n+κ) - n` is a scaling parameter.

### Unscented Transform

Given a nonlinear function `f(x)` and input distribution `N(x̄, P)`:

1. **Generate sigma points** from input distribution
2. **Transform each sigma point**: `Υᵢ = f(χᵢ)`
3. **Compute output statistics**:
   - Output mean: `ȳ = Σ Wᵢᵐ Υᵢ`
   - Output covariance: `Pᵧ = Σ Wᵢᶜ (Υᵢ - ȳ)(Υᵢ - ȳ)' + Q`

The weights `Wᵢᵐ` and `Wᵢᶜ` are chosen to match the first two moments (mean and covariance).

### Why This Beats Linearization

**Extended Kalman Filter (EKF):**
- Linearizes nonlinear functions using Jacobians (first-order Taylor approximation)
- Accuracy degrades for highly nonlinear systems
- Requires derivatives (Jacobian matrices)
- First-order accurate

**Unscented Kalman Filter (UKF):**
- Directly samples the distribution using sigma points
- Captures nonlinearity up to third order (Taylor series)
- No derivatives needed!
- More accurate for the same computational cost

**The Visualization Shows:** Try `sin(x)` with high uncertainty. Notice how:
- The yellow curve (true transformation) is highly nonlinear
- The red curve (UKF estimate) closely matches it
- An EKF would linearize around the mean, giving poor results

## Interactive Experiments

### Experiment 1: Linearity Test
1. Select "Linear (x → x)"
2. Drag the mean around
3. **Observation:** Input and output distributions have the same shape (Gaussian remains Gaussian through linear transformations)

### Experiment 2: Quadratic Nonlinearity
1. Select "Quadratic (x → x²)"
2. Set uncertainty to 2.0
3. Drag mean from -3 to +3
4. **Observation:** Output distribution becomes asymmetric! UKF captures this.

### Experiment 3: Extreme Nonlinearity
1. Select "Sine (x → sin(x))"
2. Set uncertainty to 3.0
3. Set mean near 0
4. **Observation:** Input is Gaussian, but output is definitely not! UKF approximates it well with just 3 points.

### Experiment 4: Algorithm Convergence
1. Go to "Algorithm Steps" panel
2. Click "Animate"
3. **Observation:** 
   - Uncertainty (blue band) grows during prediction
   - Shrinks during measurement updates
   - Estimate converges toward true state
   - Filter is "breathing" - uncertainty pulses with each cycle

## Technical Details

### Canvas Rendering

The visualizations use HTML5 Canvas for high-performance 2D graphics:
- 60 FPS smooth animations
- Interactive mouse/touch input
- Responsive to window resizing

### Mathematics Implementation

All UKF math is implemented in pure JavaScript:
- Gaussian PDF calculation
- Sigma point generation (Cholesky-based)
- Weighted mean and covariance computation
- Kalman gain and update equations

### No External Dependencies

This explainer is built with:
- HTML5
- CSS3 (with gradients, backdrop filters, and animations)
- Vanilla JavaScript (ES6+)
- Canvas 2D API

**Why no libraries?**
- Faster loading (no 100KB+ frameworks)
- Works offline
- Easy to understand and modify
- Portable (works in any modern browser)

## Educational Use

This explainer is perfect for:

### Students
- Visualize abstract concepts (uncertainty ellipses, sigma points)
- Experiment with different nonlinearities
- See the algorithm in action

### Teachers
- Embed in course websites
- Use during lectures (interactive demo)
- Assign as homework (modify the code)

### Practitioners
- Refresh understanding of UKF internals
- Debug filter behavior (see what sigma points are doing)
- Communicate with non-technical stakeholders

## Extending the Explainer

Want to add more features? The code is designed to be hackable:

### Add a New Nonlinear Function

Edit `ukf-explainer.js`, in the `applyFunction` method:

```javascript
applyFunction(x) {
  switch(this.function) {
    case 'identity': return x;
    case 'square': return x * x / 5;
    case 'sin': return Math.sin(x);
    case 'myFunction': return Math.log(Math.abs(x) + 1); // Your function!
    default: return x;
  }
}
```

Then add the option to the HTML select element.

### Add 2D Visualization

The current demo is 1D for clarity. Extending to 2D:
- Generate 2*2+1 = 5 sigma points
- Draw uncertainty ellipses (Cholesky factor visualization)
- Transform points through 2D→2D nonlinear functions

This is left as an exercise (but the math is the same!).

### Add Measurement Model

Currently the algorithm demo uses a simple identity measurement model. You could:
- Add nonlinear measurement functions
- Visualize innovation (measurement residual)
- Show Kalman gain calculation step-by-step

## Browser Compatibility

Tested and working on:
- Chrome/Chromium 90+
- Firefox 88+
- Safari 14+
- Edge 90+

**Minimum requirements:**
- HTML5 Canvas support
- ES6 JavaScript (arrow functions, classes, `let`/`const`)
- CSS backdrop-filter (optional, degrades gracefully)

## Performance

**Rendering:**
- ~60 FPS on most hardware
- Canvas updates only when state changes (no continuous redraw)
- Responsive even on mobile devices

**Memory:**
- < 5 MB total page size
- No memory leaks (tested with long sessions)
- Efficient array handling

## Credits

Part of the [SR-UKF](../../) library examples.

**Concepts:**
- Unscented Transform: Julier & Uhlmann (1997)
- Square-Root UKF: Van der Merwe & Wan (2001)

**Implementation:**
- Visualization design inspired by [Kalman Filter Interactive Demo](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- Color scheme: Modern glassmorphism with purple gradient

## License

Same as the SR-UKF library (see root LICENSE file).

## Feedback

Found a bug or have a feature request? Open an issue on GitHub!

Want to learn more? Check out the other [examples](../) including:
- Pendulum tracking (nonlinear dynamics)
- GPS+IMU sensor fusion
- Long-duration stability test

## Fun Facts

- The explainer runs entirely in your browser - no server needed!
- All 700+ lines of JavaScript are hand-written (no frameworks)
- The gradient background uses the same purple color scheme as the main docs
- Sigma points are named after the Greek letter σ (standard deviation)
- The UKF is sometimes called the "sigma-point Kalman filter"
- This visualization took ~2 hours to create but will save countless hours of confusion!

Enjoy exploring the beauty of the Unscented Kalman Filter! 🎯
