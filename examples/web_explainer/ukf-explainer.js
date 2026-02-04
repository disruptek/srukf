/**
 * UKF Interactive Explainer - JavaScript
 * 
 * Visualizes how the Unscented Kalman Filter transforms uncertainty
 * through nonlinear functions using sigma points.
 */

// ============================================================================
// SIGMA POINTS VISUALIZATION
// ============================================================================

class SigmaPointsViz {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    
    // State
    this.mean = 0.0;
    this.variance = 1.0;
    this.function = 'sin';
    
    // Canvas dimensions
    this.width = this.canvas.width;
    this.height = this.canvas.height;
    this.padding = 60;
    
    // Axes ranges
    this.xMin = -5;
    this.xMax = 5;
    this.yMin = -3;
    this.yMax = 3;
    
    // Animation
    this.animating = false;
    
    this.setupInteraction();
    this.draw();
  }
  
  setupInteraction() {
    // Mouse interaction
    let isDragging = false;
    
    this.canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      this.updateMeanFromMouse(e);
    });
    
    this.canvas.addEventListener('mousemove', (e) => {
      if (isDragging) {
        this.updateMeanFromMouse(e);
      }
    });
    
    this.canvas.addEventListener('mouseup', () => {
      isDragging = false;
    });
    
    this.canvas.addEventListener('mouseleave', () => {
      isDragging = false;
    });
  }
  
  updateMeanFromMouse(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    this.mean = this.screenToWorldX(x);
    this.mean = Math.max(this.xMin, Math.min(this.xMax, this.mean));
    this.draw();
    this.updateStats();
  }
  
  worldToScreenX(x) {
    return this.padding + (x - this.xMin) / (this.xMax - this.xMin) * (this.width - 2 * this.padding);
  }
  
  worldToScreenY(y) {
    return this.height - this.padding - (y - this.yMin) / (this.yMax - this.yMin) * (this.height - 2 * this.padding);
  }
  
  screenToWorldX(sx) {
    return this.xMin + (sx - this.padding) / (this.width - 2 * this.padding) * (this.xMax - this.xMin);
  }
  
  screenToWorldY(sy) {
    return this.yMin + (this.height - this.padding - sy) / (this.height - 2 * this.padding) * (this.yMax - this.yMin);
  }
  
  // Nonlinear functions
  applyFunction(x) {
    switch(this.function) {
      case 'identity': return x;
      case 'square': return x * x / 5; // Scale down for visibility
      case 'sin': return Math.sin(x);
      case 'exp': return Math.exp(x / 5) - 1; // Shifted
      case 'tanh': return Math.tanh(x);
      default: return x;
    }
  }
  
  // Gaussian PDF
  gaussian(x, mean, variance) {
    return (1 / Math.sqrt(2 * Math.PI * variance)) * 
           Math.exp(-0.5 * Math.pow(x - mean, 2) / variance);
  }
  
  // Generate sigma points (1D case)
  generateSigmaPoints() {
    const lambda = 1.0; // UKF parameter (typically 3-n for n=1)
    const sigma = Math.sqrt((1 + lambda) * this.variance);
    
    return [
      { x: this.mean, weight: lambda / (1 + lambda) },
      { x: this.mean + sigma, weight: 1 / (2 * (1 + lambda)) },
      { x: this.mean - sigma, weight: 1 / (2 * (1 + lambda)) }
    ];
  }
  
  // Transform sigma points and compute output statistics
  transformSigmaPoints(sigmaPoints) {
    const transformed = sigmaPoints.map(sp => ({
      y: this.applyFunction(sp.x),
      weight: sp.weight
    }));
    
    // Compute output mean
    const outputMean = transformed.reduce((sum, sp) => sum + sp.weight * sp.y, 0);
    
    // Compute output variance
    const outputVariance = transformed.reduce((sum, sp) => 
      sum + sp.weight * Math.pow(sp.y - outputMean, 2), 0);
    
    return { transformed, outputMean, outputVariance };
  }
  
  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    
    // Draw axes
    this.drawAxes();
    
    // Draw true nonlinear transformation (yellow curve)
    this.drawTrueTransform();
    
    // Draw input Gaussian distribution
    this.drawInputDistribution();
    
    // Generate and draw sigma points
    const sigmaPoints = this.generateSigmaPoints();
    this.drawSigmaPoints(sigmaPoints);
    
    // Transform sigma points and draw output distribution
    const { transformed, outputMean, outputVariance } = this.transformSigmaPoints(sigmaPoints);
    this.drawOutputDistribution(outputMean, outputVariance);
    this.drawTransformedSigmaPoints(transformed);
  }
  
  drawAxes() {
    const ctx = this.ctx;
    
    // Axes lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(this.padding, this.worldToScreenY(0));
    ctx.lineTo(this.width - this.padding, this.worldToScreenY(0));
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(this.worldToScreenX(0), this.padding);
    ctx.lineTo(this.worldToScreenX(0), this.height - this.padding);
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Input State (x)', this.width / 2, this.height - 20);
    
    ctx.save();
    ctx.translate(20, this.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Output f(x)', 0, 0);
    ctx.restore();
    
    // Tick marks and numbers
    ctx.font = '12px Arial';
    for (let x = Math.ceil(this.xMin); x <= Math.floor(this.xMax); x++) {
      const sx = this.worldToScreenX(x);
      ctx.beginPath();
      ctx.moveTo(sx, this.worldToScreenY(0) - 5);
      ctx.lineTo(sx, this.worldToScreenY(0) + 5);
      ctx.stroke();
      ctx.fillText(x.toString(), sx, this.worldToScreenY(0) + 20);
    }
    
    for (let y = Math.ceil(this.yMin); y <= Math.floor(this.yMax); y++) {
      if (y === 0) continue;
      const sy = this.worldToScreenY(y);
      ctx.beginPath();
      ctx.moveTo(this.worldToScreenX(0) - 5, sy);
      ctx.lineTo(this.worldToScreenX(0) + 5, sy);
      ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(y.toString(), this.worldToScreenX(0) - 10, sy + 5);
    }
  }
  
  drawTrueTransform() {
    const ctx = this.ctx;
    ctx.strokeStyle = '#FFD700';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    ctx.globalAlpha = 0.6;
    
    ctx.beginPath();
    let first = true;
    for (let x = this.xMin; x <= this.xMax; x += 0.05) {
      const y = this.applyFunction(x);
      const sx = this.worldToScreenX(x);
      const sy = this.worldToScreenY(y);
      
      if (first) {
        ctx.moveTo(sx, sy);
        first = false;
      } else {
        ctx.lineTo(sx, sy);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1.0;
  }
  
  drawInputDistribution() {
    const ctx = this.ctx;
    
    // Draw Gaussian curve (sideways, along Y axis at current mean)
    ctx.strokeStyle = '#4169E1';
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.8;
    
    const sx = this.worldToScreenX(this.mean);
    const maxPdf = this.gaussian(this.mean, this.mean, this.variance);
    const scale = 50; // Scaling factor for visualization
    
    ctx.beginPath();
    let first = true;
    for (let y = this.yMin; y <= this.yMax; y += 0.1) {
      // Gaussian in X, evaluated at different Y heights (just for viz)
      const pdf = this.gaussian(this.mean, this.mean, this.variance);
      const offset = (pdf / maxPdf) * scale;
      const sy = this.worldToScreenY(y);
      
      if (first) {
        ctx.moveTo(sx - offset, sy);
        first = false;
      } else {
        ctx.lineTo(sx - offset, sy);
      }
    }
    
    // Close the shape
    for (let y = this.yMax; y >= this.yMin; y -= 0.1) {
      const pdf = this.gaussian(this.mean, this.mean, this.variance);
      const offset = (pdf / maxPdf) * scale;
      const sy = this.worldToScreenY(y);
      ctx.lineTo(sx + offset, sy);
    }
    ctx.closePath();
    
    ctx.fillStyle = 'rgba(65, 105, 225, 0.2)';
    ctx.fill();
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }
  
  drawSigmaPoints(sigmaPoints) {
    const ctx = this.ctx;
    
    sigmaPoints.forEach(sp => {
      const sx = this.worldToScreenX(sp.x);
      
      // Draw vertical line through function
      ctx.strokeStyle = 'rgba(50, 205, 50, 0.3)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx, this.padding);
      ctx.lineTo(sx, this.height - this.padding);
      ctx.stroke();
      
      // Draw point at y=0
      ctx.fillStyle = '#32CD32';
      ctx.beginPath();
      ctx.arc(sx, this.worldToScreenY(0), 6, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw point on function curve
      const y = this.applyFunction(sp.x);
      const sy = this.worldToScreenY(y);
      ctx.fillStyle = '#32CD32';
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, 2 * Math.PI);
      ctx.fill();
    });
  }
  
  drawTransformedSigmaPoints(transformed) {
    const ctx = this.ctx;
    
    transformed.forEach(sp => {
      // Just mark the points (already drawn by drawSigmaPoints)
      // We could add labels or emphasis here if needed
    });
  }
  
  drawOutputDistribution(outputMean, outputVariance) {
    const ctx = this.ctx;
    
    // Draw output Gaussian (horizontal, at output mean height)
    ctx.strokeStyle = '#FF6347';
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.8;
    
    const sy = this.worldToScreenY(outputMean);
    const outputStd = Math.sqrt(outputVariance);
    const maxPdf = this.gaussian(outputMean, outputMean, outputVariance);
    const scale = 50;
    
    ctx.beginPath();
    let first = true;
    for (let x = this.xMin; x <= this.xMax; x += 0.1) {
      const pdf = this.gaussian(outputMean, outputMean, outputVariance);
      const offset = (pdf / maxPdf) * scale;
      const sx = this.worldToScreenX(x);
      
      if (first) {
        ctx.moveTo(sx, sy - offset);
        first = false;
      } else {
        ctx.lineTo(sx, sy - offset);
      }
    }
    
    // Close the shape
    for (let x = this.xMax; x >= this.xMin; x -= 0.1) {
      const pdf = this.gaussian(outputMean, outputMean, outputVariance);
      const offset = (pdf / maxPdf) * scale;
      const sx = this.worldToScreenX(x);
      ctx.lineTo(sx, sy + offset);
    }
    ctx.closePath();
    
    ctx.fillStyle = 'rgba(255, 99, 71, 0.2)';
    ctx.fill();
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }
  
  updateStats() {
    const sigmaPoints = this.generateSigmaPoints();
    const { outputMean, outputVariance } = this.transformSigmaPoints(sigmaPoints);
    
    document.getElementById('inputMean').textContent = this.mean.toFixed(2);
    document.getElementById('inputVar').textContent = this.variance.toFixed(2);
    document.getElementById('outputMean').textContent = outputMean.toFixed(2);
    document.getElementById('outputVar').textContent = outputVariance.toFixed(2);
  }
  
  setVariance(v) {
    this.variance = v;
    this.draw();
    this.updateStats();
  }
  
  setFunction(f) {
    this.function = f;
    this.draw();
    this.updateStats();
  }
  
  reset() {
    this.mean = 0.0;
    this.variance = 1.0;
    this.draw();
    this.updateStats();
  }
}

// ============================================================================
// ALGORITHM STEPS VISUALIZATION
// ============================================================================

class AlgorithmStepsViz {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    
    this.width = this.canvas.width;
    this.height = this.canvas.height;
    
    // Simulation state
    this.step = 0;
    this.state = 0.0;
    this.covariance = 1.0;
    this.measurement = null;
    this.animationId = null;
    this.isPlaying = false;
    
    // History for plotting
    this.trueStates = [];
    this.estimates = [];
    this.measurements = [];
    this.measurementSteps = []; // Track which step each measurement was taken
    this.uncertainties = [];
    
    this.reset();
    this.draw();
  }
  
  reset() {
    this.step = 0;
    this.state = 0.0;
    this.covariance = 1.0;
    this.trueStates = [0.0];
    this.estimates = [0.0];
    this.measurements = [];
    this.measurementSteps = [];
    this.uncertainties = [this.covariance];
    this.updateStepDisplay();
  }
  
  nextStep() {
    const steps = [
      () => this.stepPredict(),
      () => this.stepMeasure(),
      () => this.stepUpdate(),
    ];
    
    const stepIndex = this.step % steps.length;
    steps[stepIndex]();
    this.step++;
    
    this.draw();
    this.updateStepDisplay();
  }
  
  stepPredict() {
    // Simple process model: x' = x + v, where v ~ N(0, 0.1)
    const processNoise = 0.1;
    const trueVelocity = 0.5;
    
    // True state evolution
    const trueState = this.trueStates[this.trueStates.length - 1] + trueVelocity;
    this.trueStates.push(trueState);
    
    // Predicted state
    this.state = this.estimates[this.estimates.length - 1] + 0.5; // Assume known velocity
    this.covariance = this.covariance + processNoise;
    
    this.estimates.push(this.state);
    this.uncertainties.push(this.covariance);
    
    this.updateExplanation('Predict', 
      'Predict next state using process model. Uncertainty grows due to process noise.');
  }
  
  stepMeasure() {
    // Generate noisy measurement of true state
    const measurementNoise = 0.5;
    const trueState = this.trueStates[this.trueStates.length - 1];
    this.measurement = trueState + (Math.random() - 0.5) * 2 * measurementNoise;
    this.measurements.push(this.measurement);
    this.measurementSteps.push(this.trueStates.length - 1); // Record which step this measurement is for
    
    this.updateExplanation('Measure',
      `Received measurement: ${this.measurement.toFixed(2)}. ` +
      'Now we can correct our prediction using this observation.');
  }
  
  stepUpdate() {
    // Kalman update (simplified 1D case)
    const measurementNoise = 0.5;
    const R = measurementNoise * measurementNoise;
    
    // Kalman gain
    const K = this.covariance / (this.covariance + R);
    
    // Update state
    const innovation = this.measurement - this.state;
    this.state = this.state + K * innovation;
    
    // Update covariance
    this.covariance = (1 - K) * this.covariance;
    
    // Record
    this.estimates[this.estimates.length - 1] = this.state;
    this.uncertainties[this.uncertainties.length - 1] = this.covariance;
    
    this.updateExplanation('Update',
      `Corrected estimate using measurement. Kalman gain K=${K.toFixed(2)}. ` +
      'Uncertainty decreased after incorporating measurement.');
  }
  
  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    
    const animate = () => {
      this.nextStep();
      
      if (this.step < 30 && this.isPlaying) { // Run for 10 cycles (30 steps)
        this.animationId = setTimeout(animate, 500);
      } else {
        this.isPlaying = false;
      }
    };
    
    animate();
  }
  
  stop() {
    this.isPlaying = false;
    if (this.animationId) {
      clearTimeout(this.animationId);
      this.animationId = null;
    }
  }
  
  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    
    if (this.estimates.length === 0) return;
    
    const padding = 60;
    const plotWidth = this.width - 2 * padding;
    const plotHeight = this.height - 2 * padding;
    
    // Compute Y range
    const allValues = [...this.trueStates, ...this.estimates, ...this.measurements];
    const yMin = Math.min(...allValues) - 1;
    const yMax = Math.max(...allValues) + 1;
    const xMax = this.trueStates.length - 1;
    
    const toScreenX = (x) => padding + (x / Math.max(1, xMax)) * plotWidth;
    const toScreenY = (y) => this.height - padding - ((y - yMin) / (yMax - yMin)) * plotHeight;
    
    // Draw axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, toScreenY(yMin));
    ctx.lineTo(padding, toScreenY(yMax));
    ctx.moveTo(padding, this.height - padding);
    ctx.lineTo(this.width - padding, this.height - padding);
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Time Step', this.width / 2, this.height - 20);
    ctx.save();
    ctx.translate(20, this.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('State Value', 0, 0);
    ctx.restore();
    
    // Draw uncertainty band
    ctx.fillStyle = 'rgba(65, 105, 225, 0.2)';
    ctx.beginPath();
    for (let i = 0; i < this.estimates.length; i++) {
      const x = toScreenX(i);
      const y = toScreenY(this.estimates[i]);
      const uncertainty = Math.sqrt(this.uncertainties[i]);
      if (i === 0) {
        ctx.moveTo(x, y - uncertainty * 50);
      } else {
        ctx.lineTo(x, y - uncertainty * 50);
      }
    }
    for (let i = this.estimates.length - 1; i >= 0; i--) {
      const x = toScreenX(i);
      const y = toScreenY(this.estimates[i]);
      const uncertainty = Math.sqrt(this.uncertainties[i]);
      ctx.lineTo(x, y + uncertainty * 50);
    }
    ctx.closePath();
    ctx.fill();
    
    // Draw true state
    ctx.strokeStyle = '#32CD32';
    ctx.lineWidth = 3;
    ctx.beginPath();
    this.trueStates.forEach((val, i) => {
      const x = toScreenX(i);
      const y = toScreenY(val);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // Draw estimates
    ctx.strokeStyle = '#4169E1';
    ctx.lineWidth = 3;
    ctx.beginPath();
    this.estimates.forEach((val, i) => {
      const x = toScreenX(i);
      const y = toScreenY(val);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    
    // Draw measurements
    ctx.fillStyle = '#FF6347';
    this.measurements.forEach((val, i) => {
      const stepIndex = this.measurementSteps[i];
      const x = toScreenX(stepIndex);
      const y = toScreenY(val);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
    
    // Legend
    const legendY = padding - 10;
    ctx.font = '12px Arial';
    ctx.fillStyle = '#32CD32';
    ctx.fillRect(padding, legendY, 20, 3);
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText('True', padding + 25, legendY + 3);
    
    ctx.fillStyle = '#4169E1';
    ctx.fillRect(padding + 100, legendY, 20, 3);
    ctx.fillStyle = '#fff';
    ctx.fillText('Estimate', padding + 125, legendY + 3);
    
    ctx.fillStyle = '#FF6347';
    ctx.beginPath();
    ctx.arc(padding + 220, legendY + 1, 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText('Measurements', padding + 230, legendY + 3);
  }
  
  updateStepDisplay() {
    const cycle = Math.floor(this.step / 3);
    const subStep = (this.step % 3) + 1;
    document.getElementById('currentStep').textContent = `${cycle}.${subStep}`;
    document.getElementById('stateEstimate').textContent = this.state.toFixed(2);
    document.getElementById('stateUncertainty').textContent = Math.sqrt(this.covariance).toFixed(2);
  }
  
  updateExplanation(title, text) {
    const el = document.getElementById('stepExplanation');
    el.innerHTML = `<h3>${title}</h3><p>${text}</p>`;
  }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

let sigmaViz, stepsViz;

document.addEventListener('DOMContentLoaded', () => {
  // Initialize visualizations
  sigmaViz = new SigmaPointsViz('sigmaCanvas');
  stepsViz = new AlgorithmStepsViz('stepsCanvas');
  
  // Sigma points controls
  document.getElementById('functionSelect').addEventListener('change', (e) => {
    sigmaViz.setFunction(e.target.value);
  });
  
  document.getElementById('uncertaintySlider').addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    document.getElementById('uncertaintyValue').textContent = value.toFixed(1);
    sigmaViz.setVariance(value * value); // Convert std to variance
  });
  
  document.getElementById('resetBtn').addEventListener('click', () => {
    sigmaViz.reset();
    document.getElementById('uncertaintySlider').value = 1.0;
    document.getElementById('uncertaintyValue').textContent = '1.0';
  });
  
  // Algorithm steps controls
  document.getElementById('stepBtn').addEventListener('click', () => {
    stepsViz.nextStep();
  });
  
  document.getElementById('playBtn').addEventListener('click', () => {
    const btn = document.getElementById('playBtn');
    if (stepsViz.isPlaying) {
      stepsViz.stop();
      btn.textContent = '▶ Animate';
    } else {
      stepsViz.play();
      btn.textContent = '⏸ Pause';
    }
  });
  
  document.getElementById('resetStepsBtn').addEventListener('click', () => {
    stepsViz.stop();
    stepsViz.reset();
    stepsViz.draw();
    document.getElementById('playBtn').textContent = '▶ Animate';
  });
  
  // Initial stats update
  sigmaViz.updateStats();
});
