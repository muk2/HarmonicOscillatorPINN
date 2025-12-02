# ⚛️ Harmonic Oscillator PINN  
### _A Physics-Informed Neural Network (PINN) Solving the Harmonic Oscillator ODE using Python + Rust_

This project demonstrates a **Physics-Informed Neural Network (PINN)** that learns the dynamics of a **simple harmonic oscillator** using:

- Sparse & noisy displacement measurements  
- The governing differential equation  
- Automatic differentiation (via PyTorch)  
- A Rust inference pipeline (via TorchScript + `tch`)  

It’s a modern example of combining **scientific machine learning (SciML)** with **Rust’s performance & safety**.

---

## 🔬 1. The Physical Problem

The classical **simple harmonic oscillator** satisfies:

$$
\frac{d^2x}{dt^2} + \omega^2 x = 0
$$

with closed-form solution:

$$
x(t) = A \cos(\omega t) + B \sin(\omega t)
$$

In this project:

- Angular frequency:
  $$
  \omega = 2.0
  $$
- True coefficients:
  $$
  A = 1.0,\quad B = 0.5
  $$

We provide the model only **10 noisy measurements** of $x(t)$.  
The remaining structure is learned from physics.

---

## 🧠 2. What is a Physics-Informed Neural Network?

A PINN minimizes two losses:

### **1. Data Loss**

Fits observed data:

$$
\mathcal{L}_{\text{data}} = 
\frac{1}{N}\sum_{i=1}^N \left( x_\theta(t_i) - x_{\text{obs}}(t_i) \right)^2
$$

### **2. Physics Loss**

Uses autograd to compute:

- First derivative:  $x_t = \frac{dx}{dt}$
- Second derivative: $x_{tt} = \frac{d^2 x}{dt^2}$

PINN enforces the ODE:

$$
\mathcal{L}_{\text{phys}} =
\frac{1}{M}\sum_{j=1}^M 
\left( x_{tt}(t_j) + \omega^2 x(t_j) \right)^2
$$

### **Total Loss**

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{phys}}
$$

---

## 🏗️ 3. Architecture Overview

### **Python (Training)**

- Fully connected MLP (Tanh activations)  
- Samples 200 collocation points for enforcing physics  
- Exports TorchScript model for Rust inference  

### **Rust (Inference)**

- Uses the `tch` crate  
- Loads the TorchScript model in native Rust  
- Performs fast inference suitable for real-time physics simulation  

This workflow is:

**Train in Python → Infer in Rust**.

---

## 📁 4. Project Structure


```bash
HarmonicOscillatorPINN/
│
├── python/
│   ├── pinn_harmonic.py       # Train PINN and export TorchScript
│   ├── pinn_harmonic.pt       # Saved TorchScript model
│   └── pinn_result.png        # Plot of results
│
└── rust_infer/
├── src/main.rs            # Rust inference engine
└── Cargo.toml


