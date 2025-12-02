# ⚛️ Harmonic Oscillator PINN  
### _A Physics-Informed Neural Network (PINN) Solving the Harmonic Oscillator ODE using Python + Rust_

This project demonstrates a **Physics-Informed Neural Network (PINN)** that learns the dynamics of a **simple harmonic oscillator** using:

- Sparse & noisy displacement measurements  
- The governing differential equation  
- Automatic differentiation (via PyTorch)  
- A Rust inference pipeline (via TorchScript + `tch`)  

It provides a clean, modern example of combining **scientific machine learning (SciML)** with **Rust’s high-performance safety**.

---

## 🔬 1. The Physical Problem

The classical **simple harmonic oscillator** satisfies:

\[
\frac{d^2x}{dt^2} + \omega^2 x = 0
\]

with closed-form solution:

\[
x(t) = A \cos(\omega t) + B \sin(\omega t)
\]

In this project:

- Angular frequency:  
  \[
  \omega = 2.0
  \]
- True coefficients:  
  \[
  A=1.0,\quad B=0.5
  \]

We provide the model only **10 noisy measurements** of \( x(t) \).  
The rest is learned through physics constraints.

---

## 🧠 2. What is a Physics-Informed Neural Network?

A PINN is trained by minimizing **two losses**:

### **1. Data Loss**
Fits observed points:

\[
\mathcal{L}_{\text{data}} = 
\frac{1}{N}\sum_{i=1}^N \left( x_\theta(t_i) - x_{\text{obs}}(t_i) \right)^2
\]

### **2. Physics Loss**
Enforces the ODE using autograd to compute derivatives:

\[
x_t = \frac{dx}{dt}, \quad x_{tt}=\frac{d^2 x}{dt^2}
\]

\[
\mathcal{L}_{\text{phys}} =
\frac{1}{M}\sum_{j=1}^M 
\left( x_{tt}(t_j) + \omega^2 x(t_j) \right)^2
\]

### **Total Loss**
\[
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{phys}}
\]

---

## 🏗️ 3. Architecture Overview

### **Python (Training)**
- Fully connected MLP (Tanh activations)  
- 200 collocation points for enforcing physics  
- TorchScript export for Rust  

### **Rust (Inference)**
- Uses the `tch` crate  
- Loads TorchScript model  
- Runs fast inference in native Rust  
- Suitable for real-time or embedded physics simulation  

This is a modern workflow:  
**Train with Python → Deploy with Rust**.

---

## 📁 4. Project Structure