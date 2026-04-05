# SCDAA Coursework 2025–26  
## Policy Iteration with Deep Galerkin Method for LQR

---

## 1. Group Information

- Name 1: Yufan Zhang (Student ID: s2803912)  
- Name 2: Liyi Li (Student ID: s2905691)  
- Name 3: Xinyue Zhao (Student ID: s2846397)  

Contribution:
- Equal contribution (1/3 each)  

---

## 2. Overview

This repository contains the implementation for all exercises in the SCDAA coursework.

The objective is to solve a stochastic control problem using:
- Linear Quadratic Regulator (LQR)
- Neural networks (FFN and DGM-style networks)
- Deep Galerkin Method (DGM)
- Policy Iteration Algorithm (PIA)

The workflow follows:
1. Compute exact solution using Riccati equation (Exercise 1)
2. Verify neural networks via supervised learning (Exercise 2)
3. Solve PDE using Deep Galerkin Method (Exercise 3)
4. Combine both into policy iteration (Exercise 4)

Note: Since Monte Carlo simulations are used in parts of the implementation, the results are inherently stochastic. Therefore, the generated figures may exhibit slight variations across different runs. However, the overall trends and conclusions remain consistent.

---

## 3. Requirements

This project uses:

- Python 3.x
- numpy
- scipy
- matplotlib
- torch

Install dependencies (if needed):

```bash
pip install numpy scipy matplotlib torch
```

---

## 4. File Structure

lib/                        

EX.1_1.py                   # Riccati solver (Exercise 1.1)
EX.1_2.ipynb                # Monte Carlo verification (Exercise 1.2)

EX1.2_error_vs_monte_carlo_samples.png
EX1.2_error_vs_time_discretisation.png

EX.2_1&2_2.ipynb            # Supervised learning (value + control)

EX.3.ipynb                  # Deep Galerkin method 

EX3_mc_error.png
EX3_training_loss.png

EX.4.ipynb                  # Policy iteration 

fig1_value_comparison.png   
fig2_control_comparison.png 
fig3_minimum_objective_history.png
fig4_dgm_style_training_loss.png
fig5_dgm_style_value_error.png

---

## 5. How to Run the Code

### Exercise 1

Run:
EX.1_1.py



Then open:
EX.1_2.ipynb



This will:

- simulate the SDE
- generate convergence plots (log-log error)

---

### Exercise 2

Open and run:
EX.2_1&2_2.ipynb



This will:

- train value network (DGM-style)
- train control network (FFN)
- produce training loss plots

---

### Exercise 3

Open and run:
EX.3.ipynb



This will:

- implement Deep Galerkin Method
- solve the linear PDE
- generate:
  - training loss
  - error vs Monte Carlo

---

### Exercise 4 (Main Results)

Open and run:
EX.4.ipynb



This will:

- implement policy iteration
- train both networks iteratively
- generate final plots

#### Main figures

- `fig1_value_comparison.png`
- `fig2_control_comparison.png`

#### Diagnostic figures

- `fig3_minimum_objective_history.png`
- `fig4_dgm_style_training_loss.png`
- `fig5_dgm_style_value_error.png`

---

## 6. Reproducing the Report Figures

To reproduce all figures used in the report:

1. Run all notebooks in order:
   - EX.1_2.ipynb
   - EX.2_1&2_2.ipynb
   - EX.3.ipynb
   - EX.4.ipynb

2. Figures for Exercise 4 will be saved as `.png` files in the root directory.

3. Other figures (Exercises 1–3) are generated inside the notebooks.

---

## 7. Notes

- The implementation strictly follows coursework requirements:
  - only uses numpy, scipy, matplotlib, torch
- No external ML libraries are used
- Neural networks are implemented based on:
  - FFN (for control)
  - DGM-style network (for value function)

---

## 8. Key Result

The final implementation demonstrates that:

- The value function converges to the Riccati solution
- The control converges to the optimal feedback law
- Policy iteration combined with DGM successfully solves the HJB equation

---

## 9. References

- M. Sabate-Vidales, Deep-PDE-Solvers, Github project,
  https://github.com/msabvid/Deep-PDE-Solvers,2021

- C. Jiang, Deep Galerkin Method, Github project,
  https://github.com/EurasianEagleOwl/DeepGalerkinMethod,2022

- G. dos Reis, J. Li, and D. Siska.
  Stochastic Control and Dynamic Asset Allocation lecture notes, 2026.

