# Epigenetic Drift and Phenotype Prediction

This project involves research a population of genetically identical organisms (clones) under varying environmental conditions. While their DNA sequences are identical, subtle epigenetic modifications (specifically, DNA methylation patterns) differ between individuals and influence their phenotype (observable characteristics). Below is a detailed description of the problem, methodology, and requirements.

---

## Problem Description

### Data Overview

1. **Methylation Data Representation:**
   - DNA methylation is measured at thousands of Cytosine-Guanine (CpG) dinucleotides across the genome.
   - Each CpG site can be in a methylated (1) or unmethylated (0) state.
   - Input: `M` (sparse matrix, `n_individuals x n_cpg_sites`), where each entry represents the methylation state.

2. **Phenotype Data:**
   - A noisy dataset of measured phenotypes (`P`), represented as a vector (`n_individuals x 1`).

3. **Epigenetic Drift Model:**
   - Epigenetic changes are modeled probabilistically over time. Transition probabilities are defined as:
     - `P(M_i(t+Δt) = 1 | M_i(t) = 0) = α*Δt + O(Δt^2)`
     - `P(M_i(t+Δt) = 0 | M_i(t) = 1) = β*Δt + O(Δt^2)`
   - Here, `α = f(site_index, environmental_variable)` and `β = g(site_index, environmental_variable)` are non-homogeneous functions dependent on CpG site and environmental variables.

### Custom Distance Metric

To compute distances between individuals, we use a custom formula:

\[
d(x, y, e_x, e_y) = \sqrt{\sum_i (w_i \cdot |x_i - y_i|^p) + \gamma \cdot |e_x - e_y|}
\]

- `x`, `y`: Methylation state vectors of two individuals.
- `e_x`, `e_y`: Respective environmental variable values.
- Hyperparameters:
  - `p`: Power value.
  - `w_i`: Weight for CpG site `i`.
  - `\gamma`: Scaling factor for environmental condition differences.

---

## Tasks

### 1. Predict Methylation State at Time `T`

Simulate the methylation states over time using the epigenetic drift model. Implement a finite difference method to calculate transitions in small time intervals.

- Input:
  - Initial methylation matrix `M`.
  - Drift rates `α = f(site_index, environmental_variable)` and `β = g(site_index, environmental_variable)`.
  - Time step `Δt` and total time `T`.

### 2. Phenotype Prediction Using Non-Parametric Approach

Predict the phenotype of a new individual based on their methylation profile and environmental condition.

- Approach:
  - Use the custom distance metric to identify individuals close to the new sample.
  - Employ a kernelized regression or another non-parametric method to predict phenotype based on proximity.

### 3. Model Evaluation

- Split data into training, validation, and test sets.
- Evaluate model performance using appropriate metrics, such as Root Mean Squared Error (RMSE).

---

## Requirements

### General Constraints

1. **No Neural Networks:**
   - The prediction model must be non-neural-network-based.

2. **Sparse Matrix Handling:**
   - Efficiently handle the sparse methylation matrix `M` to avoid memory issues.

3. **Efficient Simulation:**
   - Simulate methylation state transitions efficiently over large CpG sites.

4. **Hyperparameter Tuning:**
   - Choose or optimize hyperparameters (`p`, `w_i`, `\gamma`).

5. **Code Clarity and Documentation:**
   - Ensure clear, well-structured, and documented code.

### Input Example

```python
import numpy as np
import scipy.sparse as sparse

# Example data generation (replace with your actual data loading)
n_individuals = 100
n_cpg_sites = 1000
T = 10  # Total simulation time
delta_t = 0.01  # Step for simulation
p = 2  # Power value in the distance function
gamma = 0.5  # Gamma value in the distance function

M = sparse.random(n_individuals, n_cpg_sites, density=0.1, format="csr")
P = np.random.randn(n_individuals)
E = np.random.rand(n_individuals)  # Environmental variable for each individual

def f(site_index, environmental_variable):
    return 0.01 + 0.001 * (site_index % 10) + 0.001 * environmental_variable  # Example function

def g(site_index, environmental_variable):
    return 0.005 + 0.0005 * (site_index % 10) + 0.0005 * environmental_variable  # Example function
```

---

## Implementation Steps

### 1. Simulate Methylation States

- Use finite difference methods to iteratively update methylation states based on the drift model.
- Transition rates depend on `f` and `g`.

### 2. Custom Distance-Based Phenotype Prediction

- Compute distances between individuals using the custom metric.
- Use distance-weighted predictions for phenotype estimation.

### 3. Model Validation

- Perform data splitting for training, validation, and testing.
- Evaluate the model using RMSE or similar metrics.

---

## Evaluation Metrics

- **Root Mean Squared Error (RMSE):** To assess prediction accuracy.

---

## Notes

- Ensure modular, reusable code.
- Incorporate hyperparameter optimization methods (e.g., grid search or random search).

---
