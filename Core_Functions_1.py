import numpy as np
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from tqdm import tqdm

def simulate_methylation_drift(M_initial, f, g, T, delta_t, E):
    """
    Simulates methylation drift over time for each CpG site.

    Args:
        M_initial (sparse matrix): Initial methylation matrix (n_individuals x n_cpg_sites).
        f (function): Function to calculate alpha (drift towards methylation), takes site_index and environment as input.
        g (function): Function to calculate beta (drift towards unmethylation), takes site_index and environment as input.
        T (float): Total simulation time.
        delta_t (float): Time step for simulation.
        E (array): Enviromental variable for each individual.
    Returns:
        sparse matrix: Methylation matrix at time T.
    """

    M_current = M_initial.copy()
    n_individuals, n_cpg_sites = M_current.shape
    
    num_steps = int(T / delta_t)

    for _ in tqdm(range(num_steps), desc="Simulating Drift"):
      M_next = M_current.copy()
      for i in range(n_individuals):
        for j in range(n_cpg_sites):
            current_state = M_current[i, j]
            if current_state == 0:
                alpha = f(j, E[i])
                prob_methylate = alpha * delta_t
                if np.random.rand() < prob_methylate:
                    M_next[i, j] = 1
            else:
                beta = g(j, E[i])
                prob_unmethylate = beta * delta_t
                if np.random.rand() < prob_unmethylate:
                     M_next[i, j] = 0

      M_current = M_next
    return M_current


def custom_distance(x, y, e_x, e_y, p, weights, gamma):
    """
    Calculates the custom distance between two individuals based on methylation state and environment.
    Args:
        x (np.array): Methylation state vector of individual x.
        y (np.array): Methylation state vector of individual y.
        e_x (float): Environmental variable of individual x.
        e_y (float): Environmental variable of individual y.
        p (float): Power value.
        weights (np.array): CpG site weights.
        gamma (float): Scaling factor for environmental differences.

    Returns:
        float: The calculated distance.
    """
    methylation_diff = np.abs(x - y) ** p  # Corrected this line!
    weighted_diff = weights * methylation_diff
    distance = np.sqrt(np.sum(weighted_diff) + gamma * (e_x - e_y)**2)
    return distance


def predict_phenotype(M_train, P_train, E_train, M_test, E_test, p, weights, gamma, k = 5): # Don't forget use the top K
    n_test = M_test.shape[0]
    predictions = np.zeros(n_test)

    M_train_dense = M_train.toarray()
    M_test_dense = M_test.toarray()

    for i in range(n_test):
      distances = []
      for j in range(M_train_dense.shape[0]):
        dist = custom_distance(M_test_dense[i,:], M_train_dense[j,:], E_test[i], E_train[j], p, weights, gamma)
        distances.append((dist,j))
      
      distances.sort(key=lambda x: x[0])
      
      weighted_sum = 0
      total_weight = 0
      for d,j in distances[:k]:
          weight = np.exp(-d) # you can chose any weight based on your requirement
          weighted_sum += P_train[j] * weight
          total_weight += weight

      if total_weight > 0:
          predictions[i] = weighted_sum/total_weight
      else:
          predictions[i] = np.mean(P_train)

    return predictions

# Example Usage and Evaluation:
if __name__ == '__main__':
    # Example data generation (replace with your actual data loading)
    n_individuals = 100
    n_cpg_sites = 1000
    T = 10 # total simulation time
    delta_t = 0.01 # step for simulation
    p = 2 # Power value in the distance function
    gamma = 0.5 # Gamma value in the distance function

    M = sparse.random(n_individuals, n_cpg_sites, density=0.1, format="csr")
    P = np.random.randn(n_individuals)
    E = np.random.rand(n_individuals) # Environmental variable for each individual


    def f(site_index, environmental_variable):
        return 0.01 + 0.001 * (site_index % 10) + 0.001 * environmental_variable  # Example function
    def g(site_index, environmental_variable):
        return 0.005 + 0.0005 * (site_index % 10) + 0.0005 * environmental_variable  # Example function

    # Split data into train, validation, and test sets
    M_train_val, M_test, P_train_val, P_test, E_train_val, E_test = train_test_split(M, P, E, test_size=0.2, random_state=42)
    M_train, M_val, P_train, P_val, E_train, E_val = train_test_split(M_train_val, P_train_val, E_train_val, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

    # Simulate methylation drift
    M_train_T = simulate_methylation_drift(M_train, f, g, T, delta_t, E_train)
    M_val_T = simulate_methylation_drift(M_val, f, g, T, delta_t, E_val)
    M_test_T = simulate_methylation_drift(M_test, f, g, T, delta_t, E_test)

    # Hyperparameter selection
    best_rmse = float('inf')
    best_weights = None
    best_k = None

    n_cpg_sites = M_train_T.shape[1]
    weights_options = [np.ones(n_cpg_sites)/n_cpg_sites , np.random.rand(n_cpg_sites)]
    k_options = [3,5,7]
    for weights in weights_options:
      for k in k_options:
        P_val_pred = predict_phenotype(M_train_T, P_train, E_train, M_val_T, E_val, p, weights, gamma, k)
        rmse = np.sqrt(mean_squared_error(P_val, P_val_pred))
        print(f"RMSE on validation data = {rmse}, weights= {weights}, k={k}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights
            best_k = k

    print(f"Best validation RMSE: {best_rmse}, best weights: {best_weights}, best k: {best_k}")
    
    # Make prediction on test set
    P_test_pred = predict_phenotype(M_train_T, P_train, E_train, M_test_T, E_test, p, best_weights, gamma, best_k)

    # Evaluate the model on the test set
    rmse = np.sqrt(mean_squared_error(P_test, P_test_pred))
    print(f"Test RMSE: {rmse}")
    print("\nTop 5 Predicted vs. True Phenotypes (Test Set):")
    for i in range(min(5, len(P_test))):
        print(f"  Individual {i+1}: Predicted = {P_test_pred[i]:.4f}, True = {P_test[i]:.4f}")
