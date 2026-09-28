"""Scientific analysis metrics for spatial topology, modularity, and phase synchronization.

Implements the formal metrics specified in:
- docs/PHASE_4_IMPLEMENTATION_REPORT.md
- research/SPATIAL_TOPOLOGY_REVIEW.md
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_structural_modularity(
    adjacency_matrix: np.ndarray, module_assignments: np.ndarray
) -> float:
    """Compute Newman's structural modularity Q_struct for a directed or undirected network.

    Q = (1 / 2m) * sum_{i,j} [ A_ij - (k_i^out * k_j^in) / (2m) ] * delta(c_i, c_j)
    """
    A = (adjacency_matrix > 0).astype(np.float64)
    m = np.sum(A)
    if m == 0:
        return 0.0

    k_out = np.sum(A, axis=1)
    k_in = np.sum(A, axis=0)
    N = len(module_assignments)

    # Indicator matrix for module co-membership: delta_c[i, j] = 1 if c_i == c_j else 0
    delta_c = (module_assignments[:, np.newaxis] == module_assignments[np.newaxis, :]).astype(np.float64)

    expected = np.outer(k_out, k_in) / m
    B = A - expected
    Q = np.sum(B * delta_c) / m
    return float(Q)


def compute_functional_modularity(
    activations_time_series: np.ndarray, module_assignments: np.ndarray, threshold: float = 0.0
) -> Tuple[float, np.ndarray]:
    """Compute functional modularity Q_func based on activation cross-correlation matrix.

    Args:
        activations_time_series: Array of shape [T, N] representing unit activations over time.
        module_assignments: Array of shape [N] with module indices.
        threshold: Minimum correlation threshold to include as functional edge.

    Returns:
        Q_func: Functional modularity score in [-0.5, 1.0].
        corr_matrix: Pairwise Pearson correlation matrix [N, N].
    """
    T, N = activations_time_series.shape
    if T < 2 or N < 2:
        return 0.0, np.zeros((N, N), dtype=np.float32)

    # Standardize time series per unit
    stds = np.std(activations_time_series, axis=0) + 1e-8
    means = np.mean(activations_time_series, axis=0)
    normed = (activations_time_series - means) / stds

    # Pearson correlation matrix
    corr = np.dot(normed.T, normed) / (T - 1.0)
    np.fill_diagonal(corr, 0.0)

    # Positive correlation functional network
    pos_corr = np.maximum(0.0, corr)
    if threshold > 0.0:
        pos_corr[pos_corr < threshold] = 0.0

    total_weight = np.sum(pos_corr)
    if total_weight < 1e-6:
        return 0.0, corr.astype(np.float32)

    k = np.sum(pos_corr, axis=1)
    expected = np.outer(k, k) / total_weight
    delta_c = (module_assignments[:, np.newaxis] == module_assignments[np.newaxis, :]).astype(np.float64)

    Q_func = np.sum((pos_corr - expected) * delta_c) / total_weight
    return float(Q_func), corr.astype(np.float32)


def compute_within_and_cross_metrics(
    adjacency_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    module_assignments: np.ndarray,
) -> Dict[str, float]:
    """Calculate within-module vs cross-module connection density and activation correlations."""
    N = len(module_assignments)
    same_module = (module_assignments[:, np.newaxis] == module_assignments[np.newaxis, :])
    diff_module = ~same_module
    np.fill_diagonal(same_module, False)
    np.fill_diagonal(diff_module, False)

    # 1. Structural densities
    A = (adjacency_matrix > 0).astype(np.float32)
    within_edges = np.sum(A[same_module])
    cross_edges = np.sum(A[diff_module])

    n_within_pairs = np.sum(same_module)
    n_cross_pairs = np.sum(diff_module)

    within_conn_density = float(within_edges / n_within_pairs) if n_within_pairs > 0 else 0.0
    cross_conn_density = float(cross_edges / n_cross_pairs) if n_cross_pairs > 0 else 0.0

    # 2. Functional correlations
    within_corrs = corr_matrix[same_module]
    cross_corrs = corr_matrix[diff_module]

    mean_within_corr = float(np.mean(within_corrs)) if len(within_corrs) > 0 else 0.0
    mean_cross_corr = float(np.mean(cross_corrs)) if len(cross_corrs) > 0 else 0.0

    return {
        "within_connection_density": round(within_conn_density, 4),
        "cross_connection_density": round(cross_conn_density, 4),
        "mean_within_correlation": round(mean_within_corr, 4),
        "mean_cross_correlation": round(mean_cross_corr, 4),
        "functional_segregation_ratio": round(
            (mean_within_corr - mean_cross_corr) / (abs(mean_cross_corr) + 1e-4), 3
        ),
    }


def compute_participation_coefficient(
    connectivity_matrix: np.ndarray, module_assignments: np.ndarray
) -> np.ndarray:
    """Compute Guimerà & Amaral's participation coefficient for each node.

    P_i = 1 - sum_{m} (k_im / k_i)^2
    Measures whether a unit's connections are purely local (P_i ~ 0) or uniformly distributed (P_i ~ 1).
    """
    W = np.abs(connectivity_matrix)
    k_total = np.sum(W, axis=1) + 1e-8
    unique_modules = np.unique(module_assignments)

    P = np.ones(len(module_assignments), dtype=np.float32)
    for m in unique_modules:
        m_mask = (module_assignments == m)
        k_im = np.sum(W[:, m_mask], axis=1)
        P -= (k_im / k_total) ** 2

    return np.maximum(0.0, P)


def compute_phase_synchronization(activations_time_series: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute Kuramoto phase synchronization order parameter R(t) over time.

    Derives instantaneous phase theta_i(t) from state trajectory velocities:
        theta_i(t) = atan2(da_i/dt, a_i(t) - mean(a_i))
        R(t) = (1 / N) * |sum_{j=1}^N exp(i * theta_j(t))|

    Returns:
        mean_R: Mean phase coherence across time in [0.0, 1.0].
        R_t: Coherence order parameter at each time step [T].
    """
    T, N = activations_time_series.shape
    if T < 3 or N < 2:
        return 0.0, np.zeros(T, dtype=np.float32)

    # Compute numerical derivative da/dt
    da_dt = np.gradient(activations_time_series, axis=0)  # [T, N]
    centered_a = activations_time_series - np.mean(activations_time_series, axis=0, keepdims=True)

    phases = np.arctan2(da_dt, centered_a)  # [T, N]

    # Kuramoto order parameter R(t) = |(1/N) * sum_j e^{i * theta_j}|
    complex_order = np.mean(np.exp(1j * phases), axis=1)  # [T]
    R_t = np.abs(complex_order).astype(np.float32)

    mean_R = float(np.mean(R_t))
    return round(mean_R, 4), R_t


def compute_clustering_and_path_length(adjacency_matrix: np.ndarray) -> Dict[str, float]:
    """Compute average clustering coefficient and characteristic path length."""
    A = (adjacency_matrix > 0).astype(np.float32)
    N = len(A)
    if N < 3:
        return {"clustering_coefficient": 0.0, "characteristic_path_length": 1.0}

    # Symmetrized undirected graph for standard clustering
    A_sym = np.maximum(A, A.T)
    np.fill_diagonal(A_sym, 0.0)

    # Local clustering coefficient
    degrees = np.sum(A_sym, axis=1)
    A3 = np.linalg.matrix_power(A_sym, 3)
    triangles = np.diagonal(A3) / 2.0

    denom = degrees * (degrees - 1.0)
    valid = denom > 0
    clustering_per_node = np.zeros(N, dtype=np.float32)
    clustering_per_node[valid] = (2.0 * triangles[valid]) / denom[valid]
    avg_clustering = float(np.mean(clustering_per_node))

    # Approximate shortest path length via BFS (or Floyd-Warshall for moderate N)
    dist = np.full((N, N), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    dist[A_sym > 0] = 1.0

    # For fast calculation on moderate sizes (N <= 200)
    for k in range(N):
        dist = np.minimum(dist, dist[:, k, np.newaxis] + dist[np.newaxis, k, :])

    finite_paths = dist[np.isfinite(dist) & (dist > 0)]
    avg_path = float(np.mean(finite_paths)) if len(finite_paths) > 0 else float(N)

    return {
        "clustering_coefficient": round(avg_clustering, 4),
        "characteristic_path_length": round(avg_path, 3),
    }
