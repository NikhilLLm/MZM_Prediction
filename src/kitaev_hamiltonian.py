import numpy as np
import cupy as cp
from scipy import sparse
import os

def kitaev_hamiltonian(L, t, mu, delta, disorder_strength=0.0, local_potential=0.0):
    try:
        if L < 10 or t <= 0 or delta < 0:
            raise ValueError("Invalid parameters: L >= 10, t > 0, delta >= 0")
        N = 2 * L  # Spinless: 2 for particle-hole
        diag = np.zeros(N, dtype=complex)
        for i in range(L):
            disorder = disorder_strength * np.random.normal(0, 1) if disorder_strength > 0 else 0
            local = local_potential * np.exp(-((i-0)**2 + (i-(L-1))**2) / 10) if local_potential > 0 else 0
            diag[i] = -mu + disorder + local
            diag[i + L] = mu - disorder - local
        hop = -t * np.ones(L-1, dtype=complex)
        pair = delta * np.ones(L-1, dtype=complex)
        H = np.zeros((N, N), dtype=complex)
        np.fill_diagonal(H, diag)
        for i in range(L-1):
            H[i, i+1] = hop[i]
            H[i+1, i] = hop[i].conj()
            H[i+L, i+L+1] = -hop[i]
            H[i+L+1, i+L] = -hop[i].conj()
            H[i, i+L+1] = pair[i]
            H[i+L+1, i] = -pair[i].conj()
            H[i+1, i+L] = -pair[i]
            H[i+L, i+1] = pair[i].conj()
        return H
    except Exception as e:
        print(f"Error in kitaev_hamiltonian: {e}")
        return None

def make_kitaev_system(t, mu, delta, disorder_strength, local_potential, L):
    """Wrapper for compatibility with simulate_conductance.py."""
    return kitaev_hamiltonian(L, t, mu, delta, disorder_strength, local_potential)

def topological_invariant(t, mu):
    try:
        if t <= 0:
            raise ValueError("Invalid t: must be positive")
        return 2 * t
    except Exception as e:
        print(f"Error in topological_invariant: {e}")
        return None

def generate_hamiltonians(num_samples, L=200):
    try:
        np.random.seed(42)
        params = []
        labels = []
        class_labels = []  # 0: topological, 1: ABS-like, 2: large disorder
        t_range = [25, 31]
        delta_range = [0.1, 0.3]
        # Three classes: topological, ABS-like, large disorder
        samples_per_class = num_samples // 3
        remainder = num_samples % 3
        counts = [samples_per_class + (1 if i < remainder else 0) for i in range(3)]
        
        # Class 0: Topological phase (mu ensures topological, moderate disorder)
        mu_range_topo = [-50, 50]  # Ensures |mu| < 2t
        disorder_range_topo = [0.75, 2.25]
        for _ in range(counts[0]):
            t = float(cp.random.uniform(*t_range).get().item())
            mu = float(cp.random.uniform(*mu_range_topo).get().item())
            delta = float(cp.random.uniform(*delta_range).get().item())
            disorder = float(cp.random.uniform(*disorder_range_topo).get().item())
            mu_c = topological_invariant(t, mu)
            if mu_c is not None:
                params.append((t, mu, delta, disorder, 0.0))
                labels.append(mu_c)
                class_labels.append(0)  # Topological class
        
        # Class 1: ABS-like (trivial phase with disorder to induce ABS-like states)
        mu_range_abs = [-100, 100]  # We'll select mu to ensure trivial phase
        disorder_range_abs = [0.75, 2.25]
        for _ in range(counts[1]):
            t = float(cp.random.uniform(*t_range).get().item())
            mu_c = 2 * t  # Critical mu for this t
            # Ensure mu is in trivial phase: |mu| > 2t
            mu = float(cp.random.uniform(*mu_range_abs).get().item())
            while abs(mu) <= mu_c:  # Keep sampling until mu is in trivial phase
                mu = float(cp.random.uniform(*mu_range_abs).get().item())
            delta = float(cp.random.uniform(*delta_range).get().item())
            disorder = float(cp.random.uniform(*disorder_range_abs).get().item())
            local = 1.0  # Local potential to induce ABS-like states
            if mu_c is not None:
                params.append((t, mu, delta, disorder, local))
                labels.append(mu_c)
                class_labels.append(1)  # ABS-like class
        
        # Class 2: Large disorder (mu in [-60, 60], large disorder)
        mu_range_large_disorder = [-60, 60]
        disorder_range_large = [5, 10]  # Large disorder
        for _ in range(counts[2]):
            t = float(cp.random.uniform(*t_range).get().item())
            mu = float(cp.random.uniform(*mu_range_large_disorder).get().item())
            delta = float(cp.random.uniform(*delta_range).get().item())
            disorder = float(cp.random.uniform(*disorder_range_large).get().item())
            mu_c = topological_invariant(t, mu)
            if mu_c is not None:
                params.append((t, mu, delta, disorder, 0.0))
                labels.append(mu_c)
                class_labels.append(2)  # Large disorder class
        
        if len(params) != num_samples:
            raise ValueError(f"Expected {num_samples} samples, but generated {len(params)}")
        print(f"Generated {len(params)}/{num_samples} samples")
        return params, labels, class_labels
    except Exception as e:
        print(f"Error in generate_hamiltonians: {e}")
        return None, None, None

if __name__ == "__main__":
    params, labels, class_labels = generate_hamiltonians(num_samples=4000)