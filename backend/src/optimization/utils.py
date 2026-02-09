import numpy as np
import pandas as pd

def make_psd(matrix: np.ndarray) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix.
    Replaces negative eigenvalues with zero.
    """
    # Check for NaNs or Infs
    if np.any(~np.isfinite(matrix)):
        # Replace with Identity or clean up?
        # A simple fallback is to zero out off-diagonals or replace with diagonal matrix?
        # Or better: raise an error?
        # Here we just zero them out for stability, but ideally upstream should fix.
        # But let's be safe: replace NaN with 0.
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure symmetry
    matrix = (matrix + matrix.T) / 2
    
    try:
        eigvals, eigvecs = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        # If eigenvalues fail to converge, return diagonal matrix (Identity basically)
        return np.eye(matrix.shape[0]) * np.max(np.diag(matrix)) # scaled identity

    
    # Clip negative eigenvalues to 0 (or small epsilon)
    eigvals = np.maximum(eigvals, 1e-8)
    
    # Reconstruct
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def clean_weights(weights: pd.Series, cutoff: float = 1e-4) -> pd.Series:
    """
    Zero out small weights and renormalize.
    """
    w = weights.copy()
    w[w < cutoff] = 0
    return w / w.sum()
