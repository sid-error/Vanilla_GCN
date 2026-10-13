import numpy as np
import torch

def normalize_adjacency(A):
    """
    Calculates the symmetrically normalized adjacency matrix.
    Equation: A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
    """
    num_nodes = A.shape[0]
    
    # 1. Add self-loops to the adjacency matrix: A_tilde = A + I
    # This ensures a node's own features are included when aggregating neighbors.
    A_tilde = A + np.eye(num_nodes, dtype=np.float32)
    
    # 2. Compute the degree matrix for A_tilde (D_tilde)
    degrees = A_tilde.sum(axis=1)
    
    # 3. Compute D_tilde^(-1/2)
    # Since we added self-loops, no degree is 0, so division by zero is avoided.
    d_inv_sqrt = np.power(degrees, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # 4. Perform the symmetric normalization
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    return A_hat.astype(np.float32)

def preprocess_data(A, X, labels, train_mask):
    """
    Normalizes the graph and converts all numpy arrays to PyTorch tensors.
    """
    A_hat = normalize_adjacency(A)
    
    # We do not need to row-normalize features (X) here because we are using 
    # an Identity matrix (one-hot encoding), so each row already sums to 1.
    
    return {
        "features": torch.tensor(X, dtype=torch.float32),
        "adj": torch.tensor(A_hat, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "train_mask": torch.tensor(train_mask, dtype=torch.bool)
    }

if __name__ == "__main__":
    from data_loader import load_florentine_families
    
    A, X, labels, train_mask, node_names = load_florentine_families()
    data = preprocess_data(A, X, labels, train_mask)
    
    print("Preprocessing complete.")
    print(f"Normalized Adjacency Matrix (A_hat) shape: {data['adj'].shape}")
    # In a symmetrically normalized matrix, rows don't sum strictly to 1, 
    # but the matrix trace and eigenvalues are stabilized.
    print(f"A_hat[0, 0] (self-loop weight for node 0): {data['adj'][0, 0]:.4f}")