import networkx as nx
import numpy as np

def load_florentine_families():
    """
    Loads the Florentine Families graph.
    Returns the Adjacency matrix, Feature matrix, Labels, and Train mask.
    """
    # Load the established dataset built into NetworkX
    G = nx.florentine_families_graph()
    num_nodes = G.number_of_nodes()
    
    # 1. Adjacency Matrix (A)
    # The graph has 15 nodes representing Renaissance Florentine families.
    A = nx.to_numpy_array(G, dtype=np.float32)
    
    # 2. Feature Matrix (X)
    # Since nodes don't have explicit attributes (like word vectors), 
    # we use an Identity matrix as features (one-hot encoding each family).
    X = np.eye(num_nodes, dtype=np.float32)
    
    # 3. Labels (y)
    # We will create a synthetic classification task: predicting if a family 
    # has high marriage ties (degree > 2) or low marriage ties (degree <= 2).
    # Class 1: High connectivity, Class 0: Low connectivity
    labels = np.array([1 if G.degree(n) > 2 else 0 for n in G.nodes()], dtype=np.int64)
    
    # 4. Train Mask
    # We will train on 4 out of the 15 nodes (semi-supervised learning).
    train_mask = np.zeros(num_nodes, dtype=bool)
    # We pick two known high-degree families (Medici, Strozzi) and two low-degree ones
    # (assuming standard NetworkX node ordering, we will just pick by index)
    train_indices = [0, 1, 8, 9] 
    for idx in train_indices:
        train_mask[idx] = True
        
    # Get node names for your mathematical report later
    node_names = list(G.nodes())
    
    return A, X, labels, train_mask, node_names

if __name__ == "__main__":
    A, X, labels, train_mask, node_names = load_florentine_families()
    print(f"Loaded Florentine Families dataset:")
    print(f"Nodes: {A.shape[0]}")
    print(f"Edges: {int(A.sum() / 2)}")
    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Training Nodes: {train_mask.sum()}")
    print(f"\nNode Names:\n{node_names}")