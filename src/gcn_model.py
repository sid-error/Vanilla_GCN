import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):
    """
    A single Vanilla GCN layer.
    Computes: H_out = A_hat @ H_in @ W
    """
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        # Trainable weight matrix W
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Standard initialization for stable gradients
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, adj):
        # 1. Feature transformation: H_in @ W
        support = torch.mm(node_features, self.weight)
        
        # 2. Neighborhood aggregation: A_hat @ (H_in @ W)
        # Using torch.mm since our adjacency matrix is a dense tensor for this small graph
        output = torch.mm(adj, support) 
        
        return output + self.bias

class VanillaGCN(nn.Module):
    """
    2-Layer Vanilla GCN for node classification.
    """
    def __init__(self, in_features, hidden_features, num_classes):
        super(VanillaGCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, num_classes)

    def forward(self, node_features, adj):
        # Layer 1: Convolution -> ReLU Activation
        x = self.gc1(node_features, adj)
        x = F.relu(x)
        
        # Layer 2: Convolution -> Raw Logits
        # We don't apply softmax here because PyTorch's CrossEntropyLoss handles it internally
        x = self.gc2(x, adj)
        return x

if __name__ == "__main__":
    # Quick test to ensure the dimensions align
    in_feat = 15
    hidden = 8
    out_classes = 2
    
    dummy_features = torch.rand(15, in_feat)
    dummy_adj = torch.rand(15, 15)
    
    model = VanillaGCN(in_features=in_feat, hidden_features=hidden, num_classes=out_classes)
    logits = model(dummy_features, dummy_adj)
    
    print("Model initialized successfully.")
    print(f"Output shape (Nodes x Classes): {logits.shape}")