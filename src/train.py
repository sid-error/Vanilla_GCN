import torch
import torch.nn.functional as F
from data_loader import load_florentine_families
from preprocessing import preprocess_data
from gcn_model import VanillaGCN

def compute_accuracy(logits, labels, mask):
    """Calculates accuracy for the nodes specified by the mask."""
    predictions = logits.argmax(dim=1)
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def train_gcn():
    # 1. Load and Preprocess Data
    A, X, labels, train_mask, node_names = load_florentine_families()
    data = preprocess_data(A, X, labels, train_mask)
    
    # We will use all non-training nodes as our test set for this small graph
    test_mask = ~data["train_mask"]
    
    # 2. Initialize Model and Optimizer
    in_features = data["features"].shape[1]
    hidden_features = 8
    num_classes = len(torch.unique(data["labels"]))
    
    model = VanillaGCN(in_features, hidden_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print(f"Starting training on {data['train_mask'].sum().item()} nodes...\n")
    
    # 3. Training Loop
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on the FULL graph
        logits = model(data["features"], data["adj"])
        
        # Compute loss ONLY on training nodes (Transductive learning)
        loss = F.cross_entropy(
            logits[data["train_mask"]], 
            data["labels"][data["train_mask"]]
        )
        
        # Backward pass and weight update
        loss.backward()
        optimizer.step()
        
        # 4. Evaluation
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(data["features"], data["adj"])
                train_acc = compute_accuracy(val_logits, data["labels"], data["train_mask"])
                test_acc = compute_accuracy(val_logits, data["labels"], test_mask)
                
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    train_gcn()