import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import os

# --- IMPORTS FROM YOUR FILES ---
from data_loader import load_elliptic2_data
# If you didn't make model.py, I included the class below just in case.
# If you have model.py, you can uncomment: from model import EllipticGraphSAGE
from torch_geometric.nn import SAGEConv

# --- CONFIGURATION (Adjust for the Powerful Computer) ---
DATA_PATH = "./dataset"  # The folder where you unzip the CSVs
BATCH_SIZE = 4096        # High batch size for powerful GPU (A100/H100)
HIDDEN_CHANNELS = 256
LR = 0.005
EPOCHS = 50
NUM_NEIGHBORS = [15, 10, 5] # Sample 3 layers deep

# --- MODEL DEFINITION (Included here for safety) ---
class EllipticGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 3 (Output)
        x = self.conv3(x, edge_index)
        return x

# --- MAIN EXECUTION ---
def main():
    print("🚀 Starting Elliptic2 Training Pipeline...")
    
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    # 2. Load Data
    # This calls the script you just wrote
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERROR: Data path '{DATA_PATH}' does not exist.")
        print("   Please unzip the dataset files into a folder named 'dataset'.")
        return

    data = load_elliptic2_data(DATA_PATH)
    
    # 3. Create Loaders (The Engine)
    # This splits the massive graph into small chunks the GPU can handle
    print("   Initializing NeighborLoaders...")
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=4 # Use CPU cores to load data faster
    )
    
    test_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.test_mask,
        num_workers=4
    )

    # 4. Initialize Model
    # Elliptic2 usually has ~167 features
    model = EllipticGraphSAGE(
        in_channels=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=2 # 0=Licit, 1=Illicit
    ).to(device)
    
    # --- CRITICAL: CLASS WEIGHTS ---
    # Illicit transactions are rare (maybe 10%). 
    # We tell the model: "Pay 7x more attention to Illicit nodes."
    # If you don't do this, the model will just guess "Licit" for everything.
    class_weights = torch.tensor([1.0, 7.0]).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 5. Training Loop
    print("\n   🔥 Training Started...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            out = model(batch.x, batch.edge_index)
            
            # Loss Calculation (only on the specific batch nodes)
            # We slice 'out' and 'y' to match the batch size
            batch_size = batch.batch_size
            loss = criterion(out[:batch_size], batch.y[:batch_size])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Epoch {epoch+1:02d} | Loss: {total_loss / len(train_loader):.4f}")

        # --- EVALUATION EVERY 5 EPOCHS ---
        if (epoch + 1) % 5 == 0:
            test_f1 = evaluate(model, test_loader, device)
            print(f"   📊 Test F1-Score (Illicit class): {test_f1:.4f}")

    # 6. Save Model
    torch.save(model.state_dict(), "elliptic2_sage_model.pth")
    print("\n✅ Training Complete. Model saved as 'elliptic2_sage_model.pth'")

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out[:batch.batch_size].argmax(dim=1)
            
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y[:batch.batch_size].cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate F1 specifically for the '1' (Illicit) class
    # This is the number your teacher cares about.
    return f1_score(all_labels, all_preds, average='binary', pos_label=1)

if __name__ == "__main__":
    main()
