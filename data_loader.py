import torch
import pandas as pd
import os
import numpy as np
from torch_geometric.data import Data

def load_elliptic2_data(root_dir):
    print(f"📂 Loading Elliptic2 data from: {root_dir}")
    
    # --- 1. DEFINE SPECIFIC FILES based on your snippets ---
    path_bg_nodes = os.path.join(root_dir, 'background_nodes.csv') # Features
    path_bg_edges = os.path.join(root_dir, 'background_edges.csv') # Edges
    path_nodes    = os.path.join(root_dir, 'nodes.csv')            # Mapping clId -> ccId
    path_cc       = os.path.join(root_dir, 'connected_components.csv') # Labels (ccId -> Label)

    # --- 2. LOAD BACKGROUND NODES (The Features) ---
    print("   Reading background nodes (Features)...")
    df_bg = pd.read_csv(path_bg_nodes)
    
    # Your snippet showed 'clId' in background_nodes earlier. 
    # If not, we assume col 0 is the ID.
    node_id_col = 'clId' if 'clId' in df_bg.columns else df_bg.columns[0]
    
    # Map Real ID (clId) -> Internal Index (0, 1, 2...)
    raw_ids = df_bg[node_id_col].astype(str).values
    num_nodes = len(raw_ids)
    id_to_idx = {raw_id: i for i, raw_id in enumerate(raw_ids)}
    
    # Extract Features (cols starting with 'feat')
    feat_cols = [c for c in df_bg.columns if str(c).startswith('feat')]
    if not feat_cols:
         # Fallback if names are different
         feat_cols = [c for c in df_bg.columns if c not in [node_id_col, 'time', 'class']]
    
    x = torch.tensor(df_bg[feat_cols].values, dtype=torch.float)
    print(f"   Loaded {num_nodes} nodes with {len(feat_cols)} features.")

    # --- 3. LOAD BACKGROUND EDGES ---
    print("   Reading background edges...")
    df_edges = pd.read_csv(path_bg_edges)
    
    # Map clId1 -> clId2
    src = df_edges['clId1'].astype(str).map(id_to_idx).dropna().values.astype(int)
    dst = df_edges['clId2'].astype(str).map(id_to_idx).dropna().values.astype(int)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # --- 4. THE CRITICAL PART: MERGING LABELS (Node -> CC -> Label) ---
    print("   🔗 Linking Nodes -> Components -> Labels...")
    
    # Load the mapping files
    df_n = pd.read_csv(path_nodes, dtype={'clId': str, 'ccId': str})
    df_cc = pd.read_csv(path_cc, dtype={'ccId': str})
    
    # Merge: nodes.csv + connected_components.csv
    # This attaches the 'ccLabel' to the 'clId'
    df_merged = df_n.merge(df_cc, on='ccId', how='left')
    
    # Create the label tensor (Default -1)
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    
    # Map labels to 0 (Licit) and 1 (Illicit)
    count_illicit = 0
    count_licit = 0
    
    for _, row in df_merged.iterrows():
        real_id = row['clId']
        raw_label = str(row['ccLabel']).lower()
        
        # Check if this node exists in our background graph
        if real_id in id_to_idx:
            idx = id_to_idx[real_id]
            
            # Logic: "1" or "illicit" -> 1. "0", "2", "licit" -> 0.
            if 'illicit' in raw_label or raw_label == '1':
                y[idx] = 1
                count_illicit += 1
            elif 'licit' in raw_label or raw_label == '2' or raw_label == '0':
                y[idx] = 0
                count_licit += 1
                
    print(f"   ✅ Labeled {count_illicit} Illicit and {count_licit} Licit nodes.")

    # --- 5. CREATE DATA OBJECT ---
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Create Masks
    indices = torch.where(y != -1)[0]
    perm = torch.randperm(len(indices))
    train_len = int(0.7 * len(indices))
    train_idx = indices[perm[:train_len]]
    test_idx = indices[perm[train_len:]]
    
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_idx] = True
    
    return data
