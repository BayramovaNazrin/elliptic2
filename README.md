# Elliptic2: Inductive Graph Learning

## Project Overview
This project applies **Inductive Graph Learning** to the Elliptic2 Anti-Money Laundering (AML) dataset.
It utilizes a **GraphSAGE backbone** with **Neighbor Sampling** to process the large-scale transaction graph (49M nodes) on high-performance hardware.

## Methodology
1. **Graph Construction:**
   - **Nodes:** Transactions (from `background_nodes.csv`)
   - **Edges:** Money flows (from `background_edges.csv`)
   - **Labels:** Mapped via Connected Components (`clId` -> `ccId` -> `label`).

2. **Scalability Strategy:**
   - Uses `NeighborLoader` to handle the 49M node "Ocean" by sampling subgraphs (mini-batches) rather than loading the full adjacency matrix into VRAM.

3. **Future Work (Satellite++):**
   - The current pipeline establishes the baseline transaction graph. 
   - Next steps involve integrating the **Entity (Wallet)** features explicitly using a Heterogeneous GNN approach.

## How to Run
1. Place dataset files in `./dataset/`
2. Install requirements:
   `pip install -r requirements.txt`
3. Run training:
   `bash run.sh`
