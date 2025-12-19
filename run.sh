#!/bin/bash

# 1. Clear GPU Cache (Good practice on shared servers)
echo "🧹 Cleaning up memory..."
rm -rf __pycache__

# 2. Run the Training
echo "🚀 Starting Elliptic2 Training..."
python train.py

echo "✅ Done."
