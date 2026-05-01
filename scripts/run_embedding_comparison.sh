#!/bin/bash

# FoldTree2 Embedding Size Comparison Script
# Runs training for 10, 20, 30, and 40 embeddings sequentially
# Each run trains with all three decoders (Sequence + Geometry + FoldX)

set -e  # Exit on error

echo "========================================="
echo "FoldTree2 Embedding Size Comparison"
echo "========================================="
echo ""
echo "This script will train 4 models sequentially:"
echo "  1. 10 embeddings"
echo "  2. 20 embeddings"
echo "  3. 30 embeddings"
echo "  4. 40 embeddings"
echo ""
echo "Output will be saved to: ./models/embedding_comparison/"
echo "TensorBoard logs: ./runs/"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to start..."
sleep 5

# Create output directory
mkdir -p models/embedding_comparison

# Training function with error handling
run_training() {
    local config=$1
    local name=$2
    
    echo ""
    echo "========================================="
    echo "Starting: $name"
    echo "Config: $config"
    echo "Started at: $(date)"
    echo "========================================="
    
    if python foldtree2/learn_monodecoder.py --config "$config"; then
        echo ""
        echo "✓ Completed: $name"
        echo "Finished at: $(date)"
    else
        echo ""
        echo "✗ Failed: $name"
        echo "Error occurred at: $(date)"
        echo "Check logs above for details"
        exit 1
    fi
}

# Run all training configurations
run_training "config_10_embeddings.yaml" "10 Embeddings"
run_training "config_20_embeddings.yaml" "20 Embeddings"
run_training "config_30_embeddings.yaml" "30 Embeddings"
run_training "config_40_embeddings.yaml" "40 Embeddings"

echo ""
echo "========================================="
echo "All training runs completed successfully!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - Models: ./models/embedding_comparison/"
echo "  - Logs: ./runs/"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir=./runs/"
echo ""
echo "Finished at: $(date)"
