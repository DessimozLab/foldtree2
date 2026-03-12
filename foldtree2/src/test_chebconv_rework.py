#!/usr/bin/env python3
"""
Test script to verify that StableChebConv can be called with (-1, -1) pattern
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data

# Import the reworked class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foldtree2.src.chebconv import StableChebConv

def test_lazy_initialization():
    """Test that StableChebConv can be called with (-1, -1) and works correctly"""
    
    # Create test data
    x = torch.randn(10, 16)  # 10 nodes, 16 features
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    
    # Test 1: Create with (-1, -1) pattern like SAGEConv
    hidden_channels = {'edge_type1': [32, 64, 32]}
    edge_type = 'edge_type1'
    i = 0
    
    # This is the pattern you wanted to support:
    conv = StableChebConv((-1, -1), hidden_channels[edge_type][i])
    
    print(f"Created conv layer: {conv}")
    print(f"Before forward: in_channels = {conv.in_channels}")
    
    # Test forward pass - this should trigger lazy initialization
    out = conv(x, edge_index)
    
    print(f"After forward: in_channels = {conv.in_channels}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output channels: {hidden_channels[edge_type][i]}")
    
    assert out.shape == (10, hidden_channels[edge_type][i]), f"Expected shape (10, {hidden_channels[edge_type][i]}), got {out.shape}"
    
    # Test 2: Normal initialization should still work
    conv_normal = StableChebConv(16, 32)
    print(f"\nNormal conv layer: {conv_normal}")
    
    out_normal = conv_normal(x, edge_index)
    print(f"Normal output shape: {out_normal.shape}")
    
    assert out_normal.shape == (10, 32), f"Expected shape (10, 32), got {out_normal.shape}"
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_lazy_initialization()