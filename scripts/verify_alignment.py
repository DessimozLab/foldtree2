#!/usr/bin/env python3
"""
Verification script to check alignment between notebook and learn_monodecoder.py defaults.
"""

import yaml
import argparse

def get_script_defaults():
    """Extract default values from learn_monodecoder.py argument parser."""
    # Recreate the argument parser to extract defaults
    parser = argparse.ArgumentParser(description='Check defaults')
    
    # Add all arguments with their defaults
    parser.add_argument('--hidden-size', type=int, default=150)
    parser.add_argument('--num-embeddings', type=int, default=30)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2)
    parser.add_argument('--edge-weight', type=float, default=0.1)
    parser.add_argument('--logit-weight', type=float, default=0.1)
    parser.add_argument('--x-weight', type=float, default=0.1)
    parser.add_argument('--fft2-weight', type=float, default=0.01)
    parser.add_argument('--vq-weight', type=float, default=0.005)
    parser.add_argument('--angles-weight', type=float, default=0.1)
    parser.add_argument('--ss-weight', type=float, default=0.1)
    parser.add_argument('--muon-lr', type=float, default=0.02)
    parser.add_argument('--adamw-lr', type=float, default=1e-4)
    parser.add_argument('--commitment-cost', type=float, default=0.9)
    parser.add_argument('--commitment-warmup-steps', type=int, default=1000)
    parser.add_argument('--commitment-start', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='structs_train_final.h5')
    
    args = parser.parse_args([])  # Parse with no arguments to get defaults
    return vars(args)

def get_notebook_defaults():
    """Notebook default values."""
    return {
        'hidden_size': 150,
        'num_embeddings': 30,
        'embedding_dim': 128,
        'batch_size': 10,
        'gradient_accumulation_steps': 2,
        'edge_weight': 0.1,
        'logit_weight': 0.1,
        'x_weight': 0.1,
        'fft2_weight': 0.01,
        'vq_weight': 0.005,
        'angles_weight': 0.1,
        'ss_weight': 0.1,
        'muon_lr': 0.02,
        'adamw_lr': 1e-4,
        'commitment_cost': 0.9,
        'commitment_warmup_steps': 1000,
        'commitment_start': 0.5,
        'dataset': 'structs_train_final.h5'
    }

def main():
    print("="*80)
    print("VERIFYING ALIGNMENT BETWEEN NOTEBOOK AND SCRIPT")
    print("="*80)
    print()
    
    script_defaults = get_script_defaults()
    notebook_defaults = get_notebook_defaults()
    
    # Check alignment
    mismatches = []
    matches = []
    
    for key in notebook_defaults:
        script_key = key.replace('_', '-')  # Handle hyphen/underscore
        
        notebook_val = notebook_defaults[key]
        script_val = script_defaults.get(key, script_defaults.get(script_key))
        
        if notebook_val == script_val:
            matches.append((key, notebook_val, script_val))
        else:
            mismatches.append((key, notebook_val, script_val))
    
    # Print results
    if matches:
        print("✅ MATCHING DEFAULTS:")
        print("-" * 80)
        for key, nb_val, sc_val in matches:
            print(f"  {key:30s} = {nb_val}")
        print()
    
    if mismatches:
        print("❌ MISMATCHED DEFAULTS:")
        print("-" * 80)
        for key, nb_val, sc_val in mismatches:
            print(f"  {key:30s}")
            print(f"    Notebook: {nb_val}")
            print(f"    Script:   {sc_val}")
        print()
    
    # Summary
    total = len(notebook_defaults)
    matched = len(matches)
    print("="*80)
    print(f"SUMMARY: {matched}/{total} parameters match ({100*matched/total:.1f}%)")
    print("="*80)
    
    if mismatches:
        print("\n⚠️  WARNING: Some parameters don't match!")
        return 1
    else:
        print("\n✅ SUCCESS: All parameters match!")
        return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
