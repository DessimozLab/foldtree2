#!/usr/bin/env python3
"""
Test loading benchmark config files to verify parameter mapping.
"""

import yaml
import os
import sys

def test_config_file(config_path):
    """Test loading a config file and check key parameters."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_path}")
    print('='*80)
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expected parameters
    expected_params = [
        'model_name',
        'epochs',
        'batch_size',
        'hidden_size',
        'num_embeddings',
        'embedding_dim',
        'use_muon',
        'muon_lr',
        'adamw_lr',
    ]
    
    # Loss weights (use compact names as in config)
    loss_weights = [
        'edgeweight',
        'logitweight',
        'xweight',
        'vqweight',
        'angles_weight',
        'ss_weight',
    ]
    
    print("\n📋 Key Parameters:")
    print("-" * 80)
    for param in expected_params:
        if param in config:
            print(f"  ✓ {param:25s} = {config[param]}")
        else:
            print(f"  ✗ {param:25s} = MISSING")
    
    print("\n⚖️  Loss Weights:")
    print("-" * 80)
    for weight in loss_weights:
        if weight in config:
            print(f"  ✓ {weight:25s} = {config[weight]}")
        else:
            print(f"  ✗ {weight:25s} = MISSING (will use default)")
    
    # Check for parameter name mapping issues
    print("\n🔄 Parameter Mapping:")
    print("-" * 80)
    
    # These parameters might use different names in config vs CLI
    mappings = {
        'edgeweight': 'edge_weight',
        'logitweight': 'logit_weight',
        'xweight': 'x_weight',
        'fft2weight': 'fft2_weight',
        'vqweight': 'vq_weight',
    }
    
    for config_key, cli_key in mappings.items():
        if config_key in config:
            print(f"  ✓ '{config_key}' → '--{cli_key.replace('_', '-')}' = {config[config_key]}")
        else:
            print(f"  ⊘ '{config_key}' not in config (will use CLI default)")
    
    # Verify critical settings for benchmark
    print("\n🎯 Benchmark-Specific Settings:")
    print("-" * 80)
    
    critical_settings = {
        'num_embeddings': None,  # Should vary by config
        'use_muon': True,
        'mixed_precision': True,
        'clip_grad': True,
        'mask_plddt': True,
    }
    
    issues = []
    for setting, expected in critical_settings.items():
        if setting in config:
            actual = config[setting]
            if expected is not None and actual != expected:
                print(f"  ⚠️  {setting:25s} = {actual} (expected {expected})")
                issues.append(f"{setting}: {actual} vs expected {expected}")
            else:
                print(f"  ✓ {setting:25s} = {actual}")
        else:
            print(f"  ⊘ {setting:25s} not specified (will use default)")
    
    success = len(issues) == 0
    
    if success:
        print(f"\n✅ Config file is valid and ready to use!")
    else:
        print(f"\n⚠️  Config file has {len(issues)} potential issues")
        for issue in issues:
            print(f"    - {issue}")
    
    return success

def main():
    print("="*80)
    print("BENCHMARK CONFIG FILE VERIFICATION")
    print("="*80)
    
    # Test all benchmark configs
    config_dir = 'benchmark_configs'
    config_files = [
        'config_10_embeddings.yaml',
        'config_15_embeddings.yaml',
        'config_20_embeddings.yaml',
        'config_25_embeddings.yaml',
        'config_30_embeddings.yaml',
        'config_35_embeddings.yaml',
        'config_40_embeddings.yaml',
    ]
    
    results = {}
    for config_file in config_files:
        config_path = os.path.join(config_dir, config_file)
        try:
            success = test_config_file(config_path)
            results[config_file] = success
        except Exception as e:
            print(f"\n❌ Error testing {config_file}: {e}")
            results[config_file] = False
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for config_file, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {config_file}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {passed}/{total} configs passed ({100*passed/total:.0f}%)")
    print("="*80)
    
    if passed == total:
        print("\n✅ All benchmark configs are ready to use with learn_monodecoder.py!")
        return 0
    else:
        print("\n⚠️  Some configs may need adjustments")
        return 1

if __name__ == '__main__':
    sys.exit(main())
