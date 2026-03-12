#!/usr/bin/env python
"""
Test script for FoldTree2 conda package
"""
import sys
import subprocess

def test_imports():
    """Test that all core modules can be imported"""
    print("Testing module imports...")
    try:
        import foldtree2
        print("  ✓ foldtree2")
        
        import foldtree2.src.encoder
        print("  ✓ foldtree2.src.encoder")
        
        import foldtree2.src.mono_decoders
        print("  ✓ foldtree2.src.mono_decoders")
        
        import foldtree2.src.pdbgraph
        print("  ✓ foldtree2.src.pdbgraph")
        
        print("✓ All core modules imported successfully\n")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}\n")
        return False

def test_entry_points():
    """Test that entry points are available"""
    print("Testing command-line entry points...")
    
    commands = [
        ['foldtree2', '--about'],
        ['pdbs-to-graphs', '--help'],
        ['makesubmat', '--help']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                timeout=10,
                text=True
            )
            if result.returncode == 0:
                print(f"  ✓ Command '{' '.join(cmd)}' works")
            else:
                print(f"  ✗ Command '{' '.join(cmd)}' failed with return code {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"  ✗ Command '{' '.join(cmd)}' timed out")
            return False
        except FileNotFoundError:
            print(f"  ✗ Command '{' '.join(cmd)}' not found")
            return False
        except Exception as e:
            print(f"  ✗ Command '{' '.join(cmd)}' error: {e}")
            return False
    
    print("✓ All entry points work correctly\n")
    return True

def test_dependencies():
    """Test that key dependencies are available"""
    print("Testing key dependencies...")
    
    deps = [
        'torch',
        'torch_geometric',
        'Bio',
        'pytorch_lightning',
        'numpy',
        'pandas'
    ]
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} not found")
            return False
    
    print("✓ All key dependencies available\n")
    return True

if __name__ == '__main__':
    print("="*60)
    print("FoldTree2 Conda Package Test Suite")
    print("="*60 + "\n")
    
    success = (
        test_imports() and 
        test_dependencies() and
        test_entry_points()
    )
    
    if success:
        print("="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        sys.exit(0)
    else:
        print("="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)
