"""
Script to verify that all deep learning sections use PyTorch.
"""

import json
import re

def check_pytorch_usage():
    # Load notebook
    with open('reinforcement_learning_zero_to_hero.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print("Checking PyTorch usage in deep learning sections...")
    print("=" * 70)
    
    # Track findings
    pytorch_imports_found = False
    tensorflow_found = False
    keras_found = False
    neural_network_classes = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        
        # Check for imports
        if 'import torch' in source:
            pytorch_imports_found = True
            print(f"✓ Cell {i}: PyTorch imports found")
        
        if 'tensorflow' in source.lower() or 'import tf' in source:
            tensorflow_found = True
            print(f"⚠ Cell {i}: TensorFlow found!")
        
        if 'keras' in source.lower():
            keras_found = True
            print(f"⚠ Cell {i}: Keras found!")
        
        # Check for neural network class definitions
        if 'class' in source and 'nn.Module' in source:
            # Extract class name
            match = re.search(r'class\s+(\w+)\s*\(nn\.Module\)', source)
            if match:
                class_name = match.group(1)
                neural_network_classes.append((i, class_name))
                print(f"✓ Cell {i}: PyTorch neural network class '{class_name}' found")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"PyTorch imports found: {'✓ YES' if pytorch_imports_found else '✗ NO'}")
    print(f"TensorFlow found: {'✗ YES (should be removed)' if tensorflow_found else '✓ NO'}")
    print(f"Keras found: {'✗ YES (should be removed)' if keras_found else '✓ NO'}")
    print(f"\nPyTorch neural network classes found: {len(neural_network_classes)}")
    
    if neural_network_classes:
        print("\nNeural Network Classes:")
        for cell_idx, class_name in neural_network_classes:
            print(f"  - {class_name} (cell {cell_idx})")
    
    print("\n" + "=" * 70)
    
    if pytorch_imports_found and not tensorflow_found and not keras_found:
        print("✅ ALL DEEP LEARNING SECTIONS USE PYTORCH!")
        print("   The notebook is properly configured.")
        return True
    else:
        print("⚠️  ISSUES FOUND:")
        if not pytorch_imports_found:
            print("   - PyTorch imports not found")
        if tensorflow_found:
            print("   - TensorFlow found (should use PyTorch instead)")
        if keras_found:
            print("   - Keras found (should use PyTorch instead)")
        return False

if __name__ == "__main__":
    check_pytorch_usage()
