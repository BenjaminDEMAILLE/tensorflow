#!/usr/bin/env python3
"""
Manual extraction helper - extracts ONE operation at a time with complete function bodies.
This is a MANUAL process - review each extraction before proceeding to next.
"""

import re
import sys
from pathlib import Path

def find_function_body(lines, start_idx):
    """Find complete function body from start to matching closing brace."""
    brace_count = 0
    in_function = False
    func_lines = []
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        func_lines.append(line)
        
        # Count braces
        for char in line:
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1
                if in_function and brace_count == 0:
                    return func_lines, i
    
    return func_lines, len(lines)

def extract_operation(monolith_file, operation_name, output_file):
    """
    Extract ONE operation (Create/Delete/Compute functions + registrations).
    
    Args:
        monolith_file: Path to mps_pluggable_device_plugin.mm
        operation_name: e.g., "Equal", "MatMul", "Conv2D"
        output_file: Target kernel file (e.g., kernels/mps_comparison_ops.mm)
    """
    print(f"\n{'='*80}")
    print(f"Extracting: {operation_name}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    with open(monolith_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Patterns to find
    patterns = {
        'create': fr'^\s*(?:extern\s+"C"\s+)?(?:void\*|TF_OpKernel\*)\s+MPS{operation_name}_Create\s*\(',
        'delete': fr'^\s*(?:extern\s+"C"\s+)?void\s+MPS{operation_name}_Delete\s*\(',
        'compute': fr'^\s*(?:extern\s+"C"\s+)?void\s+MPS{operation_name}_Compute\s*\(',
    }
    
    extracted_functions = {}
    
    # Find each function
    for func_type, pattern in patterns.items():
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                print(f"  Found {func_type}: line {i+1}")
                func_body, end_idx = find_function_body(lines, i)
                extracted_functions[func_type] = {
                    'start': i,
                    'end': end_idx,
                    'lines': func_body
                }
                break
    
    if not extracted_functions:
        print(f"  ⚠️  No functions found for {operation_name}")
        return False
    
    # Find registrations
    registration_lines = []
    in_registration = False
    for i, line in enumerate(lines):
        # Look for kernel builder with this operation
        if f'TF_NewKernelBuilder("{operation_name}"' in line or \
           f'RegisterKernelBuilder("MPS{operation_name}' in line:
            # Extract this registration block
            reg_block, end_idx = find_function_body(lines, i-1 if lines[i-1].strip().startswith('TF_KernelBuilder') else i)
            registration_lines.extend(reg_block)
            print(f"  Found registration: line {i+1}")
    
    # Show what was found
    print(f"\n  Summary:")
    for func_type, data in extracted_functions.items():
        print(f"    {func_type}: {len(data['lines'])} lines")
    print(f"    registrations: {len(registration_lines)} lines")
    
    # Generate output
    print(f"\n  ✅ Extraction complete!")
    print(f"  Next: Review and manually copy to {output_file}")
    
    # Save to temp file for review
    temp_file = Path(output_file).parent / f"_extract_{operation_name}.txt"
    with open(temp_file, 'w') as f:
        f.write(f"// Extracted {operation_name}\n\n")
        f.write("// IMPLEMENTATIONS\n")
        f.write("// " + "="*76 + "\n\n")
        for func_type in ['create', 'delete', 'compute']:
            if func_type in extracted_functions:
                f.write('\n'.join(extracted_functions[func_type]['lines']))
                f.write('\n\n')
        
        f.write("\n// REGISTRATIONS\n")
        f.write("// " + "="*76 + "\n\n")
        f.write('\n'.join(registration_lines))
    
    print(f"  📝 Saved to: {temp_file}")
    print(f"\n  MANUAL STEPS:")
    print(f"    1. Review {temp_file}")
    print(f"    2. Copy implementations to {output_file}")
    print(f"    3. Add registrations to Register*Ops() function")
    print(f"    4. Test compilation")
    print(f"    5. Delete {temp_file}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 manual_extract.py <operation_name> [category]")
        print("Example: python3 manual_extract.py Equal comparison")
        print("Example: python3 manual_extract.py MatMul nn")
        sys.exit(1)
    
    operation_name = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Determine category if not provided
    if not category:
        categories = {
            'Equal': 'comparison', 'NotEqual': 'comparison', 'Less': 'comparison',
            'LessEqual': 'comparison', 'Greater': 'comparison', 'GreaterEqual': 'comparison',
            'Relu': 'activation', 'Sigmoid': 'activation', 'Tanh': 'activation',
            'MatMul': 'nn', 'Conv2D': 'nn', 'MaxPool': 'nn',
            'Add': 'elementwise', 'Mul': 'elementwise', 'Sub': 'elementwise',
        }
        category = categories.get(operation_name, 'unknown')
    
    monolith = Path(__file__).parent.parent / 'mps_pluggable_device_plugin.mm'
    output = Path(__file__).parent.parent / 'kernels' / f'mps_{category}_ops.mm'
    
    if not monolith.exists():
        print(f"❌ Monolith file not found: {monolith}")
        sys.exit(1)
    
    extract_operation(monolith, operation_name, output)

if __name__ == '__main__':
    main()
