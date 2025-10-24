#!/usr/bin/env python3
"""
Script to split monolithic MPS plugin into modular structure.
Extracts operations from mps_pluggable_device_plugin.mm into categorized files.
"""

import re
import sys

# Operation categories and their target files
OP_CATEGORIES = {
    'elementwise': {
        'file': 'mps_elementwise_ops.mm',
        'ops': ['Add', 'Sub', 'Mul', 'Div', 'Neg', 'Abs', 'Sign', 'Sqrt', 'Rsqrt',
                'Square', 'Exp', 'Expm1', 'Log', 'Log1p', 'Sin', 'Cos', 'Tan',
                'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh',
                'Atanh', 'Ceil', 'Floor', 'Round', 'Rint', 'Pow', 'Maximum', 
                'Minimum', 'SquaredDifference', 'Reciprocal', 'RealDiv', 'FloorDiv',
                'FloorMod', 'TruncateDiv', 'TruncateMod', 'Mod']
    },
    'activation': {
        'file': 'mps_activation_ops.mm',
        'ops': ['Relu', 'Relu6', 'Elu', 'Selu', 'Leaky Relu', 'Gelu', 'Swish', 
                'Softplus', 'Softsign', 'Sigmoid', 'Tanh']
    },
    'comparison': {
        'file': 'mps_comparison_ops.mm',
        'ops': ['Equal', 'NotEqual', 'Less', 'LessEqual', 'Greater', 'GreaterEqual']
    },
    'logical': {
        'file': 'mps_logical_ops.mm',
        'ops': ['LogicalAnd', 'LogicalOr', 'LogicalNot']
    },
    'reduction': {
        'file': 'mps_reduction_ops.mm',
        'ops': ['Sum', 'Mean', 'Max', 'Min', 'Prod', 'All', 'Any', 'ArgMax', 'ArgMin']
    },
    'tensor': {
        'file': 'mps_tensor_ops.mm',
        'ops': ['Cast', 'Reshape', 'Transpose', 'Concat', 'Slice', 'StridedSlice',
                'Fill', 'ZerosLike', 'OnesLike', 'Pad', 'MirrorPad', 'Tile', 
                'Select', 'ClipByValue']
    },
    'indexing': {
        'file': 'mps_indexing_ops.mm',
        'ops': ['Split', 'SplitV', 'GatherV2', 'GatherNd', 'TensorScatterUpdate',
                'TensorScatterAdd', 'ScatterNd']
    },
    'nn': {
        'file': 'mps_nn_ops.mm',
        'ops': ['Conv2D', 'DepthwiseConv2dNative', 'MaxPool', 'AvgPool', 'MatMul',
                'FusedBatchNorm', 'FusedBatchNormV3', 'Softmax', 'LogSoftmax']
    },
    'utility': {
        'file': 'mps_utility_ops.mm',
        'ops': ['OneHot', 'Range', 'IsFinite', 'IsInf', 'IsNan', 'Shape', 'Size',
                'Rank', 'Identity']
    }
}

def extract_kernel_registration(content, op_name):
    """Extract kernel registration for a specific op."""
    # Pattern to match kernel registration
    pattern = rf'REGISTER_KERNEL_BUILDER\([^)]*Name\("{op_name}"\)[^)]*\)[^{{]*{{[^}}]*}}'
    matches = re.findall(pattern, content, re.DOTALL)
    return matches

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'tensorflow/mps/mps_pluggable_device_plugin.mm'
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Extract headers
    header_end = content.find('namespace {')
    if header_end == -1:
        header_end = content.find('namespace tensorflow')
    headers = content[:header_end]
    
    print("\nExtracting operations by category...")
    for category, info in OP_CATEGORIES.items():
        print(f"\n  {category}: {info['file']}")
        op_code = []
        
        for op in info['ops']:
            registrations = extract_kernel_registration(content, op)
            if registrations:
                print(f"    Found: {op}")
                op_code.extend(registrations)
        
        if op_code:
            output_path = f"tensorflow/mps/kernels/{info['file']}"
            with open(output_path, 'w') as f:
                f.write(headers)
                f.write('\n\nnamespace tensorflow {\nnamespace mps {\n\n')
                f.write('\n\n'.join(op_code))
                f.write('\n\n}  // namespace mps\n}  // namespace tensorflow\n')
            print(f"    Wrote {len(op_code)} registrations to {output_path}")

if __name__ == '__main__':
    main()
