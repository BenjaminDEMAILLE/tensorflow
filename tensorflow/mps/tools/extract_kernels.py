#!/usr/bin/env python3
"""
Automated kernel extraction from monolithic MPS plugin.
Extracts all 110+ operations into categorized files.
"""

import re
import sys
from pathlib import Path

# Operation categorization mapping
OP_CATEGORIES = {
    'elementwise': [
        'Add', 'AddV2', 'Sub', 'Mul', 'Div', 'Neg', 'Abs', 'Sign', 'Sqrt', 'Rsqrt',
        'Square', 'Exp', 'Expm1', 'Log', 'Log1p', 'Sin', 'Cos', 'Tan',
        'Asin', 'Acos', 'Atan', 'Sinh', 'Cosh', 'Tanh', 'Asinh', 'Acosh',
        'Atanh', 'Ceil', 'Floor', 'Round', 'Rint', 'Pow', 'Maximum', 
        'Minimum', 'SquaredDifference', 'Reciprocal', 'RealDiv', 'FloorDiv',
        'FloorMod', 'TruncateDiv', 'TruncateMod', 'Mod', 'Erf'
    ],
    'activation': [
        'Relu', 'Relu6', 'Elu', 'Selu', 'LeakyRelu', 'Gelu', 'Swish', 
        'Softplus', 'Softsign', 'Sigmoid'
    ],
    'comparison': [
        'Equal', 'NotEqual', 'Less', 'LessEqual', 'Greater', 'GreaterEqual'
    ],
    'logical': [
        'LogicalAnd', 'LogicalOr', 'LogicalNot'
    ],
    'reduction': [
        'Sum', 'Mean', 'Max', 'Min', 'Prod', 'All', 'Any', 'ArgMax', 'ArgMin'
    ],
    'tensor': [
        'Cast', 'Reshape', 'Transpose', 'Concat', 'ConcatV2', 'Slice', 'StridedSlice',
        'Fill', 'ZerosLike', 'OnesLike', 'Pad', 'MirrorPad', 'Tile', 
        'Select', 'SelectV2', 'ClipByValue', 'Identity', 'Shape', 'Size', 'Rank'
    ],
    'indexing': [
        'Split', 'SplitV', 'GatherV2', 'GatherNd', 'TensorScatterUpdate',
        'TensorScatterAdd', 'ScatterNd'
    ],
    'nn': [
        'Conv2D', 'DepthwiseConv2dNative', 'MaxPool', 'AvgPool', 'MatMul',
        'FusedBatchNorm', 'FusedBatchNormV3', 'Softmax', 'LogSoftmax',
        'BatchMatMul', 'BatchMatMulV2'
    ],
    'utility': [
        'OneHot', 'Range', 'IsFinite', 'IsInf', 'IsNan'
    ]
}

def read_file(filepath):
    """Read entire file content."""
    with open(filepath, 'r') as f:
        return f.read()

def extract_headers(content):
    """Extract header includes from monolithic file."""
    lines = content.split('\n')
    headers = []
    in_headers = True
    
    for line in lines:
        if in_headers:
            if line.startswith('#include') or line.startswith('#import') or \
               line.startswith('/*') or line.startswith(' *') or line.startswith(' ==') or \
               line.strip() == '':
                headers.append(line)
            elif 'namespace' in line or 'constexpr' in line:
                in_headers = False
        else:
            break
    
    return '\n'.join(headers)

def extract_function_impl(content, op_name, variant=''):
    """Extract implementation for a specific operation."""
    # Try different function name patterns
    patterns = [
        f'void\\* MPS{op_name}{variant}_Create',
        f'void MPS{op_name}{variant}_Compute',
        f'void MPS{op_name}{variant}_Delete',
    ]
    
    implementations = []
    for pattern in patterns:
        # Find function definition
        regex = re.compile(
            f'{pattern}\\([^{{]*\\)\\s*{{[^}}]*}}',
            re.DOTALL | re.MULTILINE
        )
        matches = regex.findall(content)
        if matches:
            implementations.extend(matches)
    
    return implementations

def extract_registrations(content, op_name):
    """Extract kernel registration code for an operation."""
    # Look for TF_RegisterKernelBuilder calls
    pattern = f'TF_RegisterKernelBuilder\\("MPS{op_name}[^"]*"'
    matches = []
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        if pattern in lines[i] or f'MPS{op_name}' in lines[i]:
            # Capture context around registration
            start = max(0, i - 5)
            end = min(len(lines), i + 3)
            matches.append('\n'.join(lines[start:end]))
        i += 1
    
    return matches

def create_category_file(category, ops, monolithic_content, output_dir):
    """Create a category file with extracted implementations."""
    filename = f"mps_{category}_ops.mm"
    filepath = output_dir / 'kernels' / filename
    
    print(f"\n📝 Creating {filename}...")
    
    # Start with headers
    headers = """/* Copyright 2025 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/mps/ops/mps_ops_registry.h"
#include "tensorflow/mps/utils/mps_utils.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

using namespace tensorflow::mps;

namespace tensorflow {
namespace mps {

"""
    
    # Extract implementations for each op
    implementations = []
    registrations = []
    
    for op in ops:
        print(f"  Extracting {op}...")
        # Try to find implementations
        impls = extract_function_impl(monolithic_content, op)
        if impls:
            implementations.extend(impls)
            print(f"    ✓ Found {len(impls)} functions")
        
        # Try variants (Half, BFloat16)
        for variant in ['Half', 'BFloat16', '']:
            impls = extract_function_impl(monolithic_content, op, variant)
            implementations.extend(impls)
    
    # Footer with registration function
    footer = f"""

// ============================================================================
// Registration
// ============================================================================

void Register{category.capitalize()}Ops(const char* platform_name, TF_Status* status) {{
    // TODO: Add registration calls
    // Example: REGISTER_MPS_UNARY_OP_3DTYPE(OpName, platform_name, status);
}}

}}  // namespace mps
}}  // namespace tensorflow
"""
    
    # Write file
    content = headers
    if implementations:
        content += "\n// ============================================================================\n"
        content += f"// {category.capitalize()} Operations\n"
        content += "// ============================================================================\n\n"
        content += '\n\n'.join(set(implementations))  # Remove duplicates
    else:
        content += f"// TODO: Extract implementations from mps_pluggable_device_plugin.mm\n"
        content += f"// Operations: {', '.join(ops[:10])}\n"
        if len(ops) > 10:
            content += f"// ... and {len(ops) - 10} more\n"
    
    content += footer
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Wrote {filepath}")
    return len(implementations)

def main():
    # Paths
    tf_dir = Path(__file__).parent.parent.parent.parent
    mps_dir = tf_dir / 'tensorflow' / 'mps'
    monolithic_file = mps_dir / 'mps_pluggable_device_plugin.mm'
    
    if not monolithic_file.exists():
        print(f"❌ Error: {monolithic_file} not found")
        return 1
    
    print("🔄 MPS Kernel Extraction Tool")
    print("=" * 60)
    print(f"Source: {monolithic_file}")
    print(f"Output: {mps_dir / 'kernels'}/")
    print()
    
    # Read monolithic file
    print("📖 Reading monolithic file...")
    content = read_file(monolithic_file)
    print(f"  Lines: {len(content.splitlines())}")
    
    # Extract for each category
    total_extracted = 0
    for category, ops in OP_CATEGORIES.items():
        extracted = create_category_file(category, ops, content, mps_dir)
        total_extracted += extracted
    
    print("\n" + "=" * 60)
    print(f"✅ Extraction complete!")
    print(f"   Categories: {len(OP_CATEGORIES)}")
    print(f"   Operations: {sum(len(ops) for ops in OP_CATEGORIES.values())}")
    print(f"   Functions extracted: {total_extracted}")
    print()
    print("📝 Next steps:")
    print("   1. Review extracted files in tensorflow/mps/kernels/")
    print("   2. Add registration calls in each Register*Ops() function")
    print("   3. Test compilation: bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
