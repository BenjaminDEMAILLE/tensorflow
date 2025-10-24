#!/bin/bash
# Migration script: Monolithic → Modular MPS Backend
# Extracts operations from mps_pluggable_device_plugin.mm into categorized files

set -e

echo "🔄 MPS Backend Modularization Migration"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPS_DIR="$(dirname "$SCRIPT_DIR")"
MONOLITHIC_FILE="$MPS_DIR/mps_pluggable_device_plugin.mm"

# Check if monolithic file exists
if [ ! -f "$MONOLITHIC_FILE" ]; then
    echo "❌ Error: $MONOLITHIC_FILE not found"
    exit 1
fi

echo "📁 Source: $MONOLITHIC_FILE"
echo "📊 Status: $(wc -l < "$MONOLITHIC_FILE") lines"
echo ""

# Phase 1: Infrastructure (already complete)
echo "✅ Phase 1: Infrastructure"
echo "   - device/mps_device.{h,mm}"
echo "   - device/mps_device_factory.mm"
echo "   - ops/mps_ops_registry.{h,mm}"
echo "   - utils/mps_utils.h"
echo ""

# Phase 2: Kernel extraction (TODO)
echo "🔄 Phase 2: Kernel Extraction"
echo "   ⏳ kernels/mps_elementwise_ops.mm (41 ops)"
echo "   ⏳ kernels/mps_activation_ops.mm (9 ops)"
echo "   ⏳ kernels/mps_comparison_ops.mm (6 ops)"
echo "   ⏳ kernels/mps_logical_ops.mm (3 ops)"
echo "   ⏳ kernels/mps_reduction_ops.mm (7 ops)"
echo "   ⏳ kernels/mps_tensor_ops.mm (15 ops)"
echo "   ⏳ kernels/mps_indexing_ops.mm (5 ops)"
echo "   ⏳ kernels/mps_nn_ops.mm (7 ops)"
echo "   ⏳ kernels/mps_utility_ops.mm (3 ops)"
echo ""

# Phase 3: Build system update
echo "⏳ Phase 3: Build System Update"
echo "   - Update BUILD to use modular targets"
echo "   - Create per-category test targets"
echo ""

# Phase 4: Testing
echo "⏳ Phase 4: Testing"
echo "   - Run smoke tests"
echo "   - Run category tests"
echo "   - Performance benchmarks"
echo ""

echo "📝 Next Steps:"
echo "   1. Extract kernel implementations from monolithic file"
echo "   2. Update BUILD file to use modular structure"
echo "   3. Run tests: bazel test //tensorflow/mps/..."
echo "   4. Benchmark: python3 tensorflow/mps/benchmark_ops.py"
echo ""
echo "📚 Documentation: tensorflow/mps/README_MODULAR.md"
echo ""
echo "Done! 🎉"
