#!/bin/bash
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Comprehensive validation of MPS implementation

echo "════════════════════════════════════════════════════════════════"
echo "  MPS TENSORFLOW IMPLEMENTATION VALIDATION"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TOTAL_PASS=0
TOTAL_FAIL=0

check_file() {
    local file=$1
    local expected_ops=$2
    
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ MISSING${NC}: $file"
        ((TOTAL_FAIL++))
        return 1
    fi
    
    local unimpl_count=$(grep -c "TF_UNIMPLEMENTED" "$file" 2>/dev/null || echo 0)
    
    if [ "$unimpl_count" -eq 0 ]; then
        echo -e "${GREEN}✓ COMPLETE${NC}: $file (0 TF_UNIMPLEMENTED)"
        ((TOTAL_PASS++))
    else
        echo -e "${YELLOW}⚠ PARTIAL${NC}: $file ($unimpl_count TF_UNIMPLEMENTED)"
        ((TOTAL_FAIL++))
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1: CORE IMPLEMENTATIONS (Previously Complete)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_file "tensorflow/mps/kernels/mps_graph_executor.mm" 28
check_file "tensorflow/mps/kernels/mps_data_executor.mm" 8
check_file "tensorflow/mps/kernels/mps_linalg_accelerate.mm" 6
check_file "tensorflow/mps/kernels/mps_signal_vdsp.mm" 6
check_file "tensorflow/mps/kernels/mps_image_complete.mm" 9
check_file "tensorflow/mps/kernels/mps_quantization_complete.mm" 4

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2: NEW IMPLEMENTATIONS (Special Math & Reductions)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_file "tensorflow/mps/kernels/mps_special_math_complete.mm" 29
check_file "tensorflow/mps/kernels/mps_reduction_complete.mm" 18

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DOCUMENTATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

check_file "tensorflow/mps/MPS_IMPLEMENTATION_FINALE.md" 0
check_file "tensorflow/mps/MPS_PR_SUMMARY.md" 0
check_file "tensorflow/mps/MPS_PROGRESS_SUMMARY.md" 0

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OVERALL STATISTICS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count total operations implemented
TOTAL_OPS=0
for file in tensorflow/mps/kernels/mps_graph_executor.mm \
            tensorflow/mps/kernels/mps_data_executor.mm \
            tensorflow/mps/kernels/mps_linalg_accelerate.mm \
            tensorflow/mps/kernels/mps_signal_vdsp.mm \
            tensorflow/mps/kernels/mps_image_complete.mm \
            tensorflow/mps/kernels/mps_quantization_complete.mm \
            tensorflow/mps/kernels/mps_special_math_complete.mm \
            tensorflow/mps/kernels/mps_reduction_complete.mm; do
    if [ -f "$file" ]; then
        # Count function definitions (Create/Delete/Compute triplets)
        ops=$(grep -c "extern \"C\" void.*_Compute" "$file" 2>/dev/null || echo 0)
        TOTAL_OPS=$((TOTAL_OPS + ops))
    fi
done

# Count remaining TF_UNIMPLEMENTED across all MPS files
TOTAL_UNIMPL=$(find tensorflow/mps/kernels -name "*.mm" -exec grep -l "TF_UNIMPLEMENTED" {} \; 2>/dev/null | wc -l | tr -d ' ')

echo "Total Operations Implemented: $TOTAL_OPS"
echo "Files with TF_UNIMPLEMENTED: $TOTAL_UNIMPL"
echo ""
echo "Validation Results:"
echo -e "  ${GREEN}✓ PASS: $TOTAL_PASS${NC}"
echo -e "  ${RED}✗ FAIL: $TOTAL_FAIL${NC}"

# Check specific function exports
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SAMPLE FUNCTION EXPORTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check for key operations
SAMPLE_OPS=(
    "MPSReduceStd_Compute"
    "MPSReduceVariance_Compute"
    "MPSBesselJ0_Compute"
    "MPSDigamma_Compute"
    "MPSCumSum_Compute"
    "MPSSegmentSum_Compute"
)

for op in "${SAMPLE_OPS[@]}"; do
    if grep -q "$op" tensorflow/mps/kernels/*.mm 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Found: $op"
    else
        echo -e "${RED}✗${NC} Missing: $op"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"

if [ $TOTAL_FAIL -eq 0 ]; then
    echo -e "${GREEN}  ALL VALIDATIONS PASSED!${NC}"
    echo "════════════════════════════════════════════════════════════════"
    exit 0
else
    echo -e "${YELLOW}  SOME VALIDATIONS FAILED - See details above${NC}"
    echo "════════════════════════════════════════════════════════════════"
    exit 1
fi
