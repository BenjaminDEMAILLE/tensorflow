# MPS Backend - Complete Migration Guide

## 🎯 Current Status

### ✅ Phase 1: Infrastructure (COMPLETE)
- Modular directory structure created
- Device infrastructure implemented
- Registration layer with macros
- Build system prepared
- Documentation complete

### 🔄 Phase 2: Kernel Extraction (READY FOR EXECUTION)
**Status**: Infrastructure ready, awaiting manual extraction

The monolithic file (`mps_pluggable_device_plugin.mm`, 6058 lines) contains all 110+ operation implementations. These need to be copied into the appropriate category files.

## 📋 Phase 2: Detailed Extraction Plan

### Step-by-Step Guide

#### 1. Elementwise Operations (41 ops)
**Target**: `kernels/mps_elementwise_ops.mm`

**Operations to extract**:
- Binary: Add, AddV2, Sub, Mul, Div, Pow, Maximum, Minimum, SquaredDifference
- Division: RealDiv, FloorDiv, FloorMod, TruncateDiv, TruncateMod, Mod
- Unary: Neg, Abs, Sign, Sqrt, Rsqrt, Square, Reciprocal
- Transcendental: Exp, Expm1, Log, Log1p, Sin, Cos, Tan, Asin, Acos, Atan
- Hyperbolic: Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
- Rounding: Ceil, Floor, Round, Rint
- Special: Erf

**Search patterns in monolith**:
```cpp
void* MPSAdd_Create
void MPSAdd_Compute
void MPSAdd_Delete
// ... for each op
```

**Registration pattern**:
```cpp
void RegisterElementwiseOps(const char* platform_name, TF_Status* status) {
  REGISTER_MPS_BINARY_OP_3DTYPE(Add, platform_name, status);
  REGISTER_MPS_BINARY_OP_3DTYPE(Mul, platform_name, status);
  REGISTER_MPS_UNARY_OP_3DTYPE(Sin, platform_name, status);
  // ... etc
}
```

#### 2. Activation Operations (9 ops)
**Target**: `kernels/mps_activation_ops.mm`

**Operations**: Relu, Relu6, Elu, Selu, LeakyRelu, Gelu, Swish, Softplus, Softsign

**Note**: Sigmoid and Tanh are already in monolith, extract them here too

#### 3. Comparison Operations (6 ops) - ✅ ALREADY DONE
**Target**: `kernels/mps_comparison_ops.mm`

This file is already implemented as an example!

#### 4. Logical Operations (3 ops)
**Target**: `kernels/mps_logical_ops.mm`

**Operations**: LogicalAnd, LogicalOr, LogicalNot

**Special note**: These work on bool tensors with broadcasting

#### 5. Reduction Operations (7 ops)
**Target**: `kernels/mps_reduction_ops.mm`

**Operations**: Sum, Mean, Max, Min, Prod, All, Any

**Note**: ArgMax and ArgMin might also be here

#### 6. Tensor Operations (15 ops)
**Target**: `kernels/mps_tensor_ops.mm`

**Operations**:
- Shape manipulation: Cast, Reshape, Transpose, Concat, ConcatV2
- Slicing: Slice, StridedSlice (complex - supports all masks)
- Creation: Fill, ZerosLike, OnesLike
- Padding: Pad, MirrorPad
- Others: Tile, Select, SelectV2, ClipByValue, Identity, Shape, Size, Rank

#### 7. Indexing Operations (5 ops)
**Target**: `kernels/mps_indexing_ops.mm`

**Operations**: Split, SplitV, GatherV2, GatherNd, TensorScatterUpdate, TensorScatterAdd, ScatterNd

#### 8. Neural Network Operations (7 ops)
**Target**: `kernels/mps_nn_ops.mm`

**Operations**: Conv2D, DepthwiseConv2dNative, MaxPool, AvgPool, MatMul, FusedBatchNormV3, Softmax, LogSoftmax

**Note**: These are the most complex operations with many attributes

#### 9. Utility Operations (3 ops)
**Target**: `kernels/mps_utility_ops.mm`

**Operations**: OneHot, Range, IsFinite

## 🔧 Extraction Procedure

### Automated Approach (Recommended)

Use the provided extraction script:

```bash
cd /Users/benjamin/tensorflow
python3 tensorflow/mps/tools/extract_kernels.py
```

This will:
1. Parse the monolithic file
2. Extract all function implementations
3. Group by category
4. Generate registration calls
5. Create complete category files

### Manual Approach

For each operation category:

1. **Open monolithic file**:
   ```bash
   code tensorflow/mps/mps_pluggable_device_plugin.mm
   ```

2. **Search for operation** (e.g., "MPSAdd"):
   - Find `MPSAdd_Create`
   - Find `MPSAdd_Compute`
   - Find `MPSAdd_Delete`
   - Find registration calls

3. **Copy to category file**:
   - Copy all three functions
   - Copy any helper functions
   - Copy registration logic

4. **Update registration**:
   ```cpp
   void RegisterElementwiseOps(const char* platform_name, TF_Status* status) {
     REGISTER_MPS_BINARY_OP_3DTYPE(Add, platform_name, status);
     // Add more...
   }
   ```

5. **Test compilation**:
   ```bash
   bazel build //tensorflow/mps:mps_elementwise_ops
   ```

## 📊 Phase 3: Build Migration

### Step 1: Update BUILD File

Currently, BUILD uses monolithic approach. After extraction, update to use modular targets:

```python
# In tensorflow/mps/BUILD

objc_library(
    name = "mps_device_factory",
    srcs = ["device/mps_device_factory.mm"],
    deps = [
        ":mps_device",
        ":mps_utils",
        ":mps_ops_registry",
    ],
    alwayslink = 1,
)

objc_library(
    name = "mps_ops_registry",
    srcs = ["ops/mps_ops_registry.mm"],
    hdrs = ["ops/mps_ops_registry.h"],
    deps = [
        ":mps_elementwise_ops",
        ":mps_activation_ops",
        ":mps_comparison_ops",
        ":mps_logical_ops",
        ":mps_reduction_ops",
        ":mps_tensor_ops",
        ":mps_indexing_ops",
        ":mps_nn_ops",
        ":mps_utility_ops",
    ],
    alwayslink = 1,
)

# All kernel category libraries...
objc_library(
    name = "mps_elementwise_ops",
    srcs = ["kernels/mps_elementwise_ops.mm"],
    deps = [":mps_ops_registry_hdr", ":mps_utils"],
    alwayslink = 1,
)
# ... etc for all categories

cc_binary(
    name = "libtensorflow_mps_plugin.dylib",
    deps = [":mps_device_factory"],
    linkshared = 1,
    linkopts = [
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
        "-framework", "MetalPerformanceShadersGraph",
        "-framework", "Foundation",
    ],
)
```

### Step 2: Verify Compilation

```bash
# Build individual components
bazel build //tensorflow/mps:mps_utils
bazel build //tensorflow/mps:mps_device
bazel build //tensorflow/mps:mps_elementwise_ops
# ... etc

# Build final plugin
bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib
```

### Step 3: Run Tests

```bash
# Smoke test
python3 ci/smoke/mps_ops_smoke.py

# Unit tests
bazel test //tensorflow/mps:ops_test

# Benchmarks
python3 tensorflow/mps/benchmark_ops.py
```

## 🎯 Phase 4: Finalization

### 1. Archive Monolithic File

```bash
mv tensorflow/mps/mps_pluggable_device_plugin.mm \
   tensorflow/mps/mps_pluggable_device_plugin.mm.legacy
```

### 2. Update Documentation

Update `README.md`:
```markdown
# TensorFlow MPS Backend

## Architecture

This backend uses a modular CUDA-like structure:

- `device/`: StreamExecutor platform
- `ops/`: Kernel registration
- `kernels/`: Operation implementations (9 categories)
- `utils/`: Shared utilities

See [README_MODULAR.md](README_MODULAR.md) for details.
```

### 3. Performance Validation

```bash
# Run benchmarks before and after
python3 tensorflow/mps/benchmark_ops.py > results_before.txt

# After migration
python3 tensorflow/mps/benchmark_ops.py > results_after.txt

# Compare
diff results_before.txt results_after.txt
```

### 4. Create CODEOWNERS

```
# MPS Backend ownership
/tensorflow/mps/device/ @mps-team
/tensorflow/mps/ops/ @mps-team
/tensorflow/mps/kernels/mps_elementwise_ops.mm @mps-elementwise-team
/tensorflow/mps/kernels/mps_nn_ops.mm @mps-nn-team
# ... etc
```

## ⏱️ Estimated Timeline

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| **Phase 2.1** | Extract elementwise (41 ops) | 2-3 hours |
| **Phase 2.2** | Extract activation (9 ops) | 1 hour |
| **Phase 2.3** | Extract logical/reduction/tensor (25 ops) | 2-3 hours |
| **Phase 2.4** | Extract indexing/nn/utility (15 ops) | 2-3 hours |
| **Phase 3** | Update BUILD, test compilation | 1-2 hours |
| **Phase 4** | Testing, benchmarking, docs | 2-3 hours |
| **Total** | Complete migration | 10-15 hours |

## 🚀 Quick Start Commands

```bash
# 1. Extract all kernels
python3 tensorflow/mps/tools/extract_kernels.py

# 2. Verify structure
find tensorflow/mps -name "*.mm" | wc -l  # Should show ~15 files

# 3. Update BUILD
# Edit tensorflow/mps/BUILD to use modular targets

# 4. Test build
bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib

# 5. Run tests
python3 ci/smoke/mps_ops_smoke.py
bazel test //tensorflow/mps:ops_test

# 6. Benchmark
python3 tensorflow/mps/benchmark_ops.py

# 7. Commit
git add tensorflow/mps/
git commit -m "MPS: complete modular refactoring (Phase 2-4)"
```

## 📚 Reference Files

- **Structure**: `README_MODULAR.md`
- **Status**: `REFACTORING_SUMMARY.md`
- **This guide**: `MIGRATION_GUIDE.md`
- **Extraction tool**: `tools/extract_kernels.py`
- **Example**: `kernels/mps_comparison_ops.mm` (complete implementation)

## ❓ Troubleshooting

### Build Errors

**Problem**: Undefined symbols
**Solution**: Ensure all ops have Create/Compute/Delete exported with `extern "C"` or proper linkage

**Problem**: Duplicate symbols
**Solution**: Check for duplicate implementations across category files

**Problem**: Missing includes
**Solution**: Add necessary headers to category files

### Runtime Errors

**Problem**: Op not found
**Solution**: Verify registration in `RegisterXXXOps()` function

**Problem**: Wrong output
**Solution**: Compare CPU fallback vs MPSGraph implementation

### Performance Regression

**Problem**: Slower than before
**Solution**: Check if MPSGraph optimizations are enabled, not just CPU fallback

## 🎉 Success Criteria

- ✅ All 110+ ops extracted and organized
- ✅ Build succeeds: `bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib`
- ✅ All tests pass: `bazel test //tensorflow/mps/...`
- ✅ Smoke test passes: `python3 ci/smoke/mps_ops_smoke.py`
- ✅ Performance maintained or improved
- ✅ Documentation updated
- ✅ Monolithic file archived

---

**Created**: October 24, 2025  
**Status**: Ready for Phase 2 execution  
**Next**: Run `python3 tensorflow/mps/tools/extract_kernels.py`
