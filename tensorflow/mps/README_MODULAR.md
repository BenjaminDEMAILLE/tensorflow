# MPS Backend - Modular Structure

This directory contains the modular implementation of the TensorFlow MPS (Metal Performance Shaders) backend, following CUDA's architectural pattern.

## 📁 Directory Structure

```
tensorflow/mps/
├── device/                      # Device & StreamExecutor infrastructure
│   ├── mps_device.h            # Device, Stream, Event structures
│   ├── mps_device.mm           # Device implementation  
│   └── mps_device_factory.mm   # Plugin registration & SE_InitPlugin
│
├── ops/                         # Operation registration layer
│   ├── mps_ops_registry.h      # Registration macros & forward declarations
│   └── mps_ops_registry.mm     # Main registration orchestrator
│
├── kernels/                     # Kernel implementations by category
│   ├── mps_elementwise_ops.mm  # 41 ops: Add, Mul, Sin, Cos, Exp, Log, etc.
│   ├── mps_activation_ops.mm   # 9 ops: Relu, Gelu, Swish, LeakyRelu, etc.
│   ├── mps_comparison_ops.mm   # 6 ops: Equal, Less, Greater, etc.
│   ├── mps_logical_ops.mm      # 3 ops: And, Or, Not
│   ├── mps_reduction_ops.mm    # 7 ops: Sum, Mean, Max, Min, Prod, etc.
│   ├── mps_tensor_ops.mm       # 15 ops: Reshape, Transpose, Concat, etc.
│   ├── mps_indexing_ops.mm     # 5 ops: Split, Gather, Scatter, etc.
│   ├── mps_nn_ops.mm           # 7 ops: Conv2D, MatMul, Softmax, etc.
│   └── mps_utility_ops.mm      # 3 ops: OneHot, Range, IsFinite
│
├── utils/                       # Shared utilities
│   └── mps_utils.h             # DType conversion, ObjC bridges
│
├── tests/                       # Modular test suite
│   ├── elementwise_ops_test.py
│   ├── nn_ops_test.py
│   └── comparison_ops_test.py
│
├── BUILD                        # Bazel build configuration (modular)
├── README.md                    # This file
└── mps_pluggable_device_plugin.mm  # LEGACY: Original monolithic file (6058 lines)
```

## 🏗️ Architecture

### Design Principles

1. **Separation of Concerns**
   - **device/**: StreamExecutor platform, memory, streams
   - **ops/**: Kernel registration infrastructure
   - **kernels/**: Actual operation implementations
   - **utils/**: Shared utilities (dtype conversion, etc.)

2. **Modular Compilation**
   - Each kernel category is a separate compilation unit
   - Parallel builds possible
   - Easier to maintain and test

3. **CUDA-like Structure**
   - Follows `tensorflow/core/common_runtime/gpu/` pattern
   - Similar to `tensorflow/core/kernels/*_gpu.cc` organization
   - Familiar to TensorFlow contributors

## 🔧 Build System

### Build the Plugin

```bash
# Build all components
bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib

# Build individual components
bazel build //tensorflow/mps:mps_device
bazel build //tensorflow/mps:mps_elementwise_ops
bazel build //tensorflow/mps:mps_nn_ops
```

### Build Targets

| Target | Description | Lines |
|--------|-------------|-------|
| `mps_utils` | DType conversion utilities | ~120 |
| `mps_device` | Device infrastructure | ~400 |
| `mps_device_factory` | Plugin entry point | ~200 |
| `mps_ops_registry` | Registration orchestrator | ~60 |
| `mps_elementwise_ops` | Elementwise kernels | ~800 |
| `mps_activation_ops` | Activation kernels | ~300 |
| `mps_comparison_ops` | Comparison kernels | ~200 |
| `mps_logical_ops` | Logical kernels | ~150 |
| `mps_reduction_ops` | Reduction kernels | ~400 |
| `mps_tensor_ops` | Tensor manipulation kernels | ~600 |
| `mps_indexing_ops` | Indexing kernels | ~500 |
| `mps_nn_ops` | Neural network kernels | ~700 |
| `mps_utility_ops` | Utility kernels | ~200 |

**Total: ~4,730 lines** (vs. 6,058 monolithic)

## 📝 Adding New Operations

### Step 1: Choose Category

Determine which kernel file should contain your op:
- Math operations → `kernels/mps_elementwise_ops.mm`
- Activations → `kernels/mps_activation_ops.mm`
- Tensor manipulation → `kernels/mps_tensor_ops.mm`
- etc.

### Step 2: Implement Kernel

```objc++
// In kernels/mps_elementwise_ops.mm

void* MPSMyNewOp_Create(TF_OpKernelConstruction* ctx) {
  // Initialization
  return nullptr;
}

void MPSMyNewOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  // Implementation
}

void MPSMyNewOp_Delete(void* kernel) {
  // Cleanup
}
```

### Step 3: Register Kernel

```objc++
// In kernels/mps_elementwise_ops.mm

void RegisterElementwiseOps(const char* platform_name, TF_Status* status) {
  // ... existing registrations ...
  
  REGISTER_MPS_UNARY_OP_3DTYPE(MyNewOp, platform_name, status);
}
```

### Step 4: Add Test

```python
# In tests/elementwise_ops_test.py

def test_my_new_op(self):
    x = tf.constant([1.0, 2.0, 3.0])
    with tf.device('/device:MPS:0'):
        result = tf.my_new_op(x)
    self.assertAllClose(result, expected)
```

## 🧪 Testing

### Run All Tests

```bash
bazel test //tensorflow/mps/...
```

### Run Category Tests

```bash
bazel test //tensorflow/mps:elementwise_ops_test
bazel test //tensorflow/mps:nn_ops_test
bazel test //tensorflow/mps:comparison_ops_test
```

### Smoke Test

```bash
python3 ci/smoke/mps_ops_smoke.py
```

## 📊 Comparison: Monolithic vs Modular

| Aspect | Monolithic | Modular |
|--------|-----------|---------|
| **File Size** | 1 file @ 6,058 lines | 13 files @ ~400 lines avg |
| **Build Time** | Serial (1 unit) | Parallel (13 units) |
| **Maintainability** | Difficult to navigate | Easy to find ops |
| **Testing** | All-or-nothing | Per-category |
| **Collaboration** | Merge conflicts | Independent work |
| **Code Review** | Large diffs | Focused changes |

## 🔄 Migration from Monolithic

### Current State

The original `mps_pluggable_device_plugin.mm` (6,058 lines) is preserved for reference.

### Migration Plan

1. ✅ **Phase 1**: Create infrastructure
   - device/, ops/, kernels/, utils/ structure
   - Registration macros
   - BUILD system

2. 🔄 **Phase 2**: Extract kernels (in progress)
   - Move op implementations to category files
   - Verify each category builds & tests pass

3. ⏳ **Phase 3**: Deprecate monolithic
   - Update main BUILD to use modular structure
   - Archive monolithic file
   - Update documentation

4. ⏳ **Phase 4**: Optimization
   - MPSGraph optimizations per category
   - Custom Metal shaders where needed
   - Performance tuning

## 📚 References

- **TensorFlow CUDA Backend**: `tensorflow/core/common_runtime/gpu/`
- **CUDA Kernels**: `tensorflow/core/kernels/*_gpu.cc`
- **MPS Documentation**: Apple Metal Performance Shaders Graph docs
- **Original Implementation**: `mps_pluggable_device_plugin.mm`

## 🎯 Benefits

1. **Developer Productivity**
   - Find ops faster (category-based organization)
   - Smaller compilation units (faster incremental builds)
   - Parallel development (no merge conflicts on single file)

2. **Code Quality**
   - Easier code reviews (focused diffs)
   - Better test coverage (per-category tests)
   - Clearer ownership (CODEOWNERS per category)

3. **Performance**
   - Parallel compilation (13x potential speedup)
   - Targeted optimizations per category
   - Easier to profile and benchmark

4. **Maintainability**
   - Standard TensorFlow structure (follows CUDA pattern)
   - Easier onboarding for new contributors
   - Better documentation organization

---

**Status**: Modular structure defined, infrastructure complete, kernel extraction in progress

**Last Updated**: October 24, 2025
