# TensorFlow MPS (Metal Performance Shaders) Plugin

This directory contains a **native MPS PluggableDevice backend** for TensorFlow on macOS Apple Silicon, eliminating the dependency on the external `tensorflow-metal` package.

## Overview

The MPS plugin provides GPU acceleration for TensorFlow operations using Apple's Metal Performance Shaders framework. It implements:

- **StreamExecutor C API** integration for device management, memory allocation, and async execution
- **Kernel C API** registration for core TensorFlow operations
- **Native Metal compute shaders** for elementwise operations
- **MPSMatrix** for optimized linear algebra (MatMul)
- **MPSGraph** for complex operations (Conv2D, bfloat16 support)

## Device Type

The plugin registers device type **`MPS`** (distinct from `GPU`) to avoid kernel registration conflicts with CUDA/ROCm.

**Device enumeration:**
```python
import tensorflow as tf

# List all devices
devices = tf.config.list_physical_devices()
print(devices)

# Check for MPS devices
mps_devices = tf.config.list_physical_devices('MPS')
if mps_devices:
    print(f"MPS device available: {mps_devices[0]}")
```

## Supported Operations

### Operation Coverage

| Operation | float32 | float16 (half) | bfloat16 | Implementation |
|-----------|---------|----------------|----------|----------------|
| **Identity** | ✅ | ✅ | ✅ | Forward/copy |
| **Relu** | ✅ GPU | ✅ GPU | ✅ Host | Metal shader (float/half), host convert (bf16) |
| **AddV2** | ✅ GPU | ✅ GPU | ❌ | Metal shader + scalar broadcast |
| **Mul** | ✅ GPU | ✅ GPU | ❌ | Metal shader + scalar broadcast |
| **Maximum** | ✅ GPU | ✅ GPU | ❌ | Metal shader + scalar broadcast |
| **Minimum** | ✅ GPU | ✅ GPU | ❌ | Metal shader + scalar broadcast |
| **Sigmoid** | ✅ GPU | ✅ GPU | ❌ | Metal shader |
| **Tanh** | ✅ GPU | ✅ GPU | ❌ | Metal shader |
| **MatMul** | ✅ GPU | ✅ GPU | ✅ GPU | MPSMatrixMultiplication (f32/f16), MPSGraph (bf16) |
| **Conv2D** | ✅ GPU | ❌ | ❌ | MPSGraph (NHWC, SAME/VALID padding, stride, dilation) |

**Legend:**
- ✅ GPU: Runs on Metal GPU with native dtype
- ✅ Host: CPU fallback with dtype conversion
- ❌: Not yet implemented

### Features

- **Transpose support** for MatMul (`transpose_a`, `transpose_b`)
- **Scalar broadcasting** for elementwise ops (Add, Mul, Maximum, Minimum)
- **SAME/VALID padding** for Conv2D
- **Stride and dilation** support for Conv2D
- **Mixed precision** training (float32/float16/bfloat16)

## Usage

### Eager Execution

```python
import tensorflow as tf

# Place operations on MPS device
with tf.device('/device:MPS:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
    
    # MatMul
    c = tf.matmul(a, b)
    print(c.numpy())
    
    # Elementwise ops
    d = tf.nn.relu(a + b)
    print(d.numpy())
    
    # Conv2D
    x = tf.random.uniform([1, 28, 28, 3], dtype=tf.float32)
    filters = tf.random.uniform([3, 3, 3, 16], dtype=tf.float32)
    y = tf.nn.conv2d(x, filters, strides=[1, 1, 1, 1], padding='SAME')
    print(y.shape)
```

### Graph Mode

```python
@tf.function
def compute(a, b):
    with tf.device('/device:MPS:0'):
        return tf.matmul(a, b) + tf.nn.relu(a)

result = compute(a, b)
```

### Mixed Precision

```python
# Float16 for faster training
with tf.device('/device:MPS:0'):
    a_f16 = tf.constant([[1.0, 2.0]], dtype=tf.float16)
    b_f16 = tf.constant([[3.0], [4.0]], dtype=tf.float16)
    c_f16 = tf.matmul(a_f16, b_f16)  # Native half precision on GPU

# Bfloat16 for MatMul (native via MPSGraph)
with tf.device('/device:MPS:0'):
    a_bf16 = tf.constant([[1.0, 2.0]], dtype=tf.bfloat16)
    b_bf16 = tf.constant([[3.0], [4.0]], dtype=tf.bfloat16)
    c_bf16 = tf.matmul(a_bf16, b_bf16)  # Native bfloat16 on GPU
```

## Implementation Details

### StreamExecutor Integration

The plugin implements the TensorFlow StreamExecutor C API (`SE_InitPlugin`):

- **Platform**: Name "MPS", device type "MPS"
- **Memory**: Private MTLBuffer allocation, BFC allocator enabled
- **Streams**: MTLCommandQueue per stream
- **Events**: Semaphore-based synchronization
- **Memcpy**: MTLBlitCommandEncoder for async H2D/D2H/D2D transfers
- **Timers**: Host-based profiling via `mach_absolute_time`

### Kernel Implementations

#### Metal Shaders
Elementwise operations (Relu, Add, Mul, etc.) use dynamically compiled Metal Shading Language (MSL) compute kernels:
```metal
kernel void relu_k(const device float* in [[buffer(0)]],
                   device float* out [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    out[gid] = max(in[gid], 0.0f);
}
```

#### MPSMatrix
MatMul uses `MPSMatrixMultiplication` for optimized BLAS-like operations:
- Float32: `MPSDataTypeFloat32`
- Float16: `MPSDataTypeFloat16` (native GPU half precision)

#### MPSGraph
Complex operations use MPSGraph for automatic optimization:
- **MatMul (bfloat16)**: Native `MPSDataTypeBFloat16` support
- **Conv2D**: Full convolution with NHWC layout, configurable padding/stride/dilation

### Dtype Conversion

For unsupported dtypes (e.g., bfloat16 elementwise ops), the implementation falls back to host conversion:
```cpp
// bfloat16 -> float32 -> compute -> bfloat16
float HalfToFloat(uint16_t h);
uint16_t FloatToHalf(float f);
float BFloat16ToFloat(uint16_t bf);
uint16_t FloatToBFloat16(float f);
```

## Building

The plugin is automatically included in macOS wheels:

```bash
# Build TensorFlow with MPS plugin
bazel build //tensorflow/tools/pip_package:wheel
```

The plugin dylib is packaged under `tensorflow-plugins/` and auto-loaded on macOS.

## Testing

Run the test suite:

```bash
# Unit tests (manual on macOS with Metal)
bazel test //tensorflow/mps:mps_device_test --config=macos_arm64

# Quick Python test
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('MPS'))"
```

## Limitations

- **NHWC only** for Conv2D (NCHW not supported)
- **No explicit channels_last** optimization yet
- **No DepthwiseConv2D** (planned)
- **Limited broadcasting** (only scalar broadcast for elementwise ops)
- **No gradient ops** registered yet (forward pass only)

## Roadmap

- [ ] Extend Conv2D to float16
- [ ] Add DepthwiseConv2D via MPSGraph
- [ ] Implement gradient kernels (Conv2DBackprop, etc.)
- [ ] Full NumPy-style broadcasting for elementwise ops
- [ ] Pooling operations (MaxPool, AvgPool)
- [ ] Batch normalization
- [ ] Softmax
- [ ] CI integration for macOS ARM64

## Performance

Expected speedups vs CPU on Apple M1/M2/M3:
- **MatMul (large)**: 3-10x (depending on size, dtype)
- **Conv2D**: 5-15x (typical CNN workloads)
- **Elementwise ops**: 2-5x (memory bandwidth limited)

Benchmark with:
```python
import tensorflow as tf
import time

with tf.device('/device:MPS:0'):
    a = tf.random.uniform([1000, 1000], dtype=tf.float32)
    b = tf.random.uniform([1000, 1000], dtype=tf.float32)
    
    # Warmup
    for _ in range(10):
        _ = tf.matmul(a, b)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        c = tf.matmul(a, b)
    elapsed = time.time() - start
    print(f"MPS MatMul: {elapsed:.3f}s for 100 iterations")
```

## License

Copyright 2025 The TensorFlow Authors. Apache License 2.0.
