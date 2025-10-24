# MPS Backend - Build & Test Guide

## 🔧 Prerequisites

### Required
- **macOS** with Apple Silicon (M1/M2/M3) or AMD GPU
- **Bazel** 8.0+ (already installed: `brew install bazel`)
- **Python** 3.9-3.12
- **Xcode Command Line Tools**

### Install dependencies
```bash
# Install Bazel (already done)
brew install bazel

# Install Python dependencies
pip install numpy packaging
```

## 🏗️ Building TensorFlow with MPS Backend

### Option 1: Quick Build (Plugin only)

Build just the MPS plugin shared library:

```bash
cd /Users/benjamin/tensorflow

# Build the plugin
bazel build --config=macos_arm64 \
    //tensorflow/mps:libtensorflow_mps_plugin.dylib

# Plugin location
ls -lh bazel-bin/tensorflow/mps/libtensorflow_mps_plugin.dylib
```

### Option 2: Full Build (Recommended)

Build TensorFlow with the MPS backend integrated:

```bash
# Configure TensorFlow (first time only)
./configure
# Answer prompts:
#   - Python path: (default)
#   - CUDA: N
#   - ROCm: N
#   - Optimization flags: -march=native

# Build Python wheel
bazel build --config=macos_arm64 \
    --define=no_tensorflow_py_deps=true \
    --copt=-O3 \
    //tensorflow/tools/pip_package:build_pip_package

# Create wheel
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install
pip install --force-reinstall /tmp/tensorflow_pkg/tensorflow-*.whl
```

### Option 3: Use build script

```bash
# Make executable
chmod +x build_mps.sh

# Run build
./build_mps.sh
```

## 🧪 Running Tests

### Smoke Tests
Quick validation of all ops:

```bash
# Run smoke test
python3 ci/smoke/mps_ops_smoke.py

# Or via Bazel
bazel test //tensorflow/mps:smoke_test --test_output=all
```

Expected output:
```
TF version: 2.x.x
Physical devices: [PhysicalDevice(name='/physical_device:MPS:0', device_type='MPS')]
LeakyRelu: [-0.2  -0.1   0.    1.    6.  ]
...
✅ All smoke tests passed!
```

### Unit Tests
Comprehensive op testing:

```bash
# Run all MPS tests
bazel test //tensorflow/mps:ops_test --test_output=all

# Run specific test
bazel test //tensorflow/mps:ops_test --test_filter=*Comparison* --test_output=all
```

### Performance Benchmarks
Measure op performance:

```bash
# Run benchmarks
python3 tensorflow/mps/benchmark_ops.py

# Or via Bazel
bazel run //tensorflow/mps:benchmark_ops
```

Expected output:
```
🚀 MPS Operations Performance Benchmark
========================================
TensorFlow version: 2.x.x
Devices: [PhysicalDevice(..., device_type='MPS')]

📊 Comparison Operations
========================================
  Array size: 1000
  Equal                                   :   0.0234 ms
  Less                                    :   0.0198 ms
...
```

## 🐛 Debugging

### Check MPS Plugin Load
```python
import tensorflow as tf
print("TF version:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
# Should show: [PhysicalDevice(..., device_type='MPS')]
```

### Enable Verbose Logging
```bash
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE=mps_pluggable_device_plugin=3

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

### Build with Debug Symbols
```bash
bazel build --config=macos_arm64 \
    --compilation_mode=dbg \
    --copt=-g \
    //tensorflow/mps:libtensorflow_mps_plugin.dylib
```

## 📊 Validation Checklist

After building, verify:

- [ ] `bazel-bin/tensorflow/mps/libtensorflow_mps_plugin.dylib` exists
- [ ] Plugin size is reasonable (~500KB - 2MB)
- [ ] Smoke test passes: `python3 ci/smoke/mps_ops_smoke.py`
- [ ] Unit tests pass: `bazel test //tensorflow/mps:ops_test`
- [ ] TF imports correctly: `python3 -c "import tensorflow as tf; print(tf.__version__)"`
- [ ] MPS device detected: Check `tf.config.list_physical_devices()`

## 🚀 Quick Start (One Command)

```bash
# Build, test, and install
bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib && \
bazel test //tensorflow/mps:smoke_test --test_output=all && \
echo "✅ MPS Backend ready!"
```

## 📈 Performance Tips

### Optimize Build
```bash
# Use more CPU cores
bazel build --jobs=8 ...

# Enable compiler optimizations
bazel build --copt=-O3 --copt=-march=native ...

# Disable assertions in production
bazel build --compilation_mode=opt ...
```

### Profile Operations
```bash
# Profile with Instruments
instruments -t "Time Profiler" python3 tensorflow/mps/benchmark_ops.py

# Or use TensorFlow Profiler
python3 -c "import tensorflow as tf; tf.profiler.experimental.start('logdir')"
```

## 🔄 Iterative Development

When modifying ops:

```bash
# 1. Edit code
vim tensorflow/mps/mps_pluggable_device_plugin.mm

# 2. Rebuild plugin only (fast)
bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib

# 3. Run smoke test
python3 ci/smoke/mps_ops_smoke.py

# 4. Run specific test
bazel test //tensorflow/mps:ops_test --test_filter=*YourOp*
```

## ⚡ Common Issues

### "Plugin not found"
- Ensure plugin is in wheel: `tensorflow-plugins/libtensorflow_mps_plugin.dylib`
- Check TF_PLUGGABLE_DEVICE_LIBRARY_PATH

### "Symbol not found"
- Rebuild with: `bazel clean --expunge`
- Check linkage: `otool -L bazel-bin/tensorflow/mps/libtensorflow_mps_plugin.dylib`

### Build fails with ObjC++ errors
- Ensure `-fobjc-arc` is in copts
- Check Xcode version: `xcode-select --version`

## 📚 Next Steps

1. **Run full test suite**: `bazel test //tensorflow/mps/...`
2. **Profile performance**: Run benchmarks and compare with CPU/CUDA
3. **Start refactoring**: Follow `REFACTORING_PLAN.md`
4. **Add GPU paths**: Implement MPSGraph for more ops (currently CPU fallbacks)

## 🤝 Contributing

When adding new ops:

1. Implement in `mps_pluggable_device_plugin.mm`
2. Register kernel with type constraints
3. Add test case to `ops_test.py`
4. Add smoke test to `ci/smoke/mps_ops_smoke.py`
5. Add benchmark to `benchmark_ops.py`
6. Update documentation
7. Commit with clear message

Example commit:
```
MPS: add Foo op with bar feature

- Implement MPSFoo_Compute with broadcasting
- Register for float/half/bfloat16
- Add unit tests covering edge cases
- Benchmark shows 2x speedup vs CPU
```
