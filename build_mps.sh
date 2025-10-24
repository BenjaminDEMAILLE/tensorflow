#!/bin/bash
# Script de build pour TensorFlow avec backend MPS
# Usage: ./build_mps.sh

set -e

echo "🚀 Building TensorFlow with MPS Backend..."
echo "============================================"

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "WORKSPACE" ]; then
    echo "❌ Error: Must be run from TensorFlow root directory"
    exit 1
fi

# Vérifier que Bazel est installé
if ! command -v bazel &> /dev/null && ! command -v bazelisk &> /dev/null; then
    echo "❌ Error: Bazel or Bazelisk not found"
    echo "Install with: brew install bazelisk"
    exit 1
fi

BAZEL_CMD=$(command -v bazelisk || command -v bazel)

# Configuration
export TF_NEED_CUDA=0
export TF_NEED_ROCM=0
export TF_NEED_OPENCL=0
export TF_DOWNLOAD_CLANG=0
export CC_OPT_FLAGS="-march=native"
export TF_SET_ANDROID_WORKSPACE=0

echo "📋 Configuration:"
echo "  - Python: $(python3 --version)"
echo "  - Bazel: $($BAZEL_CMD version | head -1)"
echo "  - Platform: macOS (Apple Silicon)"
echo ""

# Configure si nécessaire
if [ ! -f ".tf_configure.bazelrc" ]; then
    echo "⚙️  Running ./configure..."
    ./configure
fi

# Build le plugin MPS
echo "🔨 Building MPS plugin..."
$BAZEL_CMD build --config=macos_arm64 \
    --define=no_tensorflow_py_deps=true \
    --copt=-O3 \
    --copt=-DNDEBUG \
    //tensorflow/mps:mps_pluggable_device_plugin.so

# Build Python wheel
echo "🐍 Building Python wheel..."
$BAZEL_CMD build --config=macos_arm64 \
    --define=no_tensorflow_py_deps=true \
    --copt=-O3 \
    //tensorflow/tools/pip_package:build_pip_package

# Create wheel
echo "📦 Creating wheel..."
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

echo ""
echo "✅ Build complete!"
echo "📦 Wheel location: /tmp/tensorflow_pkg/"
echo "🔧 Install with: pip install /tmp/tensorflow_pkg/tensorflow-*.whl"
