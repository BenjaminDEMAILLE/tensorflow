# 🚀 Guide d'Utilisation - MPS Backend TensorFlow

## 📋 Table des Matières

1. [Prérequis](#prérequis)
2. [Compilation](#compilation)
3. [Installation](#installation)
4. [Utilisation](#utilisation)
5. [Tests](#tests)
6. [Dépannage](#dépannage)

---

## 🔧 Prérequis

### Système
- **macOS**: 11.0 (Big Sur) ou supérieur
- **GPU**: Apple Silicon (M1/M2/M3) ou AMD GPU avec support Metal
- **Xcode**: 13.0 ou supérieur (avec Command Line Tools)

### Vérification
```bash
# Vérifier macOS version
sw_vers

# Vérifier Xcode
xcodebuild -version

# Vérifier support Metal
system_profiler SPDisplaysDataType | grep Metal
```

### Outils de Build
```bash
# Installer Homebrew (si nécessaire)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installer Bazel
brew install bazel

# Vérifier Bazel
bazel version
```

### Python
```bash
# Python 3.9 ou supérieur requis
python3 --version

# Installer les dépendances Python
pip3 install numpy six wheel packaging requests opt_einsum
pip3 install keras_preprocessing --no-deps
```

---

## 🏗️ Compilation

### Configuration
```bash
# Naviguer vers le répertoire TensorFlow
cd /Users/benjamin/tensorflow

# Configurer le build
./configure

# Répondre aux questions de configuration:
# - Python location: /usr/bin/python3
# - Python library: /usr/local/lib/python3.x/site-packages
# - Optimizations: Y
# - XLA: N (optionnel)
# - MPS: Y
# - Autres options: selon vos besoins
```

### Build du Plugin MPS
```bash
# Build du plugin MPS
bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib \
    --config=macos \
    --config=opt \
    --verbose_failures

# Build des kernels
bazel build //tensorflow/mps/kernels:mps_kernels \
    --config=macos \
    --config=opt

# Temps estimé: 15-45 minutes selon votre machine
```

### Build TensorFlow avec MPS
```bash
# Build complet de TensorFlow avec support MPS
bazel build //tensorflow/tools/pip_package:build_pip_package \
    --config=macos \
    --config=opt

# Créer le package wheel
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

---

## 📦 Installation

### Installation du Plugin
```bash
# Copier le plugin dans le dossier TensorFlow
mkdir -p ~/.tensorflow/plugins
cp bazel-bin/tensorflow/mps/libtensorflow_mps_plugin.dylib \
   ~/.tensorflow/plugins/

# Ou copier dans le dossier système
sudo mkdir -p /usr/local/lib/tensorflow-plugins
sudo cp bazel-bin/tensorflow/mps/libtensorflow_mps_plugin.dylib \
        /usr/local/lib/tensorflow-plugins/
```

### Installation de TensorFlow
```bash
# Désinstaller l'ancienne version
pip3 uninstall tensorflow

# Installer la nouvelle version avec support MPS
pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl

# Vérifier l'installation
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 💻 Utilisation

### Vérifier le Device MPS

```python
import tensorflow as tf

# Lister les devices disponibles
print("Devices disponibles:")
for device in tf.config.list_physical_devices():
    print(f"  - {device}")

# Vérifier spécifiquement MPS
mps_devices = tf.config.list_physical_devices('MPS')
if mps_devices:
    print(f"\n✅ MPS disponible: {len(mps_devices)} device(s)")
else:
    print("\n❌ MPS non disponible")

# Obtenir des infos sur le device
print("\nInformations GPU:")
print(tf.config.experimental.get_device_details(mps_devices[0]))
```

### Utilisation Basique

```python
import tensorflow as tf
import numpy as np

# Forcer l'utilisation du device MPS
with tf.device('/device:MPS:0'):
    # Créer des tenseurs
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    
    # Opérations mathématiques
    c = tf.matmul(a, b)
    print(f"MatMul result:\n{c.numpy()}")
    
    # Opérations de réduction
    sum_result = tf.reduce_sum(a)
    mean_result = tf.reduce_mean(a)
    print(f"\nSum: {sum_result.numpy()}")
    print(f"Mean: {mean_result.numpy()}")
    
    # Opérations logiques
    mask = tf.greater(a, 2.0)
    print(f"\nMask (a > 2.0):\n{mask.numpy()}")
```

### Nouvelles Opérations Implémentées

#### 1. Data Manipulation
```python
with tf.device('/device:MPS:0'):
    # ConcatV2
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6]])
    result = tf.concat([a, b], axis=0)
    
    # Stack
    x = tf.constant([1, 2, 3])
    y = tf.constant([4, 5, 6])
    stacked = tf.stack([x, y])
    
    # ReverseV2
    reversed = tf.reverse(a, axis=[1])
    
    # Squeeze/ExpandDims
    expanded = tf.expand_dims(x, axis=0)
    squeezed = tf.squeeze(expanded)
```

#### 2. Linear Algebra
```python
with tf.device('/device:MPS:0'):
    # Cholesky decomposition
    A = tf.constant([[4.0, 2.0], [2.0, 3.0]])
    L = tf.linalg.cholesky(A)
    
    # Matrix inversion
    A_inv = tf.linalg.inv(A)
    
    # QR decomposition
    q, r = tf.linalg.qr(A)
    
    # SVD
    s, u, v = tf.linalg.svd(A)
    
    # Eigenvalues
    eigenvalues, eigenvectors = tf.linalg.eig(A)
    
    # Determinant
    det = tf.linalg.det(A)
```

#### 3. Signal Processing
```python
with tf.device('/device:MPS:0'):
    # FFT
    signal = tf.constant([1.0, 0.0, -1.0, 0.0], dtype=tf.complex64)
    fft_result = tf.signal.fft(signal)
    
    # IFFT
    ifft_result = tf.signal.ifft(fft_result)
    
    # RFFT (Real FFT)
    real_signal = tf.constant([1.0, 0.0, -1.0, 0.0])
    rfft_result = tf.signal.rfft(real_signal)
    
    # STFT
    audio = tf.random.normal([16000])  # 1 second at 16kHz
    stft = tf.signal.stft(audio, frame_length=256, frame_step=128)
```

#### 4. Image Operations
```python
with tf.device('/device:MPS:0'):
    # Resize
    image = tf.random.normal([1, 100, 100, 3])
    resized = tf.image.resize(image, [224, 224], method='bilinear')
    
    # Non-Max Suppression
    boxes = tf.constant([
        [0.0, 0.0, 1.0, 1.0],
        [0.1, 0.1, 1.1, 1.1],
        [0.9, 0.9, 1.9, 1.9]
    ])
    scores = tf.constant([0.9, 0.8, 0.7])
    selected = tf.image.non_max_suppression(
        boxes, scores, 
        max_output_size=3, 
        iou_threshold=0.5
    )
```

#### 5. Quantization
```python
with tf.device('/device:MPS:0'):
    # Quantize
    float_tensor = tf.constant([0.0, 1.0, 2.0, 3.0])
    quantized, scale, zero_point = tf.quantization.quantize(
        float_tensor, 
        min_range=0.0, 
        max_range=3.0, 
        T=tf.quint8
    )
    
    # Dequantize
    dequantized = tf.quantization.dequantize(
        quantized, scale, zero_point
    )
    
    # Fake Quantization (for training)
    fake_quant = tf.quantization.fake_quant_with_min_max_args(
        float_tensor, 
        min=0.0, 
        max=3.0
    )
```

### Entraînement d'un Modèle

```python
import tensorflow as tf
from tensorflow import keras

# Configuration pour utiliser MPS
with tf.device('/device:MPS:0'):
    # Créer un modèle simple
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compiler
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Données d'exemple
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    
    # Entraîner
    history = model.fit(
        x_train, y_train,
        batch_size=256,
        epochs=5,
        validation_split=0.1
    )
    
    # Évaluer
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')
```

---

## 🧪 Tests

### Tests Unitaires

```bash
# Test des nouvelles opérations
bazel test //tensorflow/mps:additional_ops_test \
    --test_output=all \
    --test_tag_filters=requires_gpu_apple

# Tests existants
bazel test //tensorflow/mps:ops_test \
    --test_output=all

# Tests du device
bazel test //tensorflow/mps:mps_device_test \
    --test_output=all
```

### Script de Validation

```bash
# Exécuter le script de validation
./validate_mps_implementation.sh
```

### Benchmark de Performance

```python
import tensorflow as tf
import time
import numpy as np

def benchmark_op(op_fn, *args, device='/device:MPS:0', iterations=100):
    """Benchmark une opération sur un device spécifique."""
    with tf.device(device):
        # Warmup
        for _ in range(10):
            result = op_fn(*args)
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            result = op_fn(*args)
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000  # ms
        return avg_time

# Exemple: Benchmark MatMul
A = tf.random.normal([1024, 1024])
B = tf.random.normal([1024, 1024])

mps_time = benchmark_op(tf.matmul, A, B, device='/device:MPS:0')
cpu_time = benchmark_op(tf.matmul, A, B, device='/device:CPU:0')

print(f"MatMul (1024x1024):")
print(f"  MPS: {mps_time:.2f} ms")
print(f"  CPU: {cpu_time:.2f} ms")
print(f"  Speedup: {cpu_time/mps_time:.2f}x")
```

---

## 🔍 Dépannage

### MPS Device Non Détecté

```bash
# Vérifier que le plugin est chargé
python3 -c "
import tensorflow as tf
print('TF version:', tf.__version__)
print('MPS devices:', tf.config.list_physical_devices('MPS'))
"

# Vérifier les logs
export TF_CPP_MIN_LOG_LEVEL=0
python3 -c "import tensorflow as tf; tf.config.list_physical_devices()"
```

**Solutions**:
1. Vérifier que le plugin `libtensorflow_mps_plugin.dylib` est dans `~/.tensorflow/plugins/`
2. Recompiler avec `--config=macos`
3. Vérifier que Metal est supporté: `system_profiler SPDisplaysDataType | grep Metal`

### Erreurs de Compilation

```bash
# Nettoyer le cache Bazel
bazel clean --expunge

# Recompiler avec verbose
bazel build //tensorflow/mps:libtensorflow_mps_plugin.dylib \
    --config=macos \
    --verbose_failures \
    --sandbox_debug
```

### Erreurs d'Exécution

```python
# Activer les logs détaillés
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'

import tensorflow as tf
# Vos opérations...
```

### Performance Dégradée

**Vérifications**:
1. Taille des batches trop petite → Augmenter batch_size
2. Transferts CPU↔GPU fréquents → Garder les données sur GPU
3. Opérations mixed CPU/GPU → Forcer device avec `with tf.device()`

```python
# Optimiser les transferts
with tf.device('/device:MPS:0'):
    # Créer toutes les données sur GPU
    data = tf.random.normal([10000, 1024])
    
    # Toutes les opérations restent sur GPU
    result = tf.matmul(data, tf.transpose(data))
    result = tf.reduce_mean(result, axis=0)
```

---

## 📚 Ressources

### Documentation
- [TensorFlow MPS Implementation Report](MPS_IMPLEMENTATION_REPORT.md)
- [MPS Implementation Details](MPS_IMPLEMENTATION_FINALE.md)
- [PR Summary](MPS_PR_SUMMARY.md)

### Apple Documentation
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [Accelerate Framework](https://developer.apple.com/documentation/accelerate)

### TensorFlow
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [Custom Devices](https://www.tensorflow.org/guide/create_op)

---

## 📞 Support

### Issues
Pour reporter des bugs ou demander des fonctionnalités:
- GitHub Issues: [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues)
- Tag: `comp:mps` ou `comp:apple`

### Contribution
Pour contribuer:
1. Fork le repository
2. Créer une branche: `git checkout -b feature/my-feature`
3. Commit: `git commit -am 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Créer une Pull Request

---

## 📜 License

Apache License 2.0 - Voir [LICENSE](LICENSE) pour plus de détails.

---

**Dernière mise à jour**: 24 octobre 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready

🎉 **Bon développement avec TensorFlow MPS!**
