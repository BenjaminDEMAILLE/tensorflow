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
"""Tests for MPS (Metal Performance Shaders) device plugin."""

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test


class MPSDeviceTest(test.TestCase):
  """Test suite for MPS device functionality."""

  def test_device_enumeration(self):
    """Test that MPS device is enumerated correctly."""
    devices = tf.config.list_physical_devices()
    mps_devices = [d for d in devices if 'MPS' in d.device_type]
    # On macOS with Metal support, we should have at least one MPS device
    # This test may be skipped on non-macOS platforms
    if mps_devices:
      self.assertGreater(len(mps_devices), 0)
      print(f"Found {len(mps_devices)} MPS device(s): {mps_devices}")

  def test_identity_float32(self):
    """Test Identity operation with float32."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
      y = tf.identity(x)
      self.assertAllClose(x.numpy(), y.numpy())

  def test_identity_float16(self):
    """Test Identity operation with float16."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float16)
      y = tf.identity(x)
      self.assertAllClose(x.numpy(), y.numpy())

  def test_identity_bfloat16(self):
    """Test Identity operation with bfloat16."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.bfloat16)
      y = tf.identity(x)
      self.assertAllClose(x.numpy(), y.numpy())

  def test_relu_float32(self):
    """Test ReLU operation with float32."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[-1.0, 2.0], [-3.0, 4.0]], dtype=tf.float32)
      y = tf.nn.relu(x)
      expected = np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float32)
      self.assertAllClose(y.numpy(), expected)

  def test_relu_float16(self):
    """Test ReLU operation with float16."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[-1.0, 2.0], [-3.0, 4.0]], dtype=tf.float16)
      y = tf.nn.relu(x)
      expected = np.array([[0.0, 2.0], [0.0, 4.0]], dtype=np.float16)
      self.assertAllClose(y.numpy(), expected, atol=1e-3)

  def test_add_float32(self):
    """Test AddV2 operation with float32."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
      b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
      c = tf.add(a, b)
      expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
      self.assertAllClose(c.numpy(), expected)

  def test_add_float16(self):
    """Test AddV2 operation with float16."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float16)
      b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float16)
      c = tf.add(a, b)
      expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float16)
      self.assertAllClose(c.numpy(), expected, atol=1e-3)

  def test_add_scalar_broadcast(self):
    """Test AddV2 with scalar broadcasting."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
      b = tf.constant(10.0, dtype=tf.float32)
      c = tf.add(a, b)
      expected = np.array([[11.0, 12.0], [13.0, 14.0]], dtype=np.float32)
      self.assertAllClose(c.numpy(), expected)

  def test_mul_float32(self):
    """Test Mul operation with float32."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
      b = tf.constant([[2.0, 3.0], [4.0, 5.0]], dtype=tf.float32)
      c = tf.multiply(a, b)
      expected = np.array([[2.0, 6.0], [12.0, 20.0]], dtype=np.float32)
      self.assertAllClose(c.numpy(), expected)

  def test_mul_float16(self):
    """Test Mul operation with float16."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float16)
      b = tf.constant([[2.0, 3.0], [4.0, 5.0]], dtype=tf.float16)
      c = tf.multiply(a, b)
      expected = np.array([[2.0, 6.0], [12.0, 20.0]], dtype=np.float16)
      self.assertAllClose(c.numpy(), expected, atol=1e-3)

  def test_maximum_float32(self):
    """Test Maximum operation with float32."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 5.0], [3.0, 2.0]], dtype=tf.float32)
      b = tf.constant([[4.0, 2.0], [1.0, 6.0]], dtype=tf.float32)
      c = tf.maximum(a, b)
      expected = np.array([[4.0, 5.0], [3.0, 6.0]], dtype=np.float32)
      self.assertAllClose(c.numpy(), expected)

  def test_minimum_float32(self):
    """Test Minimum operation with float32."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 5.0], [3.0, 2.0]], dtype=tf.float32)
      b = tf.constant([[4.0, 2.0], [1.0, 6.0]], dtype=tf.float32)
      c = tf.minimum(a, b)
      expected = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
      self.assertAllClose(c.numpy(), expected)

  def test_sigmoid_float32(self):
    """Test Sigmoid operation with float32."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[0.0, 1.0], [-1.0, 2.0]], dtype=tf.float32)
      y = tf.nn.sigmoid(x)
      expected = 1.0 / (1.0 + np.exp(-x.numpy()))
      self.assertAllClose(y.numpy(), expected, rtol=1e-5)

  def test_sigmoid_float16(self):
    """Test Sigmoid operation with float16."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[0.0, 1.0], [-1.0, 2.0]], dtype=tf.float16)
      y = tf.nn.sigmoid(x)
      expected = 1.0 / (1.0 + np.exp(-x.numpy().astype(np.float32)))
      self.assertAllClose(y.numpy(), expected.astype(np.float16), atol=1e-3)

  def test_tanh_float32(self):
    """Test Tanh operation with float32."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[0.0, 1.0], [-1.0, 2.0]], dtype=tf.float32)
      y = tf.nn.tanh(x)
      expected = np.tanh(x.numpy())
      self.assertAllClose(y.numpy(), expected, rtol=1e-5)

  def test_tanh_float16(self):
    """Test Tanh operation with float16."""
    with tf.device('/device:MPS:0'):
      x = tf.constant([[0.0, 1.0], [-1.0, 2.0]], dtype=tf.float16)
      y = tf.nn.tanh(x)
      expected = np.tanh(x.numpy().astype(np.float32))
      self.assertAllClose(y.numpy(), expected.astype(np.float16), atol=1e-3)

  def test_matmul_float32(self):
    """Test MatMul operation with float32."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
      b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
      c = tf.matmul(a, b)
      expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
      self.assertAllClose(c.numpy(), expected)

  def test_matmul_float16(self):
    """Test MatMul operation with float16."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float16)
      b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float16)
      c = tf.matmul(a, b)
      expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float16)
      self.assertAllClose(c.numpy(), expected, atol=1e-2)

  def test_matmul_bfloat16(self):
    """Test MatMul operation with bfloat16."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.bfloat16)
      b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.bfloat16)
      c = tf.matmul(a, b)
      expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
      # bfloat16 has lower precision, use larger tolerance
      self.assertAllClose(c.numpy(), expected.astype(np.float32), atol=0.5)

  def test_matmul_transpose(self):
    """Test MatMul with transpose flags."""
    with tf.device('/device:MPS:0'):
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
      b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
      c = tf.matmul(a, b, transpose_a=True, transpose_b=False)
      # a^T @ b = [[1,3],[2,4]] @ [[5,6],[7,8]] = [[26,30],[38,44]]
      expected = np.array([[26.0, 30.0], [38.0, 44.0]], dtype=np.float32)
      self.assertAllClose(c.numpy(), expected)

  def test_conv2d_float32_valid(self):
    """Test Conv2D operation with float32 and VALID padding."""
    with tf.device('/device:MPS:0'):
      # Input: [1, 4, 4, 1]
      x = tf.constant([[[[1.], [2.], [3.], [4.]],
                        [[5.], [6.], [7.], [8.]],
                        [[9.], [10.], [11.], [12.]],
                        [[13.], [14.], [15.], [16.]]]], dtype=tf.float32)
      # Filter: [2, 2, 1, 1]
      f = tf.constant([[[[1.]], [[0.]]],
                       [[[0.]], [[1.]]]], dtype=tf.float32)
      y = tf.nn.conv2d(x, f, strides=[1, 1, 1, 1], padding='VALID')
      # Expected output shape: [1, 3, 3, 1]
      self.assertEqual(y.shape, (1, 3, 3, 1))

  def test_conv2d_float32_same(self):
    """Test Conv2D operation with float32 and SAME padding."""
    with tf.device('/device:MPS:0'):
      # Input: [1, 4, 4, 2]
      x = tf.random.uniform([1, 4, 4, 2], dtype=tf.float32)
      # Filter: [3, 3, 2, 4]
      f = tf.random.uniform([3, 3, 2, 4], dtype=tf.float32)
      y = tf.nn.conv2d(x, f, strides=[1, 1, 1, 1], padding='SAME')
      # Expected output shape: [1, 4, 4, 4]
      self.assertEqual(y.shape, (1, 4, 4, 4))

  def test_conv2d_float32_stride(self):
    """Test Conv2D operation with stride."""
    with tf.device('/device:MPS:0'):
      x = tf.random.uniform([1, 8, 8, 3], dtype=tf.float32)
      f = tf.random.uniform([3, 3, 3, 16], dtype=tf.float32)
      y = tf.nn.conv2d(x, f, strides=[1, 2, 2, 1], padding='SAME')
      # Expected output shape: [1, 4, 4, 16]
      self.assertEqual(y.shape, (1, 4, 4, 16))

  def test_graph_mode_execution(self):
    """Test execution in graph mode."""
    @tf.function
    def compute(a, b):
      with tf.device('/device:MPS:0'):
        return tf.matmul(a, b) + tf.constant(1.0, dtype=tf.float32)
    
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
    c = compute(a, b)
    expected = np.array([[20.0, 23.0], [44.0, 51.0]], dtype=np.float32)
    self.assertAllClose(c.numpy(), expected)

  def test_mixed_precision(self):
    """Test mixed precision computation."""
    with tf.device('/device:MPS:0'):
      # float32 computation
      a_f32 = tf.constant([[1.0, 2.0]], dtype=tf.float32)
      b_f32 = tf.constant([[3.0], [4.0]], dtype=tf.float32)
      c_f32 = tf.matmul(a_f32, b_f32)
      
      # float16 computation
      a_f16 = tf.constant([[1.0, 2.0]], dtype=tf.float16)
      b_f16 = tf.constant([[3.0], [4.0]], dtype=tf.float16)
      c_f16 = tf.matmul(a_f16, b_f16)
      
      # Results should match (within tolerance)
      self.assertAllClose(c_f32.numpy(), c_f16.numpy().astype(np.float32), atol=1e-2)


if __name__ == '__main__':
  test.main()
