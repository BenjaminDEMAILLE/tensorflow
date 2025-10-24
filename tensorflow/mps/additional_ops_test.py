"""Additional unit tests for MPS-backed ops: data, linalg, signal, image, quantization.

These tests exercise correctness on small inputs. They don't force device placement;
when running on a Mac with MPS plugin loaded, they should execute on the MPS device.
"""

import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class MPSDataOpsTest(test_util.TensorFlowTestCase):
  def testConcatV2(self):
    with self.cached_session():
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
      b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
      out = tf.concat([a, b], axis=1)
      self.assertAllClose(out, [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]])

  def testPackUnpack(self):
    with self.cached_session():
      elems = [tf.constant([1.0, 2.0]), tf.constant([3.0, 4.0])]
      stacked = tf.stack(elems, axis=0)
      self.assertEqual(stacked.shape, (2, 2))
      # Unstack back
      u0, u1 = tf.unstack(stacked, axis=0)
      self.assertAllClose(u0, [1.0, 2.0])
      self.assertAllClose(u1, [3.0, 4.0])

  def testReverseV2(self):
    with self.cached_session():
      x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
      r = tf.reverse(x, axis=[1])
      self.assertAllEqual(r, [[3, 2, 1], [6, 5, 4]])

  def testSqueezeExpandDims(self):
    with self.cached_session():
      x = tf.zeros((1, 3, 1, 2))
      s = tf.squeeze(x, axis=[0, 2])
      self.assertEqual(s.shape, (3, 2))
      e = tf.expand_dims(s, axis=0)
      self.assertEqual(e.shape, (1, 3, 2))


class MPSLinalgOpsTest(test_util.TensorFlowTestCase):
  def testCholeskyAndInverse(self):
    with self.cached_session():
      a = tf.constant([[4.0, 1.0], [1.0, 3.0]])  # SPD
      L = tf.linalg.cholesky(a)
      self.assertAllClose(L @ tf.transpose(L), a, atol=1e-4)
      inv = tf.linalg.inv(a)
      self.assertAllClose(a @ inv, np.eye(2), atol=1e-4)

  def testQr(self):
    with self.cached_session():
      a = tf.constant([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]])  # 3x2
      q, r = tf.linalg.qr(a, full_matrices=False)
      self.assertAllClose(a, tf.matmul(q, r), atol=1e-4)

  def testSvd(self):
    with self.cached_session():
      a = tf.constant([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0]])  # 2x3
      s, u, v = tf.linalg.svd(a, compute_uv=True, full_matrices=False)
      recon = tf.matmul(u, tf.matmul(tf.linalg.diag(s), tf.transpose(v)))
      self.assertAllClose(a, recon, atol=1e-3)

  def testEigAndDet(self):
    with self.cached_session():
      a = tf.constant([[2.0, 0.0], [0.0, 3.0]])
      w, v = tf.linalg.eig(a)
      self.assertAllClose(tf.math.real(w), [2.0, 3.0], atol=1e-6)
      det = tf.linalg.det(a)
      self.assertAllClose(det, 6.0)


class MPSSignalOpsTest(test_util.TensorFlowTestCase):
  def testFFT_IFFT(self):
    with self.cached_session():
      x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.complex64)
      X = tf.signal.fft(x)
      x_rec = tf.signal.ifft(X)
      self.assertAllClose(x, x_rec, atol=1e-4)

  def testRFFT_Length(self):
    with self.cached_session():
      x = tf.constant([1.0, 0.0, -1.0, 0.0], dtype=tf.float32)
      X = tf.signal.rfft(x)
      # For length 4, rfft length is 3
      self.assertEqual(int(X.shape[-1]), 3)

  def testSTFT_Shape(self):
    with self.cached_session():
      sr = 16000
      t = tf.linspace(0.0, 1.0, sr)
      f = 440.0
      x = tf.sin(2.0 * math.pi * f * t)
      stft = tf.signal.stft(x, frame_length=256, frame_step=128)
      self.assertEqual(int(stft.shape[-1]), 129)  # fft_unique_bins


class MPSImageOpsTest(test_util.TensorFlowTestCase):
  def testResizeBilinear(self):
    with self.cached_session():
      x = tf.ones((1, 4, 4, 3), dtype=tf.float32)
      y = tf.image.resize(x, size=(8, 8), method=tf.image.ResizeMethod.BILINEAR)
      self.assertAllClose(tf.reduce_mean(y), 1.0)

  def testNonMaxSuppressionSmall(self):
    with self.cached_session():
      boxes = tf.constant([[0.0, 0.0, 1.0, 1.0],
                           [0.1, 0.1, 1.1, 1.1],
                           [0.9, 0.9, 1.9, 1.9]], dtype=tf.float32)
      scores = tf.constant([0.9, 0.8, 0.7], dtype=tf.float32)
      idx = tf.image.non_max_suppression(boxes, scores, max_output_size=2, iou_threshold=0.5)
      self.assertLessEqual(tf.size(idx), 2)


class MPSQuantizationOpsTest(test_util.TensorFlowTestCase):
  def testQuantizeDequantizeRoundtrip(self):
    with self.cached_session():
      x = tf.constant([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tf.float32)
      q = tf.raw_ops.QuantizeV2(input=x, min_range=-1.0, max_range=1.0, T=tf.qint8, mode="SCALED", round_mode="HALF_TO_EVEN")
      deq = tf.raw_ops.Dequantize(input=q.output, min_range=q.min, max_range=q.max, mode="SCALED")
      self.assertAllClose(x, deq, atol=2e-2)

  def testFakeQuantWithMinMaxArgs(self):
    with self.cached_session():
      x = tf.linspace(-2.0, 2.0, 9)
      y = tf.raw_ops.FakeQuantWithMinMaxArgs(inputs=x, min=-1.0, max=1.0, num_bits=8, narrow_range=False)
      # Values should be clipped to [-1,1]
      self.assertGreaterEqual(tf.reduce_min(y), -1.0)
      self.assertLessEqual(tf.reduce_max(y), 1.0)


if __name__ == '__main__':
  test.main()
