"""Unit tests for MPS operations."""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class MPSComparisonOpsTest(test_util.TensorFlowTestCase):
  """Test comparison operations on MPS device."""

  def testEqual(self):
    with self.cached_session():
      a = tf.constant([1.0, 2.0, 3.0])
      b = tf.constant([2.0, 2.0, 1.0])
      result = tf.equal(a, b)
      expected = [False, True, False]
      self.assertAllEqual(result, expected)

  def testNotEqual(self):
    with self.cached_session():
      a = tf.constant([1.0, 2.0, 3.0])
      b = tf.constant([2.0, 2.0, 1.0])
      result = tf.not_equal(a, b)
      expected = [True, False, True]
      self.assertAllEqual(result, expected)

  def testLess(self):
    with self.cached_session():
      a = tf.constant([1.0, 2.0, 3.0])
      b = tf.constant([2.0, 2.0, 1.0])
      result = tf.less(a, b)
      expected = [True, False, False]
      self.assertAllEqual(result, expected)

  def testGreater(self):
    with self.cached_session():
      a = tf.constant([1.0, 2.0, 3.0])
      b = tf.constant([2.0, 2.0, 1.0])
      result = tf.greater(a, b)
      expected = [False, False, True]
      self.assertAllEqual(result, expected)

  def testComparisonBroadcasting(self):
    with self.cached_session():
      a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
      b = tf.constant([2.0, 2.0])
      result = tf.equal(a, b)
      expected = [[False, True], [False, False]]
      self.assertAllEqual(result, expected)


class MPSStridedSliceTest(test_util.TensorFlowTestCase):
  """Test StridedSlice with advanced masking."""

  def testBasicSlice(self):
    with self.cached_session():
      x = tf.reshape(tf.range(24, dtype=tf.float32), (2, 3, 4))
      result = x[:, 1:, 2:]
      self.assertEqual(result.shape, (2, 2, 2))

  def testNegativeStride(self):
    with self.cached_session():
      x = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
      result = x[::-1, ::-1]
      self.assertEqual(result.shape, (3, 4))
      # Last element should become first
      self.assertAlmostEqual(float(result[0, 0]), 11.0)

  def testNewAxis(self):
    with self.cached_session():
      x = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
      result = x[tf.newaxis, :, :, tf.newaxis]
      self.assertEqual(result.shape, (1, 2, 3, 1))

  def testEllipsis(self):
    with self.cached_session():
      x = tf.reshape(tf.range(24, dtype=tf.float32), (2, 3, 4))
      result = x[..., 1:3]
      self.assertEqual(result.shape, (2, 3, 2))

  def testShrinkAxis(self):
    with self.cached_session():
      x = tf.reshape(tf.range(24, dtype=tf.float32), (2, 3, 4))
      result = x[0, :, :]
      self.assertEqual(result.shape, (3, 4))


class MPSGatherScatterTest(test_util.TensorFlowTestCase):
  """Test Gather and Scatter operations."""

  def testGatherV2(self):
    with self.cached_session():
      params = tf.reshape(tf.range(15, dtype=tf.float32), (5, 3))
      indices = tf.constant([2, 0, 4], dtype=tf.int32)
      result = tf.gather(params, indices, axis=0)
      self.assertEqual(result.shape, (3, 3))
      # Check first row is params[2]
      self.assertAllClose(result[0], [6.0, 7.0, 8.0])

  def testGatherND(self):
    with self.cached_session():
      params = tf.reshape(tf.range(12, dtype=tf.float32), (4, 3))
      indices = tf.constant([[0], [3], [1]], dtype=tf.int32)
      result = tf.gather_nd(params, indices)
      self.assertEqual(result.shape, (3, 3))

  def testTensorScatterUpdate(self):
    with self.cached_session():
      tensor = tf.zeros((5, 3), dtype=tf.float32)
      indices = tf.constant([[0], [2]], dtype=tf.int32)
      updates = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
      result = tf.tensor_scatter_nd_update(tensor, indices, updates)
      self.assertAllClose(result[0], [1.0, 2.0, 3.0])
      self.assertAllClose(result[2], [4.0, 5.0, 6.0])

  def testTensorScatterAdd(self):
    with self.cached_session():
      tensor = tf.ones((3, 3), dtype=tf.float32)
      indices = tf.constant([[0], [2]], dtype=tf.int32)
      updates = tf.constant([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=tf.float32)
      result = tf.tensor_scatter_nd_add(tensor, indices, updates)
      # Should be 1.0 + 10.0 = 11.0 etc
      self.assertAllClose(result[0], [11.0, 21.0, 31.0])


class MPSSplitTest(test_util.TensorFlowTestCase):
  """Test Split operation."""

  def testSplitEqualParts(self):
    with self.cached_session():
      x = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
      splits = tf.split(x, num_or_size_splits=2, axis=1)
      self.assertEqual(len(splits), 2)
      self.assertEqual(splits[0].shape, (3, 2))
      self.assertEqual(splits[1].shape, (3, 2))

  def testSplitAxis0(self):
    with self.cached_session():
      x = tf.reshape(tf.range(12, dtype=tf.float32), (4, 3))
      splits = tf.split(x, num_or_size_splits=2, axis=0)
      self.assertEqual(len(splits), 2)
      self.assertEqual(splits[0].shape, (2, 3))


class MPSBooleanOpsTest(test_util.TensorFlowTestCase):
  """Test boolean reduction operations."""

  def testReduceAll(self):
    with self.cached_session():
      x = tf.constant([[True, True], [True, True]])
      result = tf.reduce_all(x)
      self.assertTrue(result)

  def testReduceAllWithFalse(self):
    with self.cached_session():
      x = tf.constant([[True, True], [False, True]])
      result = tf.reduce_all(x)
      self.assertFalse(result)

  def testReduceAny(self):
    with self.cached_session():
      x = tf.constant([[False, False], [False, True]])
      result = tf.reduce_any(x)
      self.assertTrue(result)

  def testReduceAnyAllFalse(self):
    with self.cached_session():
      x = tf.constant([[False, False], [False, False]])
      result = tf.reduce_any(x)
      self.assertFalse(result)


class MPSLogicalOpsTest(test_util.TensorFlowTestCase):
  """Test logical operations with broadcasting."""

  def testLogicalAnd(self):
    with self.cached_session():
      a = tf.constant([[True, False], [True, True]])
      b = tf.constant(True)
      result = tf.logical_and(a, b)
      expected = [[True, False], [True, True]]
      self.assertAllEqual(result, expected)

  def testLogicalOr(self):
    with self.cached_session():
      a = tf.constant([[True, False], [False, False]])
      b = tf.constant(False)
      result = tf.logical_or(a, b)
      expected = [[True, False], [False, False]]
      self.assertAllEqual(result, expected)

  def testLogicalNot(self):
    with self.cached_session():
      a = tf.constant([True, False, True])
      result = tf.logical_not(a)
      expected = [False, True, False]
      self.assertAllEqual(result, expected)


if __name__ == '__main__':
  test.main()
