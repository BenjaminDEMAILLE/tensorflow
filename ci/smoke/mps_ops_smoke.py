import os
import numpy as np
import tensorflow as tf

# Basic device listing
print("TF version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())

# Small helpers
DTYPES = [tf.float32, tf.float16, tf.bfloat16]

# Activations
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 6.0], dtype=tf.float32)
print("LeakyRelu:", tf.nn.leaky_relu(x).numpy())
print("Relu6:", tf.nn.relu6(x).numpy())
print("Elu:", tf.nn.elu(x).numpy())
print("Selu:", tf.nn.selu(x).numpy())
print("Softplus:", tf.nn.softplus(x).numpy())
print("Softsign:", tf.nn.softsign(x).numpy())

# Slice / StridedSlice
m = tf.reshape(tf.range(24, dtype=tf.float32), (2,3,4))
sl = tf.slice(m, [0,1,1], [2,2,3])
print("Slice shape:", sl.shape, "sum:", tf.reduce_sum(sl).numpy())
ssl = m[:, 0:3:1, 1:4:2]
print("StridedSlice shape:", ssl.shape, "values:", ssl.numpy())

# Fill / ZerosLike / OnesLike
f = tf.fill([2,3], 2.5)
print("Fill sum:", tf.reduce_sum(f).numpy())
zl = tf.zeros_like(f)
ol = tf.ones_like(f)
print("ZerosLike sum:", tf.reduce_sum(zl).numpy(), "OnesLike sum:", tf.reduce_sum(ol).numpy())

# Pad (constant 0)
padded = tf.pad(tf.reshape(tf.range(6, dtype=tf.float32), (2,3)), [[1,1],[2,2]])
print("Pad shape:", padded.shape, "sum:", tf.reduce_sum(padded).numpy())

# MirrorPad (reflect)
mp = tf.raw_ops.MirrorPad(input=tf.reshape(tf.range(6, dtype=tf.float32), (2,3)), paddings=[[1,1],[2,2]], mode="REFLECT")
print("MirrorPad shape:", mp.shape, "sum:", tf.reduce_sum(mp).numpy())

# Tile
T = tf.reshape(tf.range(6, dtype=tf.float32), (2,3))
Ti = tf.tile(T, [2,2])
print("Tile shape:", Ti.shape, "sum:", tf.reduce_sum(Ti).numpy())

# Select (broadcast cond scalar)
cond = tf.constant(True)
a = tf.ones((2,3), dtype=tf.float32)
b = tf.zeros((2,3), dtype=tf.float32)
sel = tf.where(cond, a, b)
print("Select sum:", tf.reduce_sum(sel).numpy())

# ClipByValue
c = tf.clip_by_value(tf.constant([-1.0, 0.5, 2.0], dtype=tf.float32), clip_value_min=0.0, clip_value_max=1.0)
print("ClipByValue:", c.numpy())

# Logical ops
la = tf.constant([[True, False],[False, True]])
lb = tf.constant(True)
print("LogicalAnd:", tf.logical_and(la, lb).numpy())
print("LogicalOr:", tf.logical_or(la, lb).numpy())
print("LogicalNot:", tf.logical_not(la).numpy())

# OneHot
oh = tf.one_hot(tf.constant([0,2,1], dtype=tf.int32), depth=4, on_value=tf.constant(1.0), off_value=tf.constant(0.0), axis=-1)
print("OneHot:", oh.numpy())

# Range
r1 = tf.range(0., 1., 0.25)
r2 = tf.range(0, 5, 2)
print("Range float:", r1.numpy(), "Range int:", r2.numpy())

# Split (equal splits)
sp_in = tf.reshape(tf.range(8, dtype=tf.float32), (2,4))
sp = tf.split(sp_in, num_or_size_splits=2, axis=1)
print("Split lens:", [s.shape for s in sp], "sums:", [tf.reduce_sum(s).numpy() for s in sp])

# GatherV2 (tf.gather axis=0)
params = tf.reshape(tf.range(15, dtype=tf.float32), (5,3))
indices = tf.constant([2,0,4], dtype=tf.int32)
g = tf.gather(params, indices, axis=0)
print("GatherV2 shape:", g.shape, "sum:", tf.reduce_sum(g).numpy())

# GatherND (batch_dims=0)
params_nd = tf.reshape(tf.range(12, dtype=tf.float32), (4,3))
idx_nd = tf.constant([[0],[3],[1]], dtype=tf.int32)
gnd = tf.gather_nd(params_nd, idx_nd)
print("GatherND shape:", gnd.shape, "sum:", tf.reduce_sum(gnd).numpy())

# StridedSlice with negative strides
m2 = tf.reshape(tf.range(12, dtype=tf.float32), (3,4))
ssl_neg = m2[::-1, ::-1]  # reverse both dimensions
print("StridedSlice reverse shape:", ssl_neg.shape, "first elem:", ssl_neg[0,0].numpy())

# StridedSlice with new_axis
m3 = tf.reshape(tf.range(6, dtype=tf.float32), (2,3))
ssl_new = m3[tf.newaxis, :, :, tf.newaxis]  # add dims at front and back
print("StridedSlice newaxis shape:", ssl_new.shape)

# Comparison ops
comp_a = tf.constant([1.0, 2.0, 3.0])
comp_b = tf.constant([2.0, 2.0, 1.0])
print("Equal:", tf.equal(comp_a, comp_b).numpy())
print("NotEqual:", tf.not_equal(comp_a, comp_b).numpy())
print("Less:", tf.less(comp_a, comp_b).numpy())
print("LessEqual:", tf.less_equal(comp_a, comp_b).numpy())
print("Greater:", tf.greater(comp_a, comp_b).numpy())
print("GreaterEqual:", tf.greater_equal(comp_a, comp_b).numpy())

# Boolean reductions
bool_t = tf.constant([[True, True], [True, False]])
print("All:", tf.reduce_all(bool_t).numpy())
print("Any:", tf.reduce_any(bool_t).numpy())
bool_all_true = tf.constant([[True, True], [True, True]])
print("All (all true):", tf.reduce_all(bool_all_true).numpy())

print("\n✅ All smoke tests passed!")
