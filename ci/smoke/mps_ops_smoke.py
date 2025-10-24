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
