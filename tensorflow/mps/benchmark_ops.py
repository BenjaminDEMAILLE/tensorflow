"""Performance benchmarks for MPS operations."""

import time
import numpy as np
import tensorflow as tf


def benchmark_op(op_fn, name, warmup=10, iterations=100):
    """Benchmark a TensorFlow operation."""
    # Warmup
    for _ in range(warmup):
        _ = op_fn()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = op_fn()
        # Force execution
        if hasattr(result, 'numpy'):
            _ = result.numpy()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    print(f"  {name:40s}: {avg_time_ms:8.4f} ms")
    return avg_time_ms


def benchmark_comparison_ops():
    """Benchmark comparison operations."""
    print("\n📊 Comparison Operations")
    print("=" * 60)
    
    sizes = [1000, 10000, 100000]
    for size in sizes:
        print(f"\n  Array size: {size}")
        a = tf.constant(np.random.randn(size).astype(np.float32))
        b = tf.constant(np.random.randn(size).astype(np.float32))
        
        benchmark_op(lambda: tf.equal(a, b), "Equal")
        benchmark_op(lambda: tf.not_equal(a, b), "NotEqual")
        benchmark_op(lambda: tf.less(a, b), "Less")
        benchmark_op(lambda: tf.greater(a, b), "Greater")


def benchmark_strided_slice():
    """Benchmark StridedSlice operations."""
    print("\n📊 StridedSlice Operations")
    print("=" * 60)
    
    shapes = [(100, 100), (1000, 100), (100, 1000)]
    for shape in shapes:
        print(f"\n  Shape: {shape}")
        x = tf.constant(np.random.randn(*shape).astype(np.float32))
        
        benchmark_op(lambda: x[:50, :50], "Basic slice")
        benchmark_op(lambda: x[::-1, ::-1], "Reverse (negative stride)")
        benchmark_op(lambda: x[tf.newaxis, :, :], "New axis")
        benchmark_op(lambda: x[::2, ::2], "Stride 2")


def benchmark_gather_scatter():
    """Benchmark Gather and Scatter operations."""
    print("\n📊 Gather/Scatter Operations")
    print("=" * 60)
    
    print("\n  GatherV2 (axis=0)")
    params = tf.constant(np.random.randn(10000, 128).astype(np.float32))
    indices = tf.constant(np.random.randint(0, 10000, 1000).astype(np.int32))
    benchmark_op(lambda: tf.gather(params, indices, axis=0), "Gather 1000 indices")
    
    print("\n  GatherND")
    params_nd = tf.constant(np.random.randn(100, 100).astype(np.float32))
    indices_nd = tf.constant(np.random.randint(0, 100, (1000, 1)).astype(np.int32))
    benchmark_op(lambda: tf.gather_nd(params_nd, indices_nd), "GatherND 1000 indices")
    
    print("\n  TensorScatterUpdate")
    tensor = tf.constant(np.random.randn(1000, 128).astype(np.float32))
    scatter_indices = tf.constant(np.array([[i] for i in range(100)]).astype(np.int32))
    updates = tf.constant(np.random.randn(100, 128).astype(np.float32))
    benchmark_op(lambda: tf.tensor_scatter_nd_update(tensor, scatter_indices, updates), 
                 "Scatter 100 updates")


def benchmark_split():
    """Benchmark Split operation."""
    print("\n📊 Split Operations")
    print("=" * 60)
    
    shapes = [(1000, 128), (128, 1000), (500, 500)]
    for shape in shapes:
        print(f"\n  Shape: {shape}")
        x = tf.constant(np.random.randn(*shape).astype(np.float32))
        
        benchmark_op(lambda: tf.split(x, num_or_size_splits=2, axis=0), 
                     "Split axis=0, 2 parts")
        benchmark_op(lambda: tf.split(x, num_or_size_splits=4, axis=1), 
                     "Split axis=1, 4 parts")


def benchmark_logical_ops():
    """Benchmark logical operations."""
    print("\n📊 Logical Operations")
    print("=" * 60)
    
    sizes = [10000, 100000, 1000000]
    for size in sizes:
        print(f"\n  Array size: {size}")
        a = tf.constant(np.random.rand(size) > 0.5)
        b = tf.constant(np.random.rand(size) > 0.5)
        
        benchmark_op(lambda: tf.logical_and(a, b), "LogicalAnd")
        benchmark_op(lambda: tf.logical_or(a, b), "LogicalOr")
        benchmark_op(lambda: tf.logical_not(a), "LogicalNot")


def benchmark_reductions():
    """Benchmark boolean reduction operations."""
    print("\n📊 Boolean Reductions")
    print("=" * 60)
    
    sizes = [10000, 100000, 1000000]
    for size in sizes:
        print(f"\n  Array size: {size}")
        x_all_true = tf.constant(np.ones(size, dtype=bool))
        x_mixed = tf.constant(np.random.rand(size) > 0.5)
        
        benchmark_op(lambda: tf.reduce_all(x_all_true), "ReduceAll (all true)")
        benchmark_op(lambda: tf.reduce_all(x_mixed), "ReduceAll (mixed)")
        benchmark_op(lambda: tf.reduce_any(x_mixed), "ReduceAny (mixed)")


def benchmark_matrix_ops():
    """Benchmark matrix operations for comparison."""
    print("\n📊 Matrix Operations (Baseline)")
    print("=" * 60)
    
    sizes = [128, 512, 1024]
    for size in sizes:
        print(f"\n  Matrix size: {size}x{size}")
        a = tf.constant(np.random.randn(size, size).astype(np.float32))
        b = tf.constant(np.random.randn(size, size).astype(np.float32))
        
        benchmark_op(lambda: tf.add(a, b), "Add")
        benchmark_op(lambda: tf.multiply(a, b), "Multiply")
        benchmark_op(lambda: tf.matmul(a, b), "MatMul")


def main():
    """Run all benchmarks."""
    print("🚀 MPS Operations Performance Benchmark")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Devices: {tf.config.list_physical_devices()}")
    print("=" * 60)
    
    # Run benchmarks
    benchmark_comparison_ops()
    benchmark_strided_slice()
    benchmark_gather_scatter()
    benchmark_split()
    benchmark_logical_ops()
    benchmark_reductions()
    benchmark_matrix_ops()
    
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
