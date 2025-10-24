// Sparse tensor operations with Metal
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace {
id<MTLDevice> GetMetalDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

id<MTLCommandQueue> GetCommandQueue() {
  static id<MTLCommandQueue> queue = [GetMetalDevice() newCommandQueue];
  return queue;
}
}

// SparseToDense - Convert sparse representation to dense tensor
extern "C" {

const char kSparseToDenseKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void SparseToDenseCompute(
    device const int* indices [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_indices [[buffer(3)]],
    constant int& output_size [[buffer(4)]],
    constant float& default_value [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_size) return;
    
    output[gid] = default_value;
    
    for (int i = 0; i < num_indices; i++) {
        if (indices[i] == gid) {
            output[gid] = values[i];
            break;
        }
    }
}
)";

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSSparseToDenseContext;

void* MPSSparseToDense_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSSparseToDenseContext();
  context->queue = GetCommandQueue();
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSparseToDenseKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"SparseToDenseCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSSparseToDense_Delete(void* kernel) {
  delete static_cast<MPSSparseToDenseContext*>(kernel);
}

void MPSSparseToDense_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSSparseToDenseContext*>(kernel);
    
    TF_Tensor* indices_tensor = nullptr;
    TF_Tensor* output_shape_tensor = nullptr;
    TF_Tensor* values_tensor = nullptr;
    TF_Tensor* default_value_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &indices_tensor, status);
    TF_GetInput(ctx, 1, &output_shape_tensor, status);
    TF_GetInput(ctx, 2, &values_tensor, status);
    TF_GetInput(ctx, 3, &default_value_tensor, status);
    
    int64_t num_indices = TF_Dim(values_tensor, 0);
    int32_t* shape_data = static_cast<int32_t*>(TF_TensorData(output_shape_tensor));
    int num_dims = TF_NumDims(output_shape_tensor);
    
    int64_t output_size = 1;
    for (int i = 0; i < num_dims; i++) {
      output_size *= shape_data[i];
    }
    
    int* indices_data = static_cast<int*>(TF_TensorData(indices_tensor));
    float* values_data = static_cast<float*>(TF_TensorData(values_tensor));
    float default_value = *static_cast<float*>(TF_TensorData(default_value_tensor));
    
    size_t indices_bytes = num_indices * sizeof(int);
    size_t values_bytes = num_indices * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> indices_buf = [device newBufferWithBytes:indices_data length:indices_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> values_buf = [device newBufferWithBytes:values_data length:values_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:indices_buf offset:0 atIndex:0];
    [encoder setBuffer:values_buf offset:0 atIndex:1];
    [encoder setBuffer:output_buf offset:0 atIndex:2];
    
    int num_idx = (int)num_indices;
    int out_size = (int)output_size;
    [encoder setBytes:&num_idx length:sizeof(int) atIndex:3];
    [encoder setBytes:&out_size length:sizeof(int) atIndex:4];
    [encoder setBytes:&default_value length:sizeof(float) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(output_size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, (int64_t*)shape_data, num_dims, output_bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [output_buf contents], output_bytes);
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// SparseMatMul - Sparse matrix multiplication
extern "C" {

const char kSparseMatMulKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void SparseMatMulCompute(
    device const int* a_indices [[buffer(0)]],
    device const float* a_values [[buffer(1)]],
    device const float* b [[buffer(2)]],
    device float* c [[buffer(3)]],
    constant int& a_nnz [[buffer(4)]],
    constant int& m [[buffer(5)]],
    constant int& k [[buffer(6)]],
    constant int& n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    int row = gid / n;
    int col = gid % n;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0f;
    
    // Iterate through sparse elements of row in A
    for (int i = 0; i < a_nnz; i++) {
        int sparse_row = a_indices[i * 2];
        int sparse_col = a_indices[i * 2 + 1];
        
        if (sparse_row == row) {
            sum += a_values[i] * b[sparse_col * n + col];
        }
    }
    
    c[row * n + col] = sum;
}
)";

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSSparseMatMulContext;

void* MPSSparseMatMul_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSSparseMatMulContext();
  context->queue = GetCommandQueue();
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSparseMatMulKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"SparseMatMulCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSSparseMatMul_Delete(void* kernel) {
  delete static_cast<MPSSparseMatMulContext*>(kernel);
}

void MPSSparseMatMul_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSSparseMatMulContext*>(kernel);
    
    TF_Tensor* a_indices_tensor = nullptr;
    TF_Tensor* a_values_tensor = nullptr;
    TF_Tensor* a_shape_tensor = nullptr;
    TF_Tensor* b_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &a_indices_tensor, status);
    TF_GetInput(ctx, 1, &a_values_tensor, status);
    TF_GetInput(ctx, 2, &a_shape_tensor, status);
    TF_GetInput(ctx, 3, &b_tensor, status);
    
    // A is sparse: shape [m, k], nnz non-zero elements
    // B is dense: shape [k, n]
    // C = A * B: shape [m, n]
    
    int64_t a_nnz = TF_Dim(a_values_tensor, 0);
    int64_t* a_shape_data = static_cast<int64_t*>(TF_TensorData(a_shape_tensor));
    int64_t m = a_shape_data[0];
    int64_t k = a_shape_data[1];
    int64_t n = TF_Dim(b_tensor, 1);
    
    int* a_indices_data = static_cast<int*>(TF_TensorData(a_indices_tensor));
    float* a_values_data = static_cast<float*>(TF_TensorData(a_values_tensor));
    float* b_data = static_cast<float*>(TF_TensorData(b_tensor));
    
    size_t indices_bytes = a_nnz * 2 * sizeof(int);
    size_t values_bytes = a_nnz * sizeof(float);
    size_t b_bytes = k * n * sizeof(float);
    size_t c_bytes = m * n * sizeof(float);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> indices_buf = [device newBufferWithBytes:a_indices_data length:indices_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> values_buf = [device newBufferWithBytes:a_values_data length:values_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_buf = [device newBufferWithBytes:b_data length:b_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_buf = [device newBufferWithLength:c_bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:indices_buf offset:0 atIndex:0];
    [encoder setBuffer:values_buf offset:0 atIndex:1];
    [encoder setBuffer:b_buf offset:0 atIndex:2];
    [encoder setBuffer:c_buf offset:0 atIndex:3];
    
    int nnz = (int)a_nnz;
    int m_int = (int)m;
    int k_int = (int)k;
    int n_int = (int)n;
    [encoder setBytes:&nnz length:sizeof(int) atIndex:4];
    [encoder setBytes:&m_int length:sizeof(int) atIndex:5];
    [encoder setBytes:&k_int length:sizeof(int) atIndex:6];
    [encoder setBytes:&n_int length:sizeof(int) atIndex:7];
    
    MTLSize gridSize = MTLSizeMake(m * n, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    int64_t output_dims[] = {m, n};
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 2, c_bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [c_buf contents], c_bytes);
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// SparseSoftmax - Softmax over sparse tensor
extern "C" {

typedef struct {
  id<MTLCommandQueue> queue;
} MPSSparseSoftmaxContext;

void* MPSSparseSoftmax_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSSparseSoftmaxContext();
  context->queue = GetCommandQueue();
  return context;
}

void MPSSparseSoftmax_Delete(void* kernel) {
  delete static_cast<MPSSparseSoftmaxContext*>(kernel);
}

void MPSSparseSoftmax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* indices_tensor = nullptr;
    TF_Tensor* values_tensor = nullptr;
    TF_Tensor* shape_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &indices_tensor, status);
    TF_GetInput(ctx, 1, &values_tensor, status);
    TF_GetInput(ctx, 2, &shape_tensor, status);
    
    int64_t nnz = TF_Dim(values_tensor, 0);
    float* values_data = static_cast<float*>(TF_TensorData(values_tensor));
    
    // Compute max
    float max_val = values_data[0];
    for (int i = 1; i < nnz; i++) {
      if (values_data[i] > max_val) max_val = values_data[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    float* exp_values = new float[nnz];
    for (int i = 0; i < nnz; i++) {
      exp_values[i] = expf(values_data[i] - max_val);
      sum += exp_values[i];
    }
    
    // Normalize
    for (int i = 0; i < nnz; i++) {
      exp_values[i] /= sum;
    }
    
    // Output
    size_t output_bytes = nnz * sizeof(float);
    int64_t output_dims[] = {nnz};
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 1, output_bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, exp_values, output_bytes);
    
    delete[] exp_values;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// SparseAdd - Add two sparse tensors
extern "C" {

typedef struct {
  id<MTLCommandQueue> queue;
} MPSSparseAddContext;

void* MPSSparseAdd_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSSparseAddContext();
  context->queue = GetCommandQueue();
  return context;
}

void MPSSparseAdd_Delete(void* kernel) {
  delete static_cast<MPSSparseAddContext*>(kernel);
}

void MPSSparseAdd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* a_indices_tensor = nullptr;
    TF_Tensor* a_values_tensor = nullptr;
    TF_Tensor* b_indices_tensor = nullptr;
    TF_Tensor* b_values_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &a_indices_tensor, status);
    TF_GetInput(ctx, 1, &a_values_tensor, status);
    TF_GetInput(ctx, 2, &b_indices_tensor, status);
    TF_GetInput(ctx, 3, &b_values_tensor, status);
    
    int64_t a_nnz = TF_Dim(a_values_tensor, 0);
    int64_t b_nnz = TF_Dim(b_values_tensor, 0);
    
    // Merge sparse tensors - simplified implementation
    // Real implementation would need proper index merging
    
    int64_t max_nnz = a_nnz + b_nnz;
    size_t output_bytes = max_nnz * sizeof(float);
    int64_t output_dims[] = {max_nnz};
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 1, output_bytes, status);
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// SparseReorder - Reorder sparse tensor indices
extern "C" {

typedef struct {
  id<MTLCommandQueue> queue;
} MPSSparseReorderContext;

void* MPSSparseReorder_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSSparseReorderContext();
  context->queue = GetCommandQueue();
  return context;
}

void MPSSparseReorder_Delete(void* kernel) {
  delete static_cast<MPSSparseReorderContext*>(kernel);
}

void MPSSparseReorder_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* indices_tensor = nullptr;
    TF_Tensor* values_tensor = nullptr;
    TF_Tensor* shape_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &indices_tensor, status);
    TF_GetInput(ctx, 1, &values_tensor, status);
    TF_GetInput(ctx, 2, &shape_tensor, status);
    
    int64_t nnz = TF_Dim(values_tensor, 0);
    int64_t num_dims = TF_Dim(indices_tensor, 1);
    
    // Sort indices in row-major order
    // Simplified - would need proper sorting implementation
    
    size_t indices_bytes = nnz * num_dims * sizeof(int64_t);
    size_t values_bytes = nnz * sizeof(float);
    
    int64_t indices_dims[] = {nnz, num_dims};
    TF_Tensor* output_indices = TF_AllocateOutput(ctx, 0, TF_INT64, indices_dims, 2, indices_bytes, status);
    
    int64_t values_dims[] = {nnz};
    TF_Tensor* output_values = TF_AllocateOutput(ctx, 1, TF_FLOAT, values_dims, 1, values_bytes, status);
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// SparseSlice - Slice a sparse tensor
extern "C" {

typedef struct {
  id<MTLCommandQueue> queue;
} MPSSparseSliceContext;

void* MPSSparseSlice_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSSparseSliceContext();
  context->queue = GetCommandQueue();
  return context;
}

void MPSSparseSlice_Delete(void* kernel) {
  delete static_cast<MPSSparseSliceContext*>(kernel);
}

void MPSSparseSlice_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* indices_tensor = nullptr;
    TF_Tensor* values_tensor = nullptr;
    TF_Tensor* shape_tensor = nullptr;
    TF_Tensor* start_tensor = nullptr;
    TF_Tensor* size_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &indices_tensor, status);
    TF_GetInput(ctx, 1, &values_tensor, status);
    TF_GetInput(ctx, 2, &shape_tensor, status);
    TF_GetInput(ctx, 3, &start_tensor, status);
    TF_GetInput(ctx, 4, &size_tensor, status);
    
    int64_t nnz = TF_Dim(values_tensor, 0);
    
    // Filter indices within slice bounds
    // Simplified implementation
    
    size_t output_bytes = nnz * sizeof(float);
    int64_t output_dims[] = {nnz};
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 1, output_bytes, status);
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"
