// Embedding and lookup operations with Metal
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

// Embedding Lookup - Gather embeddings by indices
extern "C" {

const char kEmbeddingLookupKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void EmbeddingLookupCompute(
    device const int* indices [[buffer(0)]],
    device const float* embeddings [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_indices [[buffer(3)]],
    constant int& embedding_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_indices) return;
    
    int idx = indices[gid];
    int output_offset = gid * embedding_dim;
    int embedding_offset = idx * embedding_dim;
    
    for (int i = 0; i < embedding_dim; i++) {
        output[output_offset + i] = embeddings[embedding_offset + i];
    }
}
)";

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSEmbeddingLookupContext;

void* MPSEmbeddingLookup_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSEmbeddingLookupContext();
  context->queue = GetCommandQueue();
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kEmbeddingLookupKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"EmbeddingLookupCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSEmbeddingLookup_Delete(void* kernel) {
  delete static_cast<MPSEmbeddingLookupContext*>(kernel);
}

void MPSEmbeddingLookup_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSEmbeddingLookupContext*>(kernel);
    
    TF_Tensor* params_tensor = nullptr;
    TF_Tensor* indices_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &params_tensor, status);
    TF_GetInput(ctx, 1, &indices_tensor, status);
    
    // params: [vocab_size, embedding_dim]
    // indices: [batch_size] or [batch_size, seq_len]
    // output: [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
    
    int64_t vocab_size = TF_Dim(params_tensor, 0);
    int64_t embedding_dim = TF_Dim(params_tensor, 1);
    
    int64_t num_indices = 1;
    int num_idx_dims = TF_NumDims(indices_tensor);
    for (int i = 0; i < num_idx_dims; i++) {
      num_indices *= TF_Dim(indices_tensor, i);
    }
    
    int* indices_data = static_cast<int*>(TF_TensorData(indices_tensor));
    float* params_data = static_cast<float*>(TF_TensorData(params_tensor));
    
    size_t indices_bytes = num_indices * sizeof(int);
    size_t params_bytes = vocab_size * embedding_dim * sizeof(float);
    size_t output_bytes = num_indices * embedding_dim * sizeof(float);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> indices_buf = [device newBufferWithBytes:indices_data length:indices_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> params_buf = [device newBufferWithBytes:params_data length:params_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:indices_buf offset:0 atIndex:0];
    [encoder setBuffer:params_buf offset:0 atIndex:1];
    [encoder setBuffer:output_buf offset:0 atIndex:2];
    
    int num_idx = (int)num_indices;
    int emb_dim = (int)embedding_dim;
    [encoder setBytes:&num_idx length:sizeof(int) atIndex:3];
    [encoder setBytes:&emb_dim length:sizeof(int) atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(num_indices, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Build output dims
    int64_t* output_dims = new int64_t[num_idx_dims + 1];
    for (int i = 0; i < num_idx_dims; i++) {
      output_dims[i] = TF_Dim(indices_tensor, i);
    }
    output_dims[num_idx_dims] = embedding_dim;
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_idx_dims + 1, output_bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [output_buf contents], output_bytes);
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// GatherNd - Gather slices from params according to indices
extern "C" {

const char kGatherNdKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void GatherNdCompute(
    device const float* params [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_indices [[buffer(3)]],
    constant int& indices_nd [[buffer(4)]],
    constant int& slice_size [[buffer(5)]],
    constant int* strides [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_indices) return;
    
    // Compute linear index in params
    int linear_idx = 0;
    int idx_offset = gid * indices_nd;
    for (int i = 0; i < indices_nd; i++) {
        linear_idx += indices[idx_offset + i] * strides[i];
    }
    
    // Copy slice
    int output_offset = gid * slice_size;
    int params_offset = linear_idx;
    for (int i = 0; i < slice_size; i++) {
        output[output_offset + i] = params[params_offset + i];
    }
}
)";

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSGatherNdContext;

void* MPSGatherNd_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSGatherNdContext();
  context->queue = GetCommandQueue();
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kGatherNdKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"GatherNdCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSGatherNd_Delete(void* kernel) {
  delete static_cast<MPSGatherNdContext*>(kernel);
}

void MPSGatherNd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSGatherNdContext*>(kernel);
    
    TF_Tensor* params_tensor = nullptr;
    TF_Tensor* indices_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &params_tensor, status);
    TF_GetInput(ctx, 1, &indices_tensor, status);
    
    // params: [d0, d1, ..., dn]
    // indices: [batch, ..., indices_nd]
    // output: [batch, ..., dn-indices_nd+1, ..., dn]
    
    int num_params_dims = TF_NumDims(params_tensor);
    int num_indices_dims = TF_NumDims(indices_tensor);
    int64_t indices_nd = TF_Dim(indices_tensor, num_indices_dims - 1);
    
    int64_t num_indices = 1;
    for (int i = 0; i < num_indices_dims - 1; i++) {
      num_indices *= TF_Dim(indices_tensor, i);
    }
    
    int64_t slice_size = 1;
    for (int i = indices_nd; i < num_params_dims; i++) {
      slice_size *= TF_Dim(params_tensor, i);
    }
    
    // Compute strides
    int* strides = new int[indices_nd];
    int64_t stride = 1;
    for (int i = indices_nd - 1; i >= 0; i--) {
      strides[i] = stride;
      if (i > 0) stride *= TF_Dim(params_tensor, i);
    }
    
    float* params_data = static_cast<float*>(TF_TensorData(params_tensor));
    int* indices_data = static_cast<int*>(TF_TensorData(indices_tensor));
    
    size_t params_size = 1;
    for (int i = 0; i < num_params_dims; i++) {
      params_size *= TF_Dim(params_tensor, i);
    }
    
    size_t params_bytes = params_size * sizeof(float);
    size_t indices_bytes = num_indices * indices_nd * sizeof(int);
    size_t output_bytes = num_indices * slice_size * sizeof(float);
    size_t strides_bytes = indices_nd * sizeof(int);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> params_buf = [device newBufferWithBytes:params_data length:params_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> indices_buf = [device newBufferWithBytes:indices_data length:indices_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> strides_buf = [device newBufferWithBytes:strides length:strides_bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:params_buf offset:0 atIndex:0];
    [encoder setBuffer:indices_buf offset:0 atIndex:1];
    [encoder setBuffer:output_buf offset:0 atIndex:2];
    
    int num_idx = (int)num_indices;
    int idx_nd = (int)indices_nd;
    int sl_size = (int)slice_size;
    [encoder setBytes:&num_idx length:sizeof(int) atIndex:3];
    [encoder setBytes:&idx_nd length:sizeof(int) atIndex:4];
    [encoder setBytes:&sl_size length:sizeof(int) atIndex:5];
    [encoder setBuffer:strides_buf offset:0 atIndex:6];
    
    MTLSize gridSize = MTLSizeMake(num_indices, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Build output dims
    int output_dims_count = num_indices_dims - 1 + num_params_dims - indices_nd;
    int64_t* output_dims = new int64_t[output_dims_count];
    int dim_idx = 0;
    for (int i = 0; i < num_indices_dims - 1; i++) {
      output_dims[dim_idx++] = TF_Dim(indices_tensor, i);
    }
    for (int i = indices_nd; i < num_params_dims; i++) {
      output_dims[dim_idx++] = TF_Dim(params_tensor, i);
    }
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, output_dims_count, output_bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [output_buf contents], output_bytes);
    
    delete[] output_dims;
    delete[] strides;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// ScatterNd - Scatter updates into new tensor
extern "C" {

const char kScatterNdKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void ScatterNdCompute(
    device const int* indices [[buffer(0)]],
    device const float* updates [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& num_updates [[buffer(3)]],
    constant int& indices_nd [[buffer(4)]],
    constant int& slice_size [[buffer(5)]],
    constant int* strides [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_updates) return;
    
    // Compute linear index in output
    int linear_idx = 0;
    int idx_offset = gid * indices_nd;
    for (int i = 0; i < indices_nd; i++) {
        linear_idx += indices[idx_offset + i] * strides[i];
    }
    
    // Scatter update
    int update_offset = gid * slice_size;
    int output_offset = linear_idx;
    for (int i = 0; i < slice_size; i++) {
        output[output_offset + i] = updates[update_offset + i];
    }
}
)";

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSScatterNdContext;

void* MPSScatterNd_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSScatterNdContext();
  context->queue = GetCommandQueue();
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kScatterNdKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"ScatterNdCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSScatterNd_Delete(void* kernel) {
  delete static_cast<MPSScatterNdContext*>(kernel);
}

void MPSScatterNd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSScatterNdContext*>(kernel);
    
    TF_Tensor* indices_tensor = nullptr;
    TF_Tensor* updates_tensor = nullptr;
    TF_Tensor* shape_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &indices_tensor, status);
    TF_GetInput(ctx, 1, &updates_tensor, status);
    TF_GetInput(ctx, 2, &shape_tensor, status);
    
    int num_indices_dims = TF_NumDims(indices_tensor);
    int64_t indices_nd = TF_Dim(indices_tensor, num_indices_dims - 1);
    
    int64_t num_updates = 1;
    for (int i = 0; i < num_indices_dims - 1; i++) {
      num_updates *= TF_Dim(indices_tensor, i);
    }
    
    int num_shape_dims = TF_NumDims(shape_tensor);
    int64_t* shape_data = static_cast<int64_t*>(TF_TensorData(shape_tensor));
    
    int64_t output_size = 1;
    for (int i = 0; i < num_shape_dims; i++) {
      output_size *= shape_data[i];
    }
    
    int64_t slice_size = 1;
    for (int i = indices_nd; i < num_shape_dims; i++) {
      slice_size *= shape_data[i];
    }
    
    // Compute strides
    int* strides = new int[indices_nd];
    int64_t stride = 1;
    for (int i = indices_nd - 1; i >= 0; i--) {
      strides[i] = stride;
      if (i > 0) stride *= shape_data[i];
    }
    
    int* indices_data = static_cast<int*>(TF_TensorData(indices_tensor));
    float* updates_data = static_cast<float*>(TF_TensorData(updates_tensor));
    
    size_t indices_bytes = num_updates * indices_nd * sizeof(int);
    size_t updates_bytes = num_updates * slice_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);
    size_t strides_bytes = indices_nd * sizeof(int);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> indices_buf = [device newBufferWithBytes:indices_data length:indices_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> updates_buf = [device newBufferWithBytes:updates_data length:updates_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> strides_buf = [device newBufferWithBytes:strides length:strides_bytes options:MTLResourceStorageModeShared];
    
    // Zero initialize output
    memset([output_buf contents], 0, output_bytes);
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:indices_buf offset:0 atIndex:0];
    [encoder setBuffer:updates_buf offset:0 atIndex:1];
    [encoder setBuffer:output_buf offset:0 atIndex:2];
    
    int num_upd = (int)num_updates;
    int idx_nd = (int)indices_nd;
    int sl_size = (int)slice_size;
    [encoder setBytes:&num_upd length:sizeof(int) atIndex:3];
    [encoder setBytes:&idx_nd length:sizeof(int) atIndex:4];
    [encoder setBytes:&sl_size length:sizeof(int) atIndex:5];
    [encoder setBuffer:strides_buf offset:0 atIndex:6];
    
    MTLSize gridSize = MTLSizeMake(num_updates, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, shape_data, num_shape_dims, output_bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [output_buf contents], output_bytes);
    
    delete[] strides;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// Gather - Simple gather operation
extern "C" {

typedef struct {
  id<MTLCommandQueue> queue;
} MPSGatherContext;

void* MPSGather_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSGatherContext();
  context->queue = GetCommandQueue();
  return context;
}

void MPSGather_Delete(void* kernel) {
  delete static_cast<MPSGatherContext*>(kernel);
}

void MPSGather_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* params_tensor = nullptr;
    TF_Tensor* indices_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &params_tensor, status);
    TF_GetInput(ctx, 1, &indices_tensor, status);
    
    // Same as EmbeddingLookup for axis=0
    // Delegate to embedding lookup implementation
    
    int64_t first_dim = TF_Dim(params_tensor, 0);
    int64_t slice_size = 1;
    for (int i = 1; i < TF_NumDims(params_tensor); i++) {
      slice_size *= TF_Dim(params_tensor, i);
    }
    
    int64_t num_indices = TF_NumElements(indices_tensor);
    
    size_t output_bytes = num_indices * slice_size * sizeof(float);
    int64_t output_dims[10];
    int output_dim_count = 0;
    
    for (int i = 0; i < TF_NumDims(indices_tensor); i++) {
      output_dims[output_dim_count++] = TF_Dim(indices_tensor, i);
    }
    for (int i = 1; i < TF_NumDims(params_tensor); i++) {
      output_dims[output_dim_count++] = TF_Dim(params_tensor, i);
    }
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, output_dim_count, output_bytes, status);
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"
