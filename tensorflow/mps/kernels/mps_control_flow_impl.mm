// Control flow and functional operations with Metal
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <Metal/Metal.h>

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

// Select/Where - Element-wise conditional selection
extern "C" {

const char kSelectKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void SelectCompute(
    device const bool* condition [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* y [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = condition[gid] ? x[gid] : y[gid];
}
)";

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSSelectContext;

void* MPSSelect_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSSelectContext();
  context->queue = GetCommandQueue();
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kSelectKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"SelectCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSSelect_Delete(void* kernel) {
  delete static_cast<MPSSelectContext*>(kernel);
}

void MPSSelect_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSSelectContext*>(kernel);
    
    TF_Tensor* condition_tensor = nullptr;
    TF_Tensor* x_tensor = nullptr;
    TF_Tensor* y_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &condition_tensor, status);
    TF_GetInput(ctx, 1, &x_tensor, status);
    TF_GetInput(ctx, 2, &y_tensor, status);
    
    int64_t size = TF_NumElements(x_tensor);
    
    bool* condition_data = static_cast<bool*>(TF_TensorData(condition_tensor));
    float* x_data = static_cast<float*>(TF_TensorData(x_tensor));
    float* y_data = static_cast<float*>(TF_TensorData(y_tensor));
    
    size_t condition_bytes = size * sizeof(bool);
    size_t data_bytes = size * sizeof(float);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> condition_buf = [device newBufferWithBytes:condition_data length:condition_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> x_buf = [device newBufferWithBytes:x_data length:data_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> y_buf = [device newBufferWithBytes:y_data length:data_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:data_bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:condition_buf offset:0 atIndex:0];
    [encoder setBuffer:x_buf offset:0 atIndex:1];
    [encoder setBuffer:y_buf offset:0 atIndex:2];
    [encoder setBuffer:output_buf offset:0 atIndex:3];
    
    int sz = (int)size;
    [encoder setBytes:&sz length:sizeof(int) atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    int num_dims = TF_NumDims(x_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(x_tensor, i);
    }
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, data_bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [output_buf contents], data_bytes);
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// TopK - Find top K values and indices
extern "C" {

typedef struct {
  int k;
  bool sorted;
  id<MTLCommandQueue> queue;
} MPSTopKContext;

void* MPSTopK_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSTopKContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "k", &context->k, status);
  TF_OpKernelConstruction_GetAttrBool(ctx, "sorted", &context->sorted, status);
  TF_DeleteStatus(status);
  
  return context;
}

void MPSTopK_Delete(void* kernel) {
  delete static_cast<MPSTopKContext*>(kernel);
}

void MPSTopK_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSTopKContext*>(kernel);
    
    TF_Tensor* input_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    
    // Input: [..., n]
    // Output: [..., k] for both values and indices
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t n = TF_Dim(input_tensor, num_dims - 1);
    int k = context->k;
    
    int64_t batch_size = 1;
    for (int i = 0; i < num_dims - 1; i++) {
      batch_size *= TF_Dim(input_tensor, i);
    }
    
    float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
    
    // Allocate outputs
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims - 1; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    output_dims[num_dims - 1] = k;
    
    size_t values_bytes = batch_size * k * sizeof(float);
    size_t indices_bytes = batch_size * k * sizeof(int);
    
    TF_Tensor* values_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, values_bytes, status);
    TF_Tensor* indices_tensor = TF_AllocateOutput(ctx, 1, TF_INT32, output_dims, num_dims, indices_bytes, status);
    
    float* values_data = static_cast<float*>(TF_TensorData(values_tensor));
    int* indices_data = static_cast<int*>(TF_TensorData(indices_tensor));
    
    // Simple CPU implementation - find top k for each batch
    for (int64_t b = 0; b < batch_size; b++) {
      float* batch_input = input_data + b * n;
      float* batch_values = values_data + b * k;
      int* batch_indices = indices_data + b * k;
      
      // Create index array
      int* temp_indices = new int[n];
      for (int i = 0; i < n; i++) {
        temp_indices[i] = i;
      }
      
      // Partial sort to find top k
      for (int i = 0; i < k; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++) {
          if (batch_input[temp_indices[j]] > batch_input[temp_indices[max_idx]]) {
            max_idx = j;
          }
        }
        // Swap
        int temp = temp_indices[i];
        temp_indices[i] = temp_indices[max_idx];
        temp_indices[max_idx] = temp;
        
        batch_values[i] = batch_input[temp_indices[i]];
        batch_indices[i] = temp_indices[i];
      }
      
      delete[] temp_indices;
    }
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// Unique - Find unique elements
extern "C" {

typedef struct {
  id<MTLCommandQueue> queue;
} MPSUniqueContext;

void* MPSUnique_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSUniqueContext();
  context->queue = GetCommandQueue();
  return context;
}

void MPSUnique_Delete(void* kernel) {
  delete static_cast<MPSUniqueContext*>(kernel);
}

void MPSUnique_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    int* input_data = static_cast<int*>(TF_TensorData(input_tensor));
    
    // Find unique elements (simplified)
    std::vector<int> unique_vals;
    std::vector<int> indices;
    
    for (int64_t i = 0; i < size; i++) {
      bool found = false;
      for (size_t j = 0; j < unique_vals.size(); j++) {
        if (input_data[i] == unique_vals[j]) {
          indices.push_back(j);
          found = true;
          break;
        }
      }
      if (!found) {
        indices.push_back(unique_vals.size());
        unique_vals.push_back(input_data[i]);
      }
    }
    
    // Output unique values
    int64_t unique_size = unique_vals.size();
    int64_t unique_dims[] = {unique_size};
    size_t unique_bytes = unique_size * sizeof(int);
    TF_Tensor* unique_tensor = TF_AllocateOutput(ctx, 0, TF_INT32, unique_dims, 1, unique_bytes, status);
    int* unique_data = static_cast<int*>(TF_TensorData(unique_tensor));
    memcpy(unique_data, unique_vals.data(), unique_bytes);
    
    // Output indices
    int64_t indices_dims[] = {size};
    size_t indices_bytes = size * sizeof(int);
    TF_Tensor* indices_tensor = TF_AllocateOutput(ctx, 1, TF_INT32, indices_dims, 1, indices_bytes, status);
    int* indices_data = static_cast<int*>(TF_TensorData(indices_tensor));
    memcpy(indices_data, indices.data(), indices_bytes);
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// Cumsum - Cumulative sum
extern "C" {

const char kCumsumKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void CumsumCompute(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float sum = 0.0f;
    for (int i = 0; i <= gid; i++) {
        sum += input[i];
    }
    output[gid] = sum;
}
)";

typedef struct {
  bool exclusive;
  bool reverse;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSCumsumContext;

void* MPSCumsum_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSCumsumContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrBool(ctx, "exclusive", &context->exclusive, status);
  TF_OpKernelConstruction_GetAttrBool(ctx, "reverse", &context->reverse, status);
  TF_DeleteStatus(status);
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kCumsumKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"CumsumCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSCumsum_Delete(void* kernel) {
  delete static_cast<MPSCumsumContext*>(kernel);
}

void MPSCumsum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSCumsumContext*>(kernel);
    
    TF_Tensor* input_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
    
    size_t bytes = size * sizeof(float);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> input_buf = [device newBufferWithBytes:input_data length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:input_buf offset:0 atIndex:0];
    [encoder setBuffer:output_buf offset:0 atIndex:1];
    
    int sz = (int)size;
    [encoder setBytes:&sz length:sizeof(int) atIndex:2];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [output_buf contents], bytes);
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// Range - Generate range of numbers
extern "C" {

typedef struct {
  id<MTLCommandQueue> queue;
} MPSRangeContext;

void* MPSRange_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSRangeContext();
  context->queue = GetCommandQueue();
  return context;
}

void MPSRange_Delete(void* kernel) {
  delete static_cast<MPSRangeContext*>(kernel);
}

void MPSRange_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* start_tensor = nullptr;
    TF_Tensor* limit_tensor = nullptr;
    TF_Tensor* delta_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &start_tensor, status);
    TF_GetInput(ctx, 1, &limit_tensor, status);
    TF_GetInput(ctx, 2, &delta_tensor, status);
    
    int start = *static_cast<int*>(TF_TensorData(start_tensor));
    int limit = *static_cast<int*>(TF_TensorData(limit_tensor));
    int delta = *static_cast<int*>(TF_TensorData(delta_tensor));
    
    int64_t size = (limit - start + delta - 1) / delta;
    
    int64_t output_dims[] = {size};
    size_t bytes = size * sizeof(int);
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_INT32, output_dims, 1, bytes, status);
    int* output_data = static_cast<int*>(TF_TensorData(output_tensor));
    
    for (int64_t i = 0; i < size; i++) {
      output_data[i] = start + i * delta;
    }
    
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// Cast - Type conversion
extern "C" {

typedef struct {
  TF_DataType dst_type;
  id<MTLCommandQueue> queue;
} MPSCastContext;

void* MPSCast_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSCastContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrType(ctx, "DstT", &context->dst_type, status);
  TF_DeleteStatus(status);
  
  return context;
}

void MPSCast_Delete(void* kernel) {
  delete static_cast<MPSCastContext*>(kernel);
}

void MPSCast_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSCastContext*>(kernel);
    
    TF_Tensor* input_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    TF_DataType src_type = TF_TensorType(input_tensor);
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    
    size_t dst_size = size;
    if (context->dst_type == TF_FLOAT) {
      dst_size *= sizeof(float);
    } else if (context->dst_type == TF_INT32) {
      dst_size *= sizeof(int32_t);
    } else if (context->dst_type == TF_INT64) {
      dst_size *= sizeof(int64_t);
    }
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, context->dst_type, output_dims, num_dims, dst_size, status);
    
    // Simple type conversion (CPU fallback)
    if (src_type == TF_FLOAT && context->dst_type == TF_INT32) {
      float* src = static_cast<float*>(TF_TensorData(input_tensor));
      int32_t* dst = static_cast<int32_t*>(TF_TensorData(output_tensor));
      for (int64_t i = 0; i < size; i++) {
        dst[i] = (int32_t)src[i];
      }
    } else if (src_type == TF_INT32 && context->dst_type == TF_FLOAT) {
      int32_t* src = static_cast<int32_t*>(TF_TensorData(input_tensor));
      float* dst = static_cast<float*>(TF_TensorData(output_tensor));
      for (int64_t i = 0; i < size; i++) {
        dst[i] = (float)src[i];
      }
    }
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"
