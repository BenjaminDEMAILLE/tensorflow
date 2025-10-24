/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// REAL IMPLEMENTATION: Softmax and reduction operations

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <vector>

namespace tensorflow {
namespace mps {

struct MPSReductionContext {
  std::vector<int64_t> axes;
  bool keep_dims;
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLLibrary> library;
  id<MTLComputePipelineState> softmax_pipeline;
};

static const char* kSoftmaxShader = R"(
#include <metal_stdlib>
using namespace metal;

// Optimized Softmax with shared memory for Apple Silicon
kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_classes [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    
    if (gid >= batch_size) return;
    
    uint base_idx = gid * num_classes;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (uint i = 0; i < num_classes; ++i) {
        max_val = max(max_val, input[base_idx + i]);
    }
    shared_max[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < num_classes; ++i) {
        sum += exp(input[base_idx + i] - max_val);
    }
    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize
    for (uint i = 0; i < num_classes; ++i) {
        output[base_idx + i] = exp(input[base_idx + i] - max_val) / sum;
    }
}

// LogSoftmax
kernel void log_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_classes [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch_size) return;
    
    uint base_idx = gid * num_classes;
    
    // Find max
    float max_val = -INFINITY;
    for (uint i = 0; i < num_classes; ++i) {
        max_val = max(max_val, input[base_idx + i]);
    }
    
    // Compute log-sum-exp
    float sum = 0.0f;
    for (uint i = 0; i < num_classes; ++i) {
        sum += exp(input[base_idx + i] - max_val);
    }
    float log_sum = log(sum) + max_val;
    
    // Compute log softmax
    for (uint i = 0; i < num_classes; ++i) {
        output[base_idx + i] = input[base_idx + i] - log_sum;
    }
}
)";

extern "C" void* MPSSoftmax_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSReductionContext();
  TF_Status* status = TF_NewStatus();
  
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  // Compile shader
  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:kSoftmaxShader];
  kernel_ctx->library = [kernel_ctx->device newLibraryWithSource:shaderSource options:nil error:&error];
  
  if (!error) {
    id<MTLFunction> function = [kernel_ctx->library newFunctionWithName:@"softmax"];
    kernel_ctx->softmax_pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:function error:&error];
  }
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void* MPSReduction_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSReductionContext();
  TF_Status* status = TF_NewStatus();
  
  int64_t* axes = nullptr;
  int axes_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "axes", &axes, &axes_len, status);
  if (TF_GetCode(status) == TF_OK && axes_len > 0) {
    kernel_ctx->axes.assign(axes, axes + axes_len);
  }
  
  kernel_ctx->keep_dims = false;
  TF_OpKernelConstruction_GetAttrBool(ctx, "keep_dims", &kernel_ctx->keep_dims, status);
  
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSReduction_Delete(void* kernel) {
  auto* ctx = static_cast<MPSReductionContext*>(kernel);
  delete ctx;
}

extern "C" void MPSSoftmax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* softmax_ctx = static_cast<MPSReductionContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_GetInput(ctx, 0, &input_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Use MPSGraph for Softmax
    int num_dims = TF_NumDims(input_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(input_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Softmax along last axis
    MPSGraphTensor* outputTensor = [graph softMaxWithTensor:inputTensor
                                                       axis:num_dims - 1
                                                       name:@"softmax"];
    
    // Create buffer and execute
    size_t data_size = total_size * sizeof(float);
    
    id<MTLBuffer> input_buffer = [softmax_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                                   length:data_size
                                                                  options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                              shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                           dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: input_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:softmax_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    // Allocate output
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, shape_vec.data(), num_dims, data_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, data_size);
  }
  
  TF_DeleteStatus(status);
}

// Generic reduction using MPSGraph
static void ComputeReduction(MPSReductionContext* ctx, TF_OpKernelContext* tf_ctx,
                             MPSGraphTensor* (^reductionBlock)(MPSGraph*, MPSGraphTensor*, NSArray*)) {
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_GetInput(tf_ctx, 0, &input_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(input_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(input_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    // Convert axes to NSArray
    NSMutableArray* axes_array = [NSMutableArray array];
    for (int64_t axis : ctx->axes) {
      [axes_array addObject:@(axis)];
    }
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Apply reduction
    MPSGraphTensor* outputTensor = reductionBlock(graph, inputTensor, axes_array);
    
    // Execute
    size_t input_size = total_size * sizeof(float);
    
    id<MTLBuffer> input_buffer = [ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                           length:input_size
                                                          options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                              shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                           dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: input_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    // Calculate output shape
    std::vector<int64_t> output_shape;
    for (int i = 0; i < num_dims; ++i) {
      bool is_reduced = false;
      for (int64_t axis : ctx->axes) {
        if (i == axis) {
          is_reduced = true;
          break;
        }
      }
      
      if (is_reduced) {
        if (ctx->keep_dims) {
          output_shape.push_back(1);
        }
      } else {
        output_shape.push_back(TF_Dim(input_tensor, i));
      }
    }
    
    size_t output_size = sizeof(float);
    for (int64_t dim : output_shape) {
      output_size *= dim;
    }
    
    TF_Tensor* output_tf = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_shape.data(), 
                                             output_shape.size(), output_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, output_size);
  }
  
  TF_DeleteStatus(status);
}

extern "C" void MPSSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* red_ctx = static_cast<MPSReductionContext*>(kernel);
  ComputeReduction(red_ctx, ctx, ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* input, NSArray* axes) {
    return [graph reductionSumWithTensor:input axes:axes name:@"sum"];
  });
}

extern "C" void MPSMean_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* red_ctx = static_cast<MPSReductionContext*>(kernel);
  ComputeReduction(red_ctx, ctx, ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* input, NSArray* axes) {
    return [graph reductionMeanWithTensor:input axes:axes name:@"mean"];
  });
}

extern "C" void MPSMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* red_ctx = static_cast<MPSReductionContext*>(kernel);
  ComputeReduction(red_ctx, ctx, ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* input, NSArray* axes) {
    return [graph reductionMaximumWithTensor:input axes:axes name:@"max"];
  });
}

extern "C" void MPSMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* red_ctx = static_cast<MPSReductionContext*>(kernel);
  ComputeReduction(red_ctx, ctx, ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* input, NSArray* axes) {
    return [graph reductionMinimumWithTensor:input axes:axes name:@"min"];
  });
}

extern "C" void MPSArgMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* red_ctx = static_cast<MPSReductionContext*>(kernel);
  ComputeReduction(red_ctx, ctx, ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* input, NSArray* axes) {
    return [graph reductionArgMaximumWithTensor:input axis:[axes[0] intValue] name:@"argmax"];
  });
}

extern "C" void MPSArgMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* red_ctx = static_cast<MPSReductionContext*>(kernel);
  ComputeReduction(red_ctx, ctx, ^MPSGraphTensor*(MPSGraph* graph, MPSGraphTensor* input, NSArray* axes) {
    return [graph reductionArgMinimumWithTensor:input axis:[axes[0] intValue] name:@"argmin"];
  });
}

}  // namespace mps
}  // namespace tensorflow
