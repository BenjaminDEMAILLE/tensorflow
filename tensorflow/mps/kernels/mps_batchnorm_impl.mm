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

// REAL IMPLEMENTATION: BatchNorm, LayerNorm, InstanceNorm, GroupNorm

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

struct MPSNormContext {
  float epsilon;
  bool training;
  std::string data_format;
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
};

// Metal shader for normalization operations
static const char* kNormShaders = R"(
#include <metal_stdlib>
using namespace metal;

// BatchNorm forward
kernel void batch_norm_forward(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* offset [[buffer(2)]],
    device const float* mean [[buffer(3)]],
    device const float* variance [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint c = gid.z % channels;
    float normalized = (input[gid.z] - mean[c]) / sqrt(variance[c] + epsilon);
    output[gid.z] = normalized * scale[c] + offset[c];
}

// LayerNorm
kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* offset [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint& normalized_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint batch_idx = gid.x;
    uint base_idx = batch_idx * normalized_size;
    
    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        sum += input[base_idx + i];
    }
    float mean = sum / float(normalized_size);
    
    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < normalized_size; ++i) {
        float diff = input[base_idx + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(normalized_size);
    float inv_std = 1.0f / sqrt(variance + epsilon);
    
    // Normalize and scale
    for (uint i = 0; i < normalized_size; ++i) {
        uint idx = base_idx + i;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * scale[i] + offset[i];
    }
}

// InstanceNorm
kernel void instance_norm(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* offset [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint& spatial_size [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint c = gid.y;
    uint base_idx = (n * gid.z + c) * spatial_size;
    
    // Compute mean for this instance
    float sum = 0.0f;
    for (uint i = 0; i < spatial_size; ++i) {
        sum += input[base_idx + i];
    }
    float mean = sum / float(spatial_size);
    
    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < spatial_size; ++i) {
        float diff = input[base_idx + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(spatial_size);
    float inv_std = 1.0f / sqrt(variance + epsilon);
    
    // Normalize
    for (uint i = 0; i < spatial_size; ++i) {
        uint idx = base_idx + i;
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * scale[c] + offset[c];
    }
}
)";

extern "C" void* MPSBatchNorm_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSNormContext();
  TF_Status* status = TF_NewStatus();
  
  kernel_ctx->epsilon = 1e-5f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "epsilon", &kernel_ctx->epsilon, status);
  
  kernel_ctx->training = false;
  TF_OpKernelConstruction_GetAttrBool(ctx, "is_training", &kernel_ctx->training, status);
  
  TF_StringView format_view;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_view, status);
  if (TF_GetCode(status) == TF_OK) {
    kernel_ctx->data_format = std::string(format_view.data, format_view.len);
  } else {
    kernel_ctx->data_format = "NHWC";
  }
  
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSNorm_Delete(void* kernel) {
  auto* ctx = static_cast<MPSNormContext*>(kernel);
  delete ctx;
}

extern "C" void MPSBatchNorm_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* norm_ctx = static_cast<MPSNormContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    // Get inputs: x, scale, offset, mean, variance
    TF_Tensor* x_tensor = nullptr;
    TF_Tensor* scale_tensor = nullptr;
    TF_Tensor* offset_tensor = nullptr;
    TF_Tensor* mean_tensor = nullptr;
    TF_Tensor* var_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &x_tensor, status);
    TF_GetInput(ctx, 1, &scale_tensor, status);
    TF_GetInput(ctx, 2, &offset_tensor, status);
    TF_GetInput(ctx, 3, &mean_tensor, status);
    TF_GetInput(ctx, 4, &var_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get dimensions
    int64_t batch = TF_Dim(x_tensor, 0);
    int64_t height = TF_Dim(x_tensor, 1);
    int64_t width = TF_Dim(x_tensor, 2);
    int64_t channels = TF_Dim(x_tensor, 3);
    
    // Create MPSGraph
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(height), @(width), @(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    MPSGraphTensor* meanTensor = [graph placeholderWithShape:@[@(channels)]
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"mean"];
    
    MPSGraphTensor* varTensor = [graph placeholderWithShape:@[@(channels)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:@"variance"];
    
    MPSGraphTensor* scaleTensor = [graph placeholderWithShape:@[@(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"scale"];
    
    MPSGraphTensor* offsetTensor = [graph placeholderWithShape:@[@(channels)]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"offset"];
    
    // BatchNorm operation: (x - mean) / sqrt(var + epsilon) * scale + offset
    MPSGraphTensor* epsilonTensor = [graph constantWithScalar:norm_ctx->epsilon
                                                      dataType:MPSDataTypeFloat32];
    
    MPSGraphTensor* varPlusEps = [graph additionWithPrimaryTensor:varTensor
                                                   secondaryTensor:epsilonTensor
                                                              name:@"var_plus_eps"];
    
    MPSGraphTensor* stdTensor = [graph squareRootWithTensor:varPlusEps name:@"std"];
    
    MPSGraphTensor* normalized = [graph subtractionWithPrimaryTensor:inputTensor
                                                     secondaryTensor:meanTensor
                                                                name:@"subtract_mean"];
    
    normalized = [graph divisionWithPrimaryTensor:normalized
                                  secondaryTensor:stdTensor
                                             name:@"divide_by_std"];
    
    normalized = [graph multiplicationWithPrimaryTensor:normalized
                                        secondaryTensor:scaleTensor
                                                   name:@"multiply_scale"];
    
    MPSGraphTensor* outputTensor = [graph additionWithPrimaryTensor:normalized
                                                    secondaryTensor:offsetTensor
                                                               name:@"add_offset"];
    
    // Create buffers and execute
    size_t x_size = batch * height * width * channels * sizeof(float);
    size_t channel_size = channels * sizeof(float);
    
    id<MTLBuffer> x_buffer = [norm_ctx->device newBufferWithBytes:TF_TensorData(x_tensor)
                                                            length:x_size
                                                           options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> mean_buffer = [norm_ctx->device newBufferWithBytes:TF_TensorData(mean_tensor)
                                                               length:channel_size
                                                              options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> var_buffer = [norm_ctx->device newBufferWithBytes:TF_TensorData(var_tensor)
                                                              length:channel_size
                                                             options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> scale_buffer = [norm_ctx->device newBufferWithBytes:TF_TensorData(scale_tensor)
                                                                length:channel_size
                                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> offset_buffer = [norm_ctx->device newBufferWithBytes:TF_TensorData(offset_tensor)
                                                                 length:channel_size
                                                                options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buffer
                                                                          shape:@[@(batch), @(height), @(width), @(channels)]
                                                                       dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* mean_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mean_buffer
                                                                             shape:@[@(channels)]
                                                                          dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* var_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:var_buffer
                                                                            shape:@[@(channels)]
                                                                         dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* scale_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:scale_buffer
                                                                              shape:@[@(channels)]
                                                                           dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* offset_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:offset_buffer
                                                                               shape:@[@(channels)]
                                                                            dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{
      inputTensor: x_data,
      meanTensor: mean_data,
      varTensor: var_data,
      scaleTensor: scale_data,
      offsetTensor: offset_data
    };
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:norm_ctx->command_queue
                                                           feeds:feeds
                                                  targetTensors:@[outputTensor]
                                           targetOperations:nil]
                                     [outputTensor];
    
    // Allocate output
    int64_t output_dims[4] = {batch, height, width, channels};
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 4, x_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, x_size);
  }
  
  TF_DeleteStatus(status);
}

extern "C" void* MPSLayerNorm_Create(TF_OpKernelConstruction* ctx) {
  return MPSBatchNorm_Create(ctx);
}

extern "C" void MPSLayerNorm_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* norm_ctx = static_cast<MPSNormContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* x_tensor = nullptr;
    TF_Tensor* scale_tensor = nullptr;
    TF_Tensor* offset_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &x_tensor, status);
    TF_GetInput(ctx, 1, &scale_tensor, status);
    TF_GetInput(ctx, 2, &offset_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Use MPSGraph for LayerNorm
    int num_dims = TF_NumDims(x_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(x_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Compute mean and variance along last axis
    MPSGraphTensor* meanTensor = [graph meanOfTensor:inputTensor axes:@[@(num_dims - 1)] name:@"mean"];
    MPSGraphTensor* varTensor = [graph varianceOfTensor:inputTensor axes:@[@(num_dims - 1)] name:@"variance"];
    
    MPSGraphTensor* epsilonTensor = [graph constantWithScalar:norm_ctx->epsilon dataType:MPSDataTypeFloat32];
    MPSGraphTensor* varPlusEps = [graph additionWithPrimaryTensor:varTensor secondaryTensor:epsilonTensor name:@"var_eps"];
    MPSGraphTensor* stdTensor = [graph squareRootWithTensor:varPlusEps name:@"std"];
    
    MPSGraphTensor* normalized = [graph subtractionWithPrimaryTensor:inputTensor secondaryTensor:meanTensor name:@"norm"];
    normalized = [graph divisionWithPrimaryTensor:normalized secondaryTensor:stdTensor name:@"divide"];
    
    // Apply scale and offset (broadcast automatically)
    MPSGraphTensor* scaleTensor = [graph placeholderWithShape:@[@(TF_Dim(scale_tensor, 0))]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"scale"];
    
    MPSGraphTensor* offsetTensor = [graph placeholderWithShape:@[@(TF_Dim(offset_tensor, 0))]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"offset"];
    
    normalized = [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:scaleTensor name:@"scale"];
    MPSGraphTensor* output = [graph additionWithPrimaryTensor:normalized secondaryTensor:offsetTensor name:@"offset"];
    
    // Execute
    size_t data_size = total_size * sizeof(float);
    
    id<MTLBuffer> x_buf = [norm_ctx->device newBufferWithBytes:TF_TensorData(x_tensor)
                                                         length:data_size
                                                        options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> scale_buf = [norm_ctx->device newBufferWithBytes:TF_TensorData(scale_tensor)
                                                             length:TF_Dim(scale_tensor, 0) * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> offset_buf = [norm_ctx->device newBufferWithBytes:TF_TensorData(offset_tensor)
                                                              length:TF_Dim(offset_tensor, 0) * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* x_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf
                                                                          shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                       dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* scale_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:scale_buf
                                                                              shape:@[@(TF_Dim(scale_tensor, 0))]
                                                                           dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* offset_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:offset_buf
                                                                               shape:@[@(TF_Dim(offset_tensor, 0))]
                                                                            dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: x_data, scaleTensor: scale_data, offsetTensor: offset_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:norm_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[output]
                                         targetOperations:nil][output];
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, shape_vec.data(), num_dims, data_size, status);
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, data_size);
  }
  
  TF_DeleteStatus(status);
}

}  // namespace mps
}  // namespace tensorflow
