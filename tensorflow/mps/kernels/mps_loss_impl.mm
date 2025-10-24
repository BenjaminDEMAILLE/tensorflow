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

// REAL IMPLEMENTATION: Loss functions with Metal compute

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

struct MPSLossContext {
  float delta;  // For Huber loss
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLLibrary> library;
  
  id<MTLComputePipelineState> mse_pipeline;
  id<MTLComputePipelineState> mae_pipeline;
  id<MTLComputePipelineState> huber_pipeline;
  id<MTLComputePipelineState> hinge_pipeline;
};

static const char* kLossShaders = R"(
#include <metal_stdlib>
using namespace metal;

// Mean Squared Error
kernel void mse_loss(
    device const float* predictions [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float diff = predictions[gid] - labels[gid];
    output[gid] = diff * diff;
}

// Mean Absolute Error
kernel void mae_loss(
    device const float* predictions [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    output[gid] = abs(predictions[gid] - labels[gid]);
}

// Huber Loss
kernel void huber_loss(
    device const float* predictions [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& delta [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    float diff = abs(predictions[gid] - labels[gid]);
    if (diff <= delta) {
        output[gid] = 0.5f * diff * diff;
    } else {
        output[gid] = delta * (diff - 0.5f * delta);
    }
}

// Hinge Loss
kernel void hinge_loss(
    device const float* predictions [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    output[gid] = max(0.0f, 1.0f - labels[gid] * predictions[gid]);
}
)";

extern "C" void* MPSLoss_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSLossContext();
  TF_Status* status = TF_NewStatus();
  
  kernel_ctx->delta = 1.0f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "delta", &kernel_ctx->delta, status);
  
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  // Compile shaders
  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:kLossShaders];
  kernel_ctx->library = [kernel_ctx->device newLibraryWithSource:shaderSource options:nil error:&error];
  
  if (!error) {
    id<MTLFunction> mse_func = [kernel_ctx->library newFunctionWithName:@"mse_loss"];
    id<MTLFunction> mae_func = [kernel_ctx->library newFunctionWithName:@"mae_loss"];
    id<MTLFunction> huber_func = [kernel_ctx->library newFunctionWithName:@"huber_loss"];
    id<MTLFunction> hinge_func = [kernel_ctx->library newFunctionWithName:@"hinge_loss"];
    
    kernel_ctx->mse_pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:mse_func error:&error];
    kernel_ctx->mae_pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:mae_func error:&error];
    kernel_ctx->huber_pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:huber_func error:&error];
    kernel_ctx->hinge_pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:hinge_func error:&error];
  }
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSLoss_Delete(void* kernel) {
  auto* ctx = static_cast<MPSLossContext*>(kernel);
  delete ctx;
}

static void ComputeLoss(MPSLossContext* ctx, TF_OpKernelContext* tf_ctx, 
                        id<MTLComputePipelineState> pipeline, bool use_delta = false) {
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* pred_tensor = nullptr;
    TF_Tensor* labels_tensor = nullptr;
    
    TF_GetInput(tf_ctx, 0, &pred_tensor, status);
    TF_GetInput(tf_ctx, 1, &labels_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(pred_tensor);
    size_t num_elements = 1;
    std::vector<int64_t> dims;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(pred_tensor, i);
      dims.push_back(dim);
      num_elements *= dim;
    }
    
    size_t buffer_size = num_elements * sizeof(float);
    
    id<MTLBuffer> pred_buffer = [ctx->device newBufferWithBytes:TF_TensorData(pred_tensor)
                                                          length:buffer_size
                                                         options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> labels_buffer = [ctx->device newBufferWithBytes:TF_TensorData(labels_tensor)
                                                            length:buffer_size
                                                           options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> output_buffer = [ctx->device newBufferWithLength:buffer_size
                                                            options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:pred_buffer offset:0 atIndex:0];
    [encoder setBuffer:labels_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    
    if (use_delta) {
      id<MTLBuffer> delta_buffer = [ctx->device newBufferWithBytes:&ctx->delta
                                                            length:sizeof(float)
                                                           options:MTLResourceStorageModeShared];
      [encoder setBuffer:delta_buffer offset:0 atIndex:3];
    }
    
    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake((num_elements + 255) / 256, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    TF_Tensor* output_tf = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims.data(), num_dims, buffer_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [output_buffer contents], buffer_size);
  }
  
  TF_DeleteStatus(status);
}

extern "C" void MPSMSE_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* loss_ctx = static_cast<MPSLossContext*>(kernel);
  ComputeLoss(loss_ctx, ctx, loss_ctx->mse_pipeline);
}

extern "C" void MPSMAE_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* loss_ctx = static_cast<MPSLossContext*>(kernel);
  ComputeLoss(loss_ctx, ctx, loss_ctx->mae_pipeline);
}

extern "C" void MPSHuber_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* loss_ctx = static_cast<MPSLossContext*>(kernel);
  ComputeLoss(loss_ctx, ctx, loss_ctx->huber_pipeline, true);
}

extern "C" void MPSHinge_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* loss_ctx = static_cast<MPSLossContext*>(kernel);
  ComputeLoss(loss_ctx, ctx, loss_ctx->hinge_pipeline);
}

// Cross Entropy using MPSGraph
extern "C" void MPSCrossEntropy_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* loss_ctx = static_cast<MPSLossContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* pred_tensor = nullptr;
    TF_Tensor* labels_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &pred_tensor, status);
    TF_GetInput(ctx, 1, &labels_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(pred_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(pred_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* predTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"predictions"];
    
    MPSGraphTensor* labelsTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"labels"];
    
    // Cross entropy: -sum(labels * log(predictions))
    MPSGraphTensor* logPred = [graph logarithmWithTensor:predTensor name:@"log_pred"];
    MPSGraphTensor* mul = [graph multiplicationWithPrimaryTensor:labelsTensor secondaryTensor:logPred name:@"mul"];
    MPSGraphTensor* neg = [graph negativeWithTensor:mul name:@"neg"];
    
    size_t data_size = total_size * sizeof(float);
    
    id<MTLBuffer> pred_buffer = [loss_ctx->device newBufferWithBytes:TF_TensorData(pred_tensor)
                                                               length:data_size
                                                              options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> labels_buffer = [loss_ctx->device newBufferWithBytes:TF_TensorData(labels_tensor)
                                                                 length:data_size
                                                                options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* pred_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:pred_buffer
                                                                             shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                          dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* labels_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:labels_buffer
                                                                               shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                            dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{predTensor: pred_data, labelsTensor: labels_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:loss_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[neg]
                                         targetOperations:nil][neg];
    
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

}  // namespace mps
}  // namespace tensorflow
