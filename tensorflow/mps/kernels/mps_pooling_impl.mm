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

// REAL IMPLEMENTATION: Pooling operations with MPSGraph

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

enum PoolingType {
  MAX_POOL,
  AVG_POOL,
  AVG_POOL_INCLUDE_PADDING,
  GLOBAL_AVG_POOL,
  GLOBAL_MAX_POOL
};

struct MPSPoolContext {
  PoolingType type;
  
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> strides;
  std::string padding;
  std::string data_format;
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
};

extern "C" void* MPSPool_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSPoolContext();
  TF_Status* status = TF_NewStatus();
  
  // Get kernel size (ksize)
  int64_t* ksize = nullptr;
  int ksize_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "ksize", &ksize, &ksize_len, status);
  if (TF_GetCode(status) == TF_OK && ksize_len > 0) {
    kernel_ctx->kernel_size.assign(ksize, ksize + ksize_len);
  }
  
  // Get strides
  int64_t* strides = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides, &strides_len, status);
  if (TF_GetCode(status) == TF_OK && strides_len > 0) {
    kernel_ctx->strides.assign(strides, strides + strides_len);
  }
  
  // Get padding
  TF_StringView padding_view;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_view, status);
  if (TF_GetCode(status) == TF_OK) {
    kernel_ctx->padding = std::string(padding_view.data, padding_view.len);
  }
  
  // Get data format
  TF_StringView format_view;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_view, status);
  if (TF_GetCode(status) == TF_OK) {
    kernel_ctx->data_format = std::string(format_view.data, format_view.len);
  } else {
    kernel_ctx->data_format = "NHWC";
  }
  
  // Initialize Metal
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSMaxPool_Create(TF_OpKernelConstruction* ctx) {
  auto* pool_ctx = static_cast<MPSPoolContext*>(MPSPool_Create(ctx));
  pool_ctx->type = MAX_POOL;
  return pool_ctx;
}

extern "C" void MPSAvgPool_Create(TF_OpKernelConstruction* ctx) {
  auto* pool_ctx = static_cast<MPSPoolContext*>(MPSPool_Create(ctx));
  pool_ctx->type = AVG_POOL;
  return pool_ctx;
}

extern "C" void MPSPool_Delete(void* kernel) {
  auto* ctx = static_cast<MPSPoolContext*>(kernel);
  delete ctx;
}

static void ComputePooling(MPSPoolContext* pool_ctx, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    // Get input tensor
    TF_Tensor* input_tensor = nullptr;
    TF_GetInput(ctx, 0, &input_tensor, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get input shape (NHWC or NCHW)
    int64_t batch = TF_Dim(input_tensor, 0);
    int64_t height, width, channels;
    
    if (pool_ctx->data_format == "NHWC") {
      height = TF_Dim(input_tensor, 1);
      width = TF_Dim(input_tensor, 2);
      channels = TF_Dim(input_tensor, 3);
    } else {
      channels = TF_Dim(input_tensor, 1);
      height = TF_Dim(input_tensor, 2);
      width = TF_Dim(input_tensor, 3);
    }
    
    // Extract kernel size and strides (skip batch and channel dims)
    int64_t kernel_h = pool_ctx->kernel_size[1];
    int64_t kernel_w = pool_ctx->kernel_size[2];
    int64_t stride_h = pool_ctx->strides[1];
    int64_t stride_w = pool_ctx->strides[2];
    
    // Calculate output dimensions
    int64_t output_h, output_w;
    int64_t pad_top = 0, pad_left = 0;
    
    if (pool_ctx->padding == "VALID") {
      output_h = (height - kernel_h) / stride_h + 1;
      output_w = (width - kernel_w) / stride_w + 1;
    } else {  // SAME padding
      output_h = (height + stride_h - 1) / stride_h;
      output_w = (width + stride_w - 1) / stride_w;
      
      int64_t pad_h_total = std::max<int64_t>(0, (output_h - 1) * stride_h + kernel_h - height);
      int64_t pad_w_total = std::max<int64_t>(0, (output_w - 1) * stride_w + kernel_w - width);
      pad_top = pad_h_total / 2;
      pad_left = pad_w_total / 2;
    }
    
    // Create MPSGraph
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Create input placeholder
    MPSGraphTensor* inputTensor;
    if (pool_ctx->data_format == "NHWC") {
      inputTensor = [graph placeholderWithShape:@[@(batch), @(height), @(width), @(channels)]
                                       dataType:MPSDataTypeFloat32
                                           name:@"input"];
    } else {
      inputTensor = [graph placeholderWithShape:@[@(batch), @(channels), @(height), @(width)]
                                       dataType:MPSDataTypeFloat32
                                           name:@"input"];
    }
    
    // Create pooling descriptor
    MPSGraphPooling2DOpDescriptor* descriptor = [[MPSGraphPooling2DOpDescriptor alloc] init];
    descriptor.kernelHeight = kernel_h;
    descriptor.kernelWidth = kernel_w;
    descriptor.strideInX = stride_w;
    descriptor.strideInY = stride_h;
    descriptor.paddingLeft = pad_left;
    descriptor.paddingRight = (pool_ctx->padding == "SAME") ? 
        (pad_w_total - pad_left) : 0;
    descriptor.paddingTop = pad_top;
    descriptor.paddingBottom = (pool_ctx->padding == "SAME") ? 
        (pad_h_total - pad_top) : 0;
    descriptor.paddingStyle = (pool_ctx->padding == "SAME") ? 
        MPSGraphPaddingStyleExplicit : MPSGraphPaddingStyleValidOnly;
    descriptor.dataLayout = (pool_ctx->data_format == "NHWC") ? 
        MPSGraphTensorNamedDataLayoutNHWC : MPSGraphTensorNamedDataLayoutNCHW;
    
    // Create pooling operation
    MPSGraphTensor* outputTensor;
    if (pool_ctx->type == MAX_POOL) {
      outputTensor = [graph maxPooling2DWithSourceTensor:inputTensor
                                              descriptor:descriptor
                                                    name:@"max_pool"];
    } else {
      outputTensor = [graph avgPooling2DWithSourceTensor:inputTensor
                                              descriptor:descriptor
                                                    name:@"avg_pool"];
    }
    
    // Create Metal buffers
    size_t input_size = batch * height * width * channels * sizeof(float);
    size_t output_size = batch * output_h * output_w * channels * sizeof(float);
    
    id<MTLBuffer> input_buffer = [pool_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                                length:input_size
                                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> output_buffer = [pool_ctx->device newBufferWithLength:output_size
                                                                 options:MTLResourceStorageModeShared];
    
    // Create MPSGraphTensorData
    NSArray<NSNumber*>* input_shape;
    if (pool_ctx->data_format == "NHWC") {
      input_shape = @[@(batch), @(height), @(width), @(channels)];
    } else {
      input_shape = @[@(batch), @(channels), @(height), @(width)];
    }
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                             shape:input_shape
                                                                          dataType:MPSDataTypeFloat32];
    
    // Execute graph
    id<MTLCommandBuffer> commandBuffer = [pool_ctx->command_queue commandBuffer];
    
    MPSGraphTensorData* outputData = [graph runWithMTLCommandQueue:pool_ctx->command_queue
                                                        inputsArray:@[inputData]
                                                   inputsTargetTensors:@[inputTensor]
                                                     outputsTensors:@[outputTensor]
                                            targetOperationsArray:nil
                                                  executionDescriptor:nil]
                                             .firstObject;
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Allocate output tensor
    int64_t output_dims[4];
    if (pool_ctx->data_format == "NHWC") {
      output_dims[0] = batch;
      output_dims[1] = output_h;
      output_dims[2] = output_w;
      output_dims[3] = channels;
    } else {
      output_dims[0] = batch;
      output_dims[1] = channels;
      output_dims[2] = output_h;
      output_dims[3] = output_w;
    }
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 4, output_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Copy result back
    memcpy(TF_TensorData(output_tf), [[outputData mpsndarray] mpsBuffer].contents, output_size);
  }
  
  TF_DeleteStatus(status);
}

extern "C" void MPSMaxPool_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* pool_ctx = static_cast<MPSPoolContext*>(kernel);
  ComputePooling(pool_ctx, ctx);
}

extern "C" void MPSAvgPool_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* pool_ctx = static_cast<MPSPoolContext*>(kernel);
  ComputePooling(pool_ctx, ctx);
}

}  // namespace mps
}  // namespace tensorflow
