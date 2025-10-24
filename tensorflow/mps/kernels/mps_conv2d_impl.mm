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

// REAL IMPLEMENTATION: Conv2D with Metal Performance Shaders

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

// Metal device manager
class MPSDeviceManager {
 public:
  static MPSDeviceManager* GetInstance() {
    static MPSDeviceManager instance;
    return &instance;
  }
  
  id<MTLDevice> GetDevice() { return device_; }
  id<MTLCommandQueue> GetCommandQueue() { return command_queue_; }
  
 private:
  MPSDeviceManager() {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_) {
      command_queue_ = [device_ newCommandQueue];
    }
  }
  
  id<MTLDevice> device_;
  id<MTLCommandQueue> command_queue_;
};

// Conv2D kernel context
struct MPSConv2DContext {
  int64_t stride_h;
  int64_t stride_w;
  int64_t pad_h;
  int64_t pad_w;
  int64_t dilation_h;
  int64_t dilation_w;
  std::string padding;
  std::string data_format;
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  MPSGraph* graph;
};

extern "C" void* MPSConv2D_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSConv2DContext();
  TF_Status* status = TF_NewStatus();
  
  // Get attributes
  int64_t strides[4];
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "strides", strides, 4, status);
  kernel_ctx->stride_h = strides[1];
  kernel_ctx->stride_w = strides[2];
  
  char padding_buf[16];
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding_buf, 16, status);
  kernel_ctx->padding = std::string(padding_buf);
  
  char data_format_buf[16];
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", data_format_buf, 16, status);
  kernel_ctx->data_format = std::string(data_format_buf);
  
  int64_t dilations[4];
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "dilations", dilations, 4, status);
  kernel_ctx->dilation_h = dilations[1];
  kernel_ctx->dilation_w = dilations[2];
  
  // Initialize Metal
  auto* device_mgr = MPSDeviceManager::GetInstance();
  kernel_ctx->device = device_mgr->GetDevice();
  kernel_ctx->command_queue = device_mgr->GetCommandQueue();
  kernel_ctx->graph = [[MPSGraph alloc] init];
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSConv2D_Delete(void* kernel) {
  auto* ctx = static_cast<MPSConv2DContext*>(kernel);
  delete ctx;
}

extern "C" void MPSConv2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* conv_ctx = static_cast<MPSConv2DContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    // Get input tensors
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* filter_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    TF_GetInput(ctx, 1, &filter_tensor, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get dimensions
    int64_t batch = TF_Dim(input_tensor, 0);
    int64_t in_height = TF_Dim(input_tensor, 1);
    int64_t in_width = TF_Dim(input_tensor, 2);
    int64_t in_channels = TF_Dim(input_tensor, 3);
    
    int64_t filter_h = TF_Dim(filter_tensor, 0);
    int64_t filter_w = TF_Dim(filter_tensor, 1);
    int64_t out_channels = TF_Dim(filter_tensor, 3);
    
    // Calculate output dimensions
    int64_t out_height, out_width;
    if (conv_ctx->padding == "SAME") {
      out_height = (in_height + conv_ctx->stride_h - 1) / conv_ctx->stride_h;
      out_width = (in_width + conv_ctx->stride_w - 1) / conv_ctx->stride_w;
      
      int64_t pad_h_total = std::max<int64_t>(0, (out_height - 1) * conv_ctx->stride_h + filter_h - in_height);
      int64_t pad_w_total = std::max<int64_t>(0, (out_width - 1) * conv_ctx->stride_w + filter_w - in_width);
      conv_ctx->pad_h = pad_h_total / 2;
      conv_ctx->pad_w = pad_w_total / 2;
    } else {
      out_height = (in_height - filter_h) / conv_ctx->stride_h + 1;
      out_width = (in_width - filter_w) / conv_ctx->stride_w + 1;
      conv_ctx->pad_h = 0;
      conv_ctx->pad_w = 0;
    }
    
    // Create Metal buffers
    size_t input_size = batch * in_height * in_width * in_channels * sizeof(float);
    size_t filter_size = filter_h * filter_w * in_channels * out_channels * sizeof(float);
    size_t output_size = batch * out_height * out_width * out_channels * sizeof(float);
    
    id<MTLBuffer> input_buffer = [conv_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                                length:input_size
                                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> filter_buffer = [conv_ctx->device newBufferWithBytes:TF_TensorData(filter_tensor)
                                                                 length:filter_size
                                                                options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> output_buffer = [conv_ctx->device newBufferWithLength:output_size
                                                                 options:MTLResourceStorageModeShared];
    
    // Create MPSGraph computation
    MPSGraph* graph = conv_ctx->graph;
    
    // Input placeholder
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(in_height), @(in_width), @(in_channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Filter placeholder
    MPSGraphTensor* filterTensor = [graph placeholderWithShape:@[@(filter_h), @(filter_w), @(in_channels), @(out_channels)]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"filter"];
    
    // Convolution descriptor
    MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:conv_ctx->stride_w
                                                                                               strideInY:conv_ctx->stride_h
                                                                                         dilationRateInX:conv_ctx->dilation_w
                                                                                         dilationRateInY:conv_ctx->dilation_h
                                                                                                  groups:1
                                                                                            paddingLeft:conv_ctx->pad_w
                                                                                           paddingRight:conv_ctx->pad_w
                                                                                              paddingTop:conv_ctx->pad_h
                                                                                           paddingBottom:conv_ctx->pad_h
                                                                                            paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                              dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                                                                           weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    
    // Perform convolution
    MPSGraphTensor* outputTensor = [graph convolution2DWithSourceTensor:inputTensor
                                                          weightsTensor:filterTensor
                                                             descriptor:desc
                                                                   name:@"conv2d"];
    
    // Create command buffer and encode
    id<MTLCommandBuffer> commandBuffer = [conv_ctx->command_queue commandBuffer];
    
    // Create tensor data
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                             shape:@[@(batch), @(in_height), @(in_width), @(in_channels)]
                                                                          dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* filterData = [[MPSGraphTensorData alloc] initWithMTLBuffer:filter_buffer
                                                                              shape:@[@(filter_h), @(filter_w), @(in_channels), @(out_channels)]
                                                                           dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer
                                                                              shape:@[@(batch), @(out_height), @(out_width), @(out_channels)]
                                                                           dataType:MPSDataTypeFloat32];
    
    // Execute graph
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputTensor: inputData,
      filterTensor: filterData
    };
    
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputTensor: outputData
    };
    
    [graph encodeToCommandBuffer:commandBuffer
                            feeds:feeds
                   targetTensors:@[outputTensor]
              targetOperations:nil
                executionDescriptor:nil];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Allocate output tensor
    int64_t output_dims[4] = {batch, out_height, out_width, out_channels};
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 4, output_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Copy result back
    memcpy(TF_TensorData(output_tf), [output_buffer contents], output_size);
  }
  
  TF_DeleteStatus(status);
}

}  // namespace mps
}  // namespace tensorflow
