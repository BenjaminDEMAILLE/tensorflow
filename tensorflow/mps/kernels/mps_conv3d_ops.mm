/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS 3D Convolution Operations
// Conv3D, MaxPool3D, AvgPool3D, Conv3DBackprop, Conv3DTranspose

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

struct MPSConv3DAttrs {
  int stride_d, stride_h, stride_w;
  int dilation_d, dilation_h, dilation_w;
  int padding;
};

void* MPSConv3D_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSConv3DAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "strides", &attrs->stride_d, s);
  attrs->stride_h = attrs->stride_d;
  attrs->stride_w = attrs->stride_d;
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "dilations", &attrs->dilation_d, s);
  attrs->dilation_h = attrs->dilation_d;
  attrs->dilation_w = attrs->dilation_d;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSConv3D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSConv3DAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  // Get input and filter tensors
  TF_Tensor* input = nullptr;
  TF_Tensor* filter = nullptr;
  
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &filter, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Input: [batch, depth, height, width, channels]
  int64_t batch = TF_Dim(input, 0);
  int64_t in_d = TF_Dim(input, 1);
  int64_t in_h = TF_Dim(input, 2);
  int64_t in_w = TF_Dim(input, 3);
  int64_t in_c = TF_Dim(input, 4);
  
  // Filter: [depth, height, width, in_channels, out_channels]
  int64_t f_d = TF_Dim(filter, 0);
  int64_t f_h = TF_Dim(filter, 1);
  int64_t f_w = TF_Dim(filter, 2);
  int64_t out_c = TF_Dim(filter, 4);
  
  // Calculate output dimensions
  int64_t out_d = (in_d - f_d) / attrs->stride_d + 1;
  int64_t out_h = (in_h - f_h) / attrs->stride_h + 1;
  int64_t out_w = (in_w - f_w) / attrs->stride_w + 1;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Create placeholders
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(in_d), @(in_h), @(in_w), @(in_c)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    MPSGraphTensor* filterTensor = [graph placeholderWithShape:@[@(f_d), @(f_h), @(f_w), @(in_c), @(out_c)]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"filter"];
    
    // Create Conv3D descriptor
    MPSGraphConvolution3DOpDescriptor* desc = [[MPSGraphConvolution3DOpDescriptor alloc] init];
    [desc setStrideInX:attrs->stride_w];
    [desc setStrideInY:attrs->stride_h];
    [desc setStrideInZ:attrs->stride_d];
    [desc setDilationRateInX:attrs->dilation_w];
    [desc setDilationRateInY:attrs->dilation_h];
    [desc setDilationRateInZ:attrs->dilation_d];
    [desc setPaddingStyle:MPSGraphPaddingStyleValid];
    [desc setDataLayout:MPSGraphTensorNamedDataLayoutNDHWC];
    [desc setWeightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
    
    // Perform convolution
    MPSGraphTensor* output = [graph convolution3DWithSourceTensor:inputTensor
                                                    weightsTensor:filterTensor
                                                       descriptor:desc
                                                             name:@"conv3d"];
    
    // Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Create buffers
    float* input_data = static_cast<float*>(TF_TensorData(input));
    float* filter_data = static_cast<float*>(TF_TensorData(filter));
    
    size_t input_bytes = batch * in_d * in_h * in_w * in_c * sizeof(float);
    size_t filter_bytes = f_d * f_h * f_w * in_c * out_c * sizeof(float);
    size_t output_bytes = batch * out_d * out_h * out_w * out_c * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> filterBuffer = [device newBufferWithBytes:filter_data
                                                     length:filter_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    // Create tensor data
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch), @(in_d), @(in_h), @(in_w), @(in_c)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* filterData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:filterBuffer
                    shape:@[@(f_d), @(f_h), @(f_w), @(in_c), @(out_c)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(batch), @(out_d), @(out_h), @(out_w), @(out_c)]
                 dataType:MPSDataTypeFloat32];
    
    // Execute
    NSDictionary* feeds = @{
      inputTensor: inputData,
      filterTensor: filterData
    };
    
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    // Allocate output
    int64_t out_dims[] = {batch, out_d, out_h, out_w, out_c};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 5, output_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    // Copy result
    float* out_data = static_cast<float*>(TF_TensorData(tf_output));
    memcpy(out_data, [outputBuffer contents], output_bytes);
  }
  
  TF_DeleteStatus(s);
}

void MPSConv3D_Delete(void* kernel) {
  delete static_cast<MPSConv3DAttrs*>(kernel);
}

// MaxPool3D Implementation
void* MPSMaxPool3D_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSConv3DAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "ksize", &attrs->stride_d, s);
  attrs->stride_h = attrs->stride_d;
  attrs->stride_w = attrs->stride_d;
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "strides", &attrs->dilation_d, s);
  attrs->dilation_h = attrs->dilation_d;
  attrs->dilation_w = attrs->dilation_d;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSMaxPool3D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSConv3DAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t batch = TF_Dim(input, 0);
  int64_t in_d = TF_Dim(input, 1);
  int64_t in_h = TF_Dim(input, 2);
  int64_t in_w = TF_Dim(input, 3);
  int64_t channels = TF_Dim(input, 4);
  
  int64_t out_d = in_d / attrs->dilation_d;
  int64_t out_h = in_h / attrs->dilation_h;
  int64_t out_w = in_w / attrs->dilation_w;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(in_d), @(in_h), @(in_w), @(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    MPSGraphPooling3DOpDescriptor* desc = [[MPSGraphPooling3DOpDescriptor alloc] init];
    [desc setKernelWidth:attrs->stride_w];
    [desc setKernelHeight:attrs->stride_h];
    [desc setKernelDepth:attrs->stride_d];
    [desc setStrideInX:attrs->dilation_w];
    [desc setStrideInY:attrs->dilation_h];
    [desc setStrideInZ:attrs->dilation_d];
    [desc setPaddingStyle:MPSGraphPaddingStyleValid];
    [desc setDataLayout:MPSGraphTensorNamedDataLayoutNDHWC];
    
    MPSGraphTensor* output = [graph maxPooling3DWithSourceTensor:inputTensor
                                                      descriptor:desc
                                                            name:@"maxpool3d"];
    
    // Execute
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* input_data = static_cast<float*>(TF_TensorData(input));
    size_t input_bytes = batch * in_d * in_h * in_w * channels * sizeof(float);
    size_t output_bytes = batch * out_d * out_h * out_w * channels * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch), @(in_d), @(in_h), @(in_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(batch), @(out_d), @(out_h), @(out_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: inputData};
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    int64_t out_dims[] = {batch, out_d, out_h, out_w, channels};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 5, output_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    float* out_data = static_cast<float*>(TF_TensorData(tf_output));
    memcpy(out_data, [outputBuffer contents], output_bytes);
  }
  
  TF_DeleteStatus(s);
}

void MPSMaxPool3D_Delete(void* kernel) {
  delete static_cast<MPSConv3DAttrs*>(kernel);
}

void RegisterConv3DOps(const char* platform_name, TF_Status* status) {
  // Conv3D
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Conv3D", platform_name,
                                                &MPSConv3D_Create,
                                                &MPSConv3D_Compute,
                                                &MPSConv3D_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSConv3D", kb, status);
  }
  
  // MaxPool3D
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MaxPool3D", platform_name,
                                                &MPSMaxPool3D_Create,
                                                &MPSMaxPool3D_Compute,
                                                &MPSMaxPool3D_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMaxPool3D", kb, status);
  }
  
  // TODO: AvgPool3D, Conv3DBackprop, Conv3DTranspose (18+ more ops)
}

}  // namespace mps
}  // namespace tensorflow
