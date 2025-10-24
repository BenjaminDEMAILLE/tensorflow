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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

namespace {

struct MPSNNExtendedContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  MPSGraph* graph;
  
  MPSNNExtendedContext() {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
    graph = [[MPSGraph new] autorelease];
  }
  
  ~MPSNNExtendedContext() {
    [commandQueue release];
    [device release];
  }
};

static MPSNNExtendedContext* GetContext() {
  static MPSNNExtendedContext* ctx = nullptr;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    ctx = new MPSNNExtendedContext();
  });
  return ctx;
}

} // namespace

// ============================================================================
// CONVOLUTION GRADIENTS
// ============================================================================

// Conv2DBackpropInput
extern "C" void* MPSConv2DBackpropInput_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSConv2DBackpropInput_Delete(void* kernel) {}
extern "C" void MPSConv2DBackpropInput_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* mps_ctx = static_cast<MPSNNExtendedContext*>(kernel);
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    // Get inputs: input_sizes, filter, out_backprop
    TF_Tensor* input_sizes = nullptr;
    TF_Tensor* filter = nullptr;
    TF_Tensor* out_backprop = nullptr;
    
    TF_GetInput(ctx, 0, &input_sizes, status);
    if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
    TF_GetInput(ctx, 1, &filter, status);
    if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
    TF_GetInput(ctx, 2, &out_backprop, status);
    if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
    
    MPSGraph* graph = mps_ctx->graph;
    
    // Get dimensions
    const int32_t* input_size_data = (const int32_t*)TF_TensorData(input_sizes);
    int batch = input_size_data[0];
    int height = input_size_data[1];
    int width = input_size_data[2];
    int channels = input_size_data[3];
    
    // Filter: [filter_height, filter_width, in_channels, out_channels]
    int filter_h = TF_Dim(filter, 0);
    int filter_w = TF_Dim(filter, 1);
    int in_ch = TF_Dim(filter, 2);
    int out_ch = TF_Dim(filter, 3);
    
    // Create placeholders
    NSArray* filter_shape = @[@(filter_h), @(filter_w), @(in_ch), @(out_ch)];
    NSArray* grad_shape = @[@(TF_Dim(out_backprop, 0)), @(TF_Dim(out_backprop, 1)), 
                            @(TF_Dim(out_backprop, 2)), @(TF_Dim(out_backprop, 3))];
    
    MPSGraphTensor* filterTensor = [graph placeholderWithShape:filter_shape
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"filter"];
    MPSGraphTensor* gradTensor = [graph placeholderWithShape:grad_shape
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"grad"];
    
    // Conv2D gradient with respect to input
    MPSGraphConvolution2DOpDescriptor* desc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    [desc setStrideInX:1 strideInY:1];
    [desc setPaddingStyle:MPSGraphPaddingStyleTF_SAME];
    [desc setDataLayout:MPSGraphTensorNamedDataLayoutNHWC];
    [desc setWeightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    
    MPSGraphTensor* inputGrad = [graph convolution2DDataGradientWithIncomingGradientTensor:gradTensor
                                                                              weightsTensor:filterTensor
                                                                                outputShape:@[@(batch), @(height), @(width), @(channels)]
                                                                   forwardConvolutionDescriptor:desc
                                                                                           name:@"conv2d_input_grad"];
    
    // Execute
    int64_t filter_nelems = filter_h * filter_w * in_ch * out_ch;
    int64_t grad_nelems = 1;
    for (int i = 0; i < TF_NumDims(out_backprop); ++i) {
      grad_nelems *= TF_Dim(out_backprop, i);
    }
    
    float* filter_data = (float*)TF_TensorData(filter);
    float* grad_data = (float*)TF_TensorData(out_backprop);
    
    id<MTLBuffer> filterBuffer = [mps_ctx->device newBufferWithBytes:filter_data
                                                              length:filter_nelems * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> gradBuffer = [mps_ctx->device newBufferWithBytes:grad_data
                                                            length:grad_nelems * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* filterData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:filterBuffer shape:filter_shape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* gradData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:gradBuffer shape:grad_shape dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{filterTensor: filterData, gradTensor: gradData};
    NSDictionary* results = [graph runWithFeeds:feeds
                                 targetTensors:@[inputGrad]
                               targetOperations:nil
                            executionDescriptor:nil];
    
    MPSGraphTensorData* resultData = results[inputGrad];
    
    int64_t output_dims[4] = {batch, height, width, channels};
    int64_t output_nelems = batch * height * width * channels;
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 4,
                                         output_nelems * sizeof(float), status);
    if (TF_GetCode(status) == TF_OK) {
      float* output_data = (float*)TF_TensorData(output);
      id<MTLBuffer> resultBuffer = [[resultData mpsndarray] mpsndArrayBuffer];
      memcpy(output_data, [resultBuffer contents], output_nelems * sizeof(float));
    } else {
      TF_OpKernelContext_Failure(ctx, status);
    }
    
    [filterData release];
    [gradData release];
    [filterBuffer release];
    [gradBuffer release];
    [desc release];
    TF_DeleteStatus(status);
  }
}

// Conv2DBackpropFilter
extern "C" void* MPSConv2DBackpropFilter_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSConv2DBackpropFilter_Delete(void* kernel) {}
extern "C" void MPSConv2DBackpropFilter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* mps_ctx = static_cast<MPSNNExtendedContext*>(kernel);
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    // Get inputs: input, filter_sizes, out_backprop
    TF_Tensor* input = nullptr;
    TF_Tensor* filter_sizes = nullptr;
    TF_Tensor* out_backprop = nullptr;
    
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
    TF_GetInput(ctx, 1, &filter_sizes, status);
    if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
    TF_GetInput(ctx, 2, &out_backprop, status);
    if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
    
    MPSGraph* graph = mps_ctx->graph;
    
    NSArray* input_shape = @[@(TF_Dim(input, 0)), @(TF_Dim(input, 1)), 
                            @(TF_Dim(input, 2)), @(TF_Dim(input, 3))];
    NSArray* grad_shape = @[@(TF_Dim(out_backprop, 0)), @(TF_Dim(out_backprop, 1)), 
                           @(TF_Dim(out_backprop, 2)), @(TF_Dim(out_backprop, 3))];
    
    const int32_t* filter_size_data = (const int32_t*)TF_TensorData(filter_sizes);
    NSArray* filter_shape = @[@(filter_size_data[0]), @(filter_size_data[1]), 
                             @(filter_size_data[2]), @(filter_size_data[3])];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:input_shape
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    MPSGraphTensor* gradTensor = [graph placeholderWithShape:grad_shape
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"grad"];
    
    MPSGraphConvolution2DOpDescriptor* desc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    [desc setStrideInX:1 strideInY:1];
    [desc setPaddingStyle:MPSGraphPaddingStyleTF_SAME];
    [desc setDataLayout:MPSGraphTensorNamedDataLayoutNHWC];
    [desc setWeightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    
    MPSGraphTensor* filterGrad = [graph convolution2DWeightsGradientWithIncomingGradientTensor:gradTensor
                                                                                   sourceTensor:inputTensor
                                                                                    outputShape:filter_shape
                                                                   forwardConvolutionDescriptor:desc
                                                                                           name:@"conv2d_filter_grad"];
    
    // Execute
    int64_t input_nelems = 1;
    for (int i = 0; i < 4; ++i) input_nelems *= TF_Dim(input, i);
    
    int64_t grad_nelems = 1;
    for (int i = 0; i < 4; ++i) grad_nelems *= TF_Dim(out_backprop, i);
    
    float* input_data = (float*)TF_TensorData(input);
    float* grad_data = (float*)TF_TensorData(out_backprop);
    
    id<MTLBuffer> inputBuffer = [mps_ctx->device newBufferWithBytes:input_data
                                                             length:input_nelems * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> gradBuffer = [mps_ctx->device newBufferWithBytes:grad_data
                                                            length:grad_nelems * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer shape:input_shape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* gradData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:gradBuffer shape:grad_shape dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: inputData, gradTensor: gradData};
    NSDictionary* results = [graph runWithFeeds:feeds
                                 targetTensors:@[filterGrad]
                               targetOperations:nil
                            executionDescriptor:nil];
    
    MPSGraphTensorData* resultData = results[filterGrad];
    
    int64_t output_dims[4] = {filter_size_data[0], filter_size_data[1], 
                             filter_size_data[2], filter_size_data[3]};
    int64_t output_nelems = output_dims[0] * output_dims[1] * output_dims[2] * output_dims[3];
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 4,
                                         output_nelems * sizeof(float), status);
    if (TF_GetCode(status) == TF_OK) {
      float* output_data = (float*)TF_TensorData(output);
      id<MTLBuffer> resultBuffer = [[resultData mpsndarray] mpsndArrayBuffer];
      memcpy(output_data, [resultBuffer contents], output_nelems * sizeof(float));
    } else {
      TF_OpKernelContext_Failure(ctx, status);
    }
    
    [inputData release];
    [gradData release];
    [inputBuffer release];
    [gradBuffer release];
    [desc release];
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// 3D POOLING OPERATIONS
// ============================================================================

// MaxPool3D
extern "C" void* MPSMaxPool3D_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSMaxPool3D_Delete(void* kernel) {}
extern "C" void MPSMaxPool3D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MaxPool3D - MPSGraph doesn't have native 3D pooling, needs custom Metal kernel");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// MaxPool3DGrad
extern "C" void* MPSMaxPool3DGrad_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSMaxPool3DGrad_Delete(void* kernel) {}
extern "C" void MPSMaxPool3DGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MaxPool3DGrad requires forward pass indices");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// AvgPool3D
extern "C" void* MPSAvgPool3D_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSAvgPool3D_Delete(void* kernel) {}
extern "C" void MPSAvgPool3D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "AvgPool3D - needs custom 3D pooling Metal kernel");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// AvgPool3DGrad
extern "C" void* MPSAvgPool3DGrad_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSAvgPool3DGrad_Delete(void* kernel) {}
extern "C" void MPSAvgPool3DGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "AvgPool3DGrad requires gradient backprop");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ============================================================================
// BATCH NORM GRADIENTS
// ============================================================================

// FusedBatchNormGrad
extern "C" void* MPSFusedBatchNormGrad_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSFusedBatchNormGrad_Delete(void* kernel) {}
extern "C" void MPSFusedBatchNormGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FusedBatchNormGrad - complex multi-output gradient operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// FusedBatchNormGradV2
extern "C" void* MPSFusedBatchNormGradV2_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSFusedBatchNormGradV2_Delete(void* kernel) {}
extern "C" void MPSFusedBatchNormGradV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  MPSFusedBatchNormGrad_Compute(kernel, ctx);
}

// FusedBatchNormGradV3
extern "C" void* MPSFusedBatchNormGradV3_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSFusedBatchNormGradV3_Delete(void* kernel) {}
extern "C" void MPSFusedBatchNormGradV3_Compute(void* kernel, TF_OpKernelContext* ctx) {
  MPSFusedBatchNormGrad_Compute(kernel, ctx);
}

// ============================================================================
// SPACE TRANSFORMATION OPERATIONS
// ============================================================================

// SpaceToBatchND
extern "C" void* MPSSpaceToBatchND_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSSpaceToBatchND_Delete(void* kernel) {}
extern "C" void MPSSpaceToBatchND_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* mps_ctx = static_cast<MPSNNExtendedContext*>(kernel);
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    MPSGraph* graph = mps_ctx->graph;
    
    int nd = TF_NumDims(input);
    NSMutableArray* input_shape = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [input_shape addObject:@(TF_Dim(input, i))];
    }
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:input_shape
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // SpaceToBatch: block_size = 2 (example)
    MPSGraphTensor* outputTensor = [graph spaceToBatchTensor:inputTensor
                                                 spatialAxes:@[@1, @2]
                                             batchAxis:0
                                            blockDimensions:@[@2, @2]
                                            usePixelShuffleOrder:NO
                                                        name:@"space_to_batch"];
    
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) nelems *= TF_Dim(input, i);
    
    float* input_data = (float*)TF_TensorData(input);
    id<MTLBuffer> inputBuffer = [mps_ctx->device newBufferWithBytes:input_data
                                                             length:nelems * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer shape:input_shape dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: inputData};
    NSDictionary* results = [graph runWithFeeds:feeds
                                 targetTensors:@[outputTensor]
                               targetOperations:nil
                            executionDescriptor:nil];
    
    MPSGraphTensorData* resultData = results[outputTensor];
    NSArray* outputShape = [resultData shape];
    int output_nd = [outputShape count];
    int64_t output_dims[8];
    int64_t output_nelems = 1;
    for (int i = 0; i < output_nd; ++i) {
      output_dims[i] = [[outputShape objectAtIndex:i] longLongValue];
      output_nelems *= output_dims[i];
    }
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, output_nd,
                                         output_nelems * sizeof(float), status);
    if (TF_GetCode(status) == TF_OK) {
      float* output_data = (float*)TF_TensorData(output);
      id<MTLBuffer> resultBuffer = [[resultData mpsndarray] mpsndArrayBuffer];
      memcpy(output_data, [resultBuffer contents], output_nelems * sizeof(float));
    } else {
      TF_OpKernelContext_Failure(ctx, status);
    }
    
    [inputData release];
    [inputBuffer release];
    TF_DeleteStatus(status);
  }
}

// BatchToSpaceND
extern "C" void* MPSBatchToSpaceND_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSBatchToSpaceND_Delete(void* kernel) {}
extern "C" void MPSBatchToSpaceND_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* mps_ctx = static_cast<MPSNNExtendedContext*>(kernel);
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    MPSGraph* graph = mps_ctx->graph;
    
    int nd = TF_NumDims(input);
    NSMutableArray* input_shape = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [input_shape addObject:@(TF_Dim(input, i))];
    }
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:input_shape
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    MPSGraphTensor* outputTensor = [graph batchToSpaceTensor:inputTensor
                                                 spatialAxes:@[@1, @2]
                                                   batchAxis:0
                                             blockDimensions:@[@2, @2]
                                       usePixelShuffleOrder:NO
                                                        name:@"batch_to_space"];
    
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) nelems *= TF_Dim(input, i);
    
    float* input_data = (float*)TF_TensorData(input);
    id<MTLBuffer> inputBuffer = [mps_ctx->device newBufferWithBytes:input_data
                                                             length:nelems * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer shape:input_shape dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: inputData};
    NSDictionary* results = [graph runWithFeeds:feeds
                                 targetTensors:@[outputTensor]
                               targetOperations:nil
                            executionDescriptor:nil];
    
    MPSGraphTensorData* resultData = results[outputTensor];
    NSArray* outputShape = [resultData shape];
    int output_nd = [outputShape count];
    int64_t output_dims[8];
    int64_t output_nelems = 1;
    for (int i = 0; i < output_nd; ++i) {
      output_dims[i] = [[outputShape objectAtIndex:i] longLongValue];
      output_nelems *= output_dims[i];
    }
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, output_nd,
                                         output_nelems * sizeof(float), status);
    if (TF_GetCode(status) == TF_OK) {
      float* output_data = (float*)TF_TensorData(output);
      id<MTLBuffer> resultBuffer = [[resultData mpsndarray] mpsndArrayBuffer];
      memcpy(output_data, [resultBuffer contents], output_nelems * sizeof(float));
    } else {
      TF_OpKernelContext_Failure(ctx, status);
    }
    
    [inputData release];
    [inputBuffer release];
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// MORPHOLOGICAL OPERATIONS - Metal compute shaders
// ============================================================================

namespace {
struct MPSMorphAttrs {
  std::vector<int64_t> strides;
  std::vector<int64_t> rates;
  std::string padding;
  id<MTLLibrary> library;
  id<MTLComputePipelineState> dilation_fwd;
  id<MTLComputePipelineState> dilation_grad_input;
  id<MTLComputePipelineState> dilation_grad_filter;
  id<MTLComputePipelineState> erosion_fwd;
};

struct MorphologyParams {
  int32_t N, H, W, C;
  int32_t H_out, W_out;
  int32_t kH, kW;
  int32_t stride_h, stride_w;
  int32_t rate_h, rate_w;
  int32_t pad_top, pad_left;
};
}

extern "C" void* MPSDilation2D_Create(TF_OpKernelConstruction* ctx) {
  auto* a = new MPSMorphAttrs();
  TF_Status* s = TF_NewStatus();
  int64_t* strides_data=nullptr; int strides_len=0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (strides_data && strides_len>0) a->strides.assign(strides_data, strides_data+strides_len);
  
  int64_t* rates_data=nullptr; int rates_len=0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "rates", &rates_data, &rates_len, s);
  if (rates_data && rates_len>0) a->rates.assign(rates_data, rates_data+rates_len);
  
  char* padding_data=nullptr; size_t padding_len=0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (padding_data) a->padding.assign(padding_data, padding_len);
  
  // Compile Metal shaders
  auto* mpsCtx = GetContext();
  NSError* error = nil;
  NSString* shaderPath = @"tensorflow/mps/kernels/mps_morphology.metal";
  NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
  
  if (!shaderSource) {
    // Fallback: embedded shader source
    shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;
struct MorphologyParams { int32_t N, H, W, C, H_out, W_out, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left; };
kernel void dilation2d_forward(device const float* input [[buffer(0)]], device const float* filter [[buffer(1)]], device float* output [[buffer(2)]], constant MorphologyParams& p [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
  int n=gid.z, oh=gid.y, ow=gid.x;
  if (n>=p.N || oh>=p.H_out || ow>=p.W_out) return;
  int h_start = oh*p.stride_h - p.pad_top, w_start = ow*p.stride_w - p.pad_left;
  for (int c=0; c<p.C; ++c) {
    float m = -INFINITY;
    for (int kh=0; kh<p.kH; ++kh) {
      int h_in = h_start + kh*p.rate_h; if (h_in<0 || h_in>=p.H) continue;
      for (int kw=0; kw<p.kW; ++kw) {
        int w_in = w_start + kw*p.rate_w; if (w_in<0 || w_in>=p.W) continue;
        int inp_idx = ((n*p.H + h_in)*p.W + w_in)*p.C + c;
        int flt_idx = (kh*p.kW + kw)*p.C + c;
        m = max(m, input[inp_idx] + filter[flt_idx]);
      }
    }
    output[((n*p.H_out + oh)*p.W_out + ow)*p.C + c] = m;
  }
}
kernel void erosion2d_forward(device const float* input [[buffer(0)]], device const float* filter [[buffer(1)]], device float* output [[buffer(2)]], constant MorphologyParams& p [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
  int n=gid.z, oh=gid.y, ow=gid.x;
  if (n>=p.N || oh>=p.H_out || ow>=p.W_out) return;
  int h_start = oh*p.stride_h - p.pad_top, w_start = ow*p.stride_w - p.pad_left;
  for (int c=0; c<p.C; ++c) {
    float m = INFINITY;
    for (int kh=0; kh<p.kH; ++kh) {
      int h_in = h_start + kh*p.rate_h; if (h_in<0 || h_in>=p.H) continue;
      for (int kw=0; kw<p.kW; ++kw) {
        int w_in = w_start + kw*p.rate_w; if (w_in<0 || w_in>=p.W) continue;
        int inp_idx = ((n*p.H + h_in)*p.W + w_in)*p.C + c;
        int flt_idx = (kh*p.kW + kw)*p.C + c;
        m = min(m, input[inp_idx] - filter[flt_idx]);
      }
    }
    output[((n*p.H_out + oh)*p.W_out + ow)*p.C + c] = m;
  }
}
kernel void dilation2d_backprop_input(device const float* input [[buffer(0)]], device const float* filter [[buffer(1)]], device const float* dY [[buffer(2)]], device float* dX [[buffer(3)]], constant MorphologyParams& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
  int n=gid.z, oh=gid.y, ow=gid.x;
  if (n>=p.N || oh>=p.H_out || ow>=p.W_out) return;
  int h_start = oh*p.stride_h - p.pad_top, w_start = ow*p.stride_w - p.pad_left;
  const float eps = 1e-6f;
  for (int c=0; c<p.C; ++c) {
    float m = -INFINITY;
    for (int kh=0; kh<p.kH; ++kh) {
      int h_in = h_start + kh*p.rate_h; if (h_in<0 || h_in>=p.H) continue;
      for (int kw=0; kw<p.kW; ++kw) {
        int w_in = w_start + kw*p.rate_w; if (w_in<0 || w_in>=p.W) continue;
        int inp_idx = ((n*p.H + h_in)*p.W + w_in)*p.C + c;
        int flt_idx = (kh*p.kW + kw)*p.C + c;
        m = max(m, input[inp_idx] + filter[flt_idx]);
      }
    }
    float g = dY[((n*p.H_out + oh)*p.W_out + ow)*p.C + c];
    for (int kh=0; kh<p.kH; ++kh) {
      int h_in = h_start + kh*p.rate_h; if (h_in<0 || h_in>=p.H) continue;
      for (int kw=0; kw<p.kW; ++kw) {
        int w_in = w_start + kw*p.rate_w; if (w_in<0 || w_in>=p.W) continue;
        int inp_idx = ((n*p.H + h_in)*p.W + w_in)*p.C + c;
        int flt_idx = (kh*p.kW + kw)*p.C + c;
        if (fabs(input[inp_idx] + filter[flt_idx] - m) <= eps) atomic_fetch_add_explicit((device atomic<float>*)&dX[inp_idx], g, memory_order_relaxed);
      }
    }
  }
}
kernel void dilation2d_backprop_filter(device const float* input [[buffer(0)]], device const float* filter [[buffer(1)]], device const float* dY [[buffer(2)]], device float* dF [[buffer(3)]], constant MorphologyParams& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
  int n=gid.z, oh=gid.y, ow=gid.x;
  if (n>=p.N || oh>=p.H_out || ow>=p.W_out) return;
  int h_start = oh*p.stride_h - p.pad_top, w_start = ow*p.stride_w - p.pad_left;
  const float eps = 1e-6f;
  for (int c=0; c<p.C; ++c) {
    float m = -INFINITY;
    for (int kh=0; kh<p.kH; ++kh) {
      int h_in = h_start + kh*p.rate_h; if (h_in<0 || h_in>=p.H) continue;
      for (int kw=0; kw<p.kW; ++kw) {
        int w_in = w_start + kw*p.rate_w; if (w_in<0 || w_in>=p.W) continue;
        int inp_idx = ((n*p.H + h_in)*p.W + w_in)*p.C + c;
        int flt_idx = (kh*p.kW + kw)*p.C + c;
        m = max(m, input[inp_idx] + filter[flt_idx]);
      }
    }
    float g = dY[((n*p.H_out + oh)*p.W_out + ow)*p.C + c];
    for (int kh=0; kh<p.kH; ++kh) {
      int h_in = h_start + kh*p.rate_h; if (h_in<0 || h_in>=p.H) continue;
      for (int kw=0; kw<p.kW; ++kw) {
        int w_in = w_start + kw*p.rate_w; if (w_in<0 || w_in>=p.W) continue;
        int inp_idx = ((n*p.H + h_in)*p.W + w_in)*p.C + c;
        int flt_idx = (kh*p.kW + kw)*p.C + c;
        if (fabs(input[inp_idx] + filter[flt_idx] - m) <= eps) atomic_fetch_add_explicit((device atomic<float>*)&dF[flt_idx], g, memory_order_relaxed);
      }
    }
  }
}
    )";
  }
  
  a->library = [mpsCtx->device newLibraryWithSource:shaderSource options:nil error:&error];
  if (!a->library) { TF_DeleteStatus(s); return a; }
  
  id<MTLFunction> dil_fwd_func = [a->library newFunctionWithName:@"dilation2d_forward"];
  id<MTLFunction> dil_grad_in_func = [a->library newFunctionWithName:@"dilation2d_backprop_input"];
  id<MTLFunction> dil_grad_flt_func = [a->library newFunctionWithName:@"dilation2d_backprop_filter"];
  id<MTLFunction> ero_fwd_func = [a->library newFunctionWithName:@"erosion2d_forward"];
  
  a->dilation_fwd = [mpsCtx->device newComputePipelineStateWithFunction:dil_fwd_func error:&error];
  a->dilation_grad_input = [mpsCtx->device newComputePipelineStateWithFunction:dil_grad_in_func error:&error];
  a->dilation_grad_filter = [mpsCtx->device newComputePipelineStateWithFunction:dil_grad_flt_func error:&error];
  a->erosion_fwd = [mpsCtx->device newComputePipelineStateWithFunction:ero_fwd_func error:&error];
  
  TF_DeleteStatus(s);
  return a;
}

extern "C" void MPSDilation2D_Delete(void* kernel) {
  auto* a = reinterpret_cast<MPSMorphAttrs*>(kernel);
  if (a->library) [a->library release];
  if (a->dilation_fwd) [a->dilation_fwd release];
  if (a->dilation_grad_input) [a->dilation_grad_input release];
  if (a->dilation_grad_filter) [a->dilation_grad_filter release];
  if (a->erosion_fwd) [a->erosion_fwd release];
  delete a;
}
extern "C" void MPSDilation2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* a = reinterpret_cast<MPSMorphAttrs*>(kernel);
  if (!a->dilation_fwd) {
    TF_Status* s = TF_NewStatus();
    TF_SetStatus(s, TF_INTERNAL, "Dilation2D Metal shaders not compiled");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input=nullptr; TF_Tensor* filter=nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &filter, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT || TF_TensorType(filter) != TF_FLOAT) { TF_SetStatus(s, TF_UNIMPLEMENTED, "Dilation2D Metal supports float only"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_NumDims(input)!=4 || TF_NumDims(filter)!=3) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Dilation2D expects input [N,H,W,C] and filter [kH,kW,C]"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t N=TF_Dim(input,0), H=TF_Dim(input,1), W=TF_Dim(input,2), C=TF_Dim(input,3);
  int64_t kH=TF_Dim(filter,0), kW=TF_Dim(filter,1), Cf=TF_Dim(filter,2);
  if (Cf != C) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Dilation2D filter channels must match input"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t stride_h = (a->strides.size()>=4) ? a->strides[1] : 1;
  int64_t stride_w = (a->strides.size()>=4) ? a->strides[2] : 1;
  int64_t rate_h   = (a->rates.size()>=4)   ? a->rates[1]   : 1;
  int64_t rate_w   = (a->rates.size()>=4)   ? a->rates[2]   : 1;
  int64_t eff_kH = (kH-1)*rate_h + 1, eff_kW = (kW-1)*rate_w + 1;
  
  bool same = (a->padding == "SAME");
  int64_t H_out = same ? ((H + stride_h - 1) / stride_h) : ((H >= eff_kH) ? ((H - eff_kH)/stride_h + 1) : 0);
  int64_t W_out = same ? ((W + stride_w - 1) / stride_w) : ((W >= eff_kW) ? ((W - eff_kW)/stride_w + 1) : 0);
  int64_t pad_h = std::max<int64_t>(0, (H_out - 1)*stride_h + eff_kH - H);
  int64_t pad_w = std::max<int64_t>(0, (W_out - 1)*stride_w + eff_kW - W);
  int64_t pad_top = same ? pad_h/2 : 0, pad_left = same ? pad_w/2 : 0;

  int64_t out_dims[4] = {N, H_out, W_out, C};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, N*H_out*W_out*C*sizeof(float), s);
  if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  auto* mpsCtx = GetContext();
  @autoreleasepool {
    size_t input_bytes = N*H*W*C*sizeof(float), filter_bytes = kH*kW*C*sizeof(float), output_bytes = N*H_out*W_out*C*sizeof(float);
    id<MTLBuffer> inputBuf  = [mpsCtx->device newBufferWithBytes:TF_TensorData(input) length:input_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> filterBuf = [mpsCtx->device newBufferWithBytes:TF_TensorData(filter) length:filter_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuf = [mpsCtx->device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    
    MorphologyParams params = { (int32_t)N, (int32_t)H, (int32_t)W, (int32_t)C, (int32_t)H_out, (int32_t)W_out,
                                (int32_t)kH, (int32_t)kW, (int32_t)stride_h, (int32_t)stride_w,
                                (int32_t)rate_h, (int32_t)rate_w, (int32_t)pad_top, (int32_t)pad_left };
    id<MTLBuffer> paramsBuf = [mpsCtx->device newBufferWithBytes:&params length:sizeof(MorphologyParams) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> cb = [mpsCtx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    [encoder setComputePipelineState:a->dilation_fwd];
    [encoder setBuffer:inputBuf offset:0 atIndex:0];
    [encoder setBuffer:filterBuf offset:0 atIndex:1];
    [encoder setBuffer:outputBuf offset:0 atIndex:2];
    [encoder setBuffer:paramsBuf offset:0 atIndex:3];
    
    MTLSize gridSize = MTLSizeMake(W_out, H_out, N);
    NSUInteger w = a->dilation_fwd.threadExecutionWidth;
    MTLSize threadGroupSize = MTLSizeMake(std::min((NSUInteger)W_out, w), 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuf.contents, output_bytes);
    [inputBuf release]; [filterBuf release]; [outputBuf release]; [paramsBuf release];
  }
  TF_DeleteStatus(s);
}

// Dilation2DBackpropInput - CPU fallback (NHWC, float)
extern "C" void* MPSDilation2DBackpropInput_Create(TF_OpKernelConstruction* ctx) { return MPSDilation2D_Create(ctx); }
extern "C" void MPSDilation2DBackpropInput_Delete(void* kernel) { MPSDilation2D_Delete(kernel); }
extern "C" void MPSDilation2DBackpropInput_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* a = reinterpret_cast<MPSMorphAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  // Inputs: input (X), filter (F), out_backprop (dY)
  TF_Tensor* X_t=nullptr; TF_Tensor* F_t=nullptr; TF_Tensor* dY_t=nullptr;
  TF_GetInput(ctx, 0, &X_t, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &F_t, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &dY_t, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(X_t)!=TF_FLOAT || TF_TensorType(F_t)!=TF_FLOAT || TF_TensorType(dY_t)!=TF_FLOAT) { TF_SetStatus(s, TF_UNIMPLEMENTED, "Dilation2DBackpropInput supports float only"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int64_t N=TF_Dim(X_t,0), H=TF_Dim(X_t,1), W=TF_Dim(X_t,2), C=TF_Dim(X_t,3);
  int64_t kH=TF_Dim(F_t,0), kW=TF_Dim(F_t,1), Cf=TF_Dim(F_t,2);
  if (Cf!=C) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Dilation2DBackpropInput: filter channels must match input"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int64_t rate_h=(a->rates.size()>=4)?a->rates[1]:1, rate_w=(a->rates.size()>=4)?a->rates[2]:1;
  int64_t stride_h=(a->strides.size()>=4)?a->strides[1]:1, stride_w=(a->strides.size()>=4)?a->strides[2]:1;
  int64_t eff_kH=(kH-1)*rate_h+1, eff_kW=(kW-1)*rate_w+1;
  bool same = (a->padding=="SAME");
  int64_t H_out = same ? ((H + stride_h - 1)/stride_h) : ((H>=eff_kH)?((H-eff_kH)/stride_h + 1):0);
  int64_t W_out = same ? ((W + stride_w - 1)/stride_w) : ((W>=eff_kW)?((W-eff_kW)/stride_w + 1):0);
  int64_t pad_h = std::max<int64_t>(0, (H_out-1)*stride_h + eff_kH - H);
  int64_t pad_w = std::max<int64_t>(0, (W_out-1)*stride_w + eff_kW - W);
  int64_t pad_top = same ? pad_h/2 : 0; int64_t pad_left = same ? pad_w/2 : 0;

  int64_t out_dims[4] = {N,H,W,C}; int64_t out_elems = N*H*W*C;
  TF_Tensor* dX_t = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, out_elems*sizeof(float), s);
  if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const float* X = (const float*)TF_TensorData(X_t); const float* F = (const float*)TF_TensorData(F_t); const float* dY = (const float*)TF_TensorData(dY_t); float* dX = (float*)TF_TensorData(dX_t);
  std::fill(dX, dX + out_elems, 0.0f);
  auto idx_in = [H,W,C](int64_t n,int64_t h,int64_t w,int64_t c){ return ((n*H + h)*W + w)*C + c; };
  auto idx_f  = [kW,C](int64_t kh,int64_t kw,int64_t c){ return (kh*kW + kw)*C + c; };

  const float eps = 1e-6f;
  for (int64_t n=0; n<N; ++n) {
    for (int64_t oh=0; oh<H_out; ++oh) {
      int64_t h_start = oh*stride_h - pad_top;
      for (int64_t ow=0; ow<W_out; ++ow) {
        int64_t w_start = ow*stride_w - pad_left;
        for (int64_t c=0; c<C; ++c) {
          // find max value in window
          float m = -std::numeric_limits<float>::infinity();
          for (int64_t kh=0; kh<kH; ++kh) {
            int64_t h_in = h_start + kh*rate_h; if (h_in<0 || h_in>=H) continue;
            for (int64_t kw=0; kw<kW; ++kw) {
              int64_t w_in = w_start + kw*rate_w; if (w_in<0 || w_in>=W) continue;
              float val = X[idx_in(n,h_in,w_in,c)] + F[idx_f(kh,kw,c)];
              if (val > m) m = val;
            }
          }
          float g = dY[ ((n*H_out + oh)*W_out + ow)*C + c ];
          if (g == 0.0f) continue;
          for (int64_t kh=0; kh<kH; ++kh) {
            int64_t h_in = h_start + kh*rate_h; if (h_in<0 || h_in>=H) continue;
            for (int64_t kw=0; kw<kW; ++kw) {
              int64_t w_in = w_start + kw*rate_w; if (w_in<0 || w_in>=W) continue;
              float val = X[idx_in(n,h_in,w_in,c)] + F[idx_f(kh,kw,c)];
              if (std::fabs(val - m) <= eps) {
                dX[idx_in(n,h_in,w_in,c)] += g;
              }
            }
          }
        }
      }
    }
  }
  TF_DeleteStatus(s);
}

// Dilation2DBackpropFilter - CPU fallback (NHWC, float)
extern "C" void* MPSDilation2DBackpropFilter_Create(TF_OpKernelConstruction* ctx) { return MPSDilation2D_Create(ctx); }
extern "C" void MPSDilation2DBackpropFilter_Delete(void* kernel) { MPSDilation2D_Delete(kernel); }
extern "C" void MPSDilation2DBackpropFilter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* a = reinterpret_cast<MPSMorphAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  // Inputs: input (X), filter (F), out_backprop (dY)
  TF_Tensor* X_t=nullptr; TF_Tensor* F_t=nullptr; TF_Tensor* dY_t=nullptr;
  TF_GetInput(ctx, 0, &X_t, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &F_t, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &dY_t, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(X_t)!=TF_FLOAT || TF_TensorType(F_t)!=TF_FLOAT || TF_TensorType(dY_t)!=TF_FLOAT) { TF_SetStatus(s, TF_UNIMPLEMENTED, "Dilation2DBackpropFilter supports float only"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int64_t N=TF_Dim(X_t,0), H=TF_Dim(X_t,1), W=TF_Dim(X_t,2), C=TF_Dim(X_t,3);
  int64_t kH=TF_Dim(F_t,0), kW=TF_Dim(F_t,1), Cf=TF_Dim(F_t,2);
  if (Cf!=C) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Dilation2DBackpropFilter: filter channels must match input"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int64_t rate_h=(a->rates.size()>=4)?a->rates[1]:1, rate_w=(a->rates.size()>=4)?a->rates[2]:1;
  int64_t stride_h=(a->strides.size()>=4)?a->strides[1]:1, stride_w=(a->strides.size()>=4)?a->strides[2]:1;
  int64_t eff_kH=(kH-1)*rate_h+1, eff_kW=(kW-1)*rate_w+1;
  bool same = (a->padding=="SAME");
  int64_t H_out = same ? ((H + stride_h - 1)/stride_h) : ((H>=eff_kH)?((H-eff_kH)/stride_h + 1):0);
  int64_t W_out = same ? ((W + stride_w - 1)/stride_w) : ((W>=eff_kW)?((W-eff_kW)/stride_w + 1):0);
  int64_t pad_h = std::max<int64_t>(0, (H_out-1)*stride_h + eff_kH - H);
  int64_t pad_w = std::max<int64_t>(0, (W_out-1)*stride_w + eff_kW - W);
  int64_t pad_top = same ? pad_h/2 : 0; int64_t pad_left = same ? pad_w/2 : 0;

  int64_t out_dims[3] = {kH,kW,C}; int64_t out_elems = kH*kW*C;
  TF_Tensor* dF_t = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 3, out_elems*sizeof(float), s);
  if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const float* X = (const float*)TF_TensorData(X_t); const float* F = (const float*)TF_TensorData(F_t); const float* dY = (const float*)TF_TensorData(dY_t); float* dF = (float*)TF_TensorData(dF_t);
  std::fill(dF, dF + out_elems, 0.0f);
  auto idx_in = [H,W,C](int64_t n,int64_t h,int64_t w,int64_t c){ return ((n*H + h)*W + w)*C + c; };
  auto idx_f  = [kW,C](int64_t kh,int64_t kw,int64_t c){ return (kh*kW + kw)*C + c; };
  const float eps = 1e-6f;
  for (int64_t n=0; n<N; ++n) {
    for (int64_t oh=0; oh<H_out; ++oh) {
      int64_t h_start = oh*stride_h - pad_top;
      for (int64_t ow=0; ow<W_out; ++ow) {
        int64_t w_start = ow*stride_w - pad_left;
        for (int64_t c=0; c<C; ++c) {
          float m = -std::numeric_limits<float>::infinity();
          for (int64_t kh=0; kh<kH; ++kh) {
            int64_t h_in = h_start + kh*rate_h; if (h_in<0 || h_in>=H) continue;
            for (int64_t kw=0; kw<kW; ++kw) {
              int64_t w_in = w_start + kw*rate_w; if (w_in<0 || w_in>=W) continue;
              float val = X[idx_in(n,h_in,w_in,c)] + F[idx_f(kh,kw,c)];
              if (val > m) m = val;
            }
          }
          float g = dY[ ((n*H_out + oh)*W_out + ow)*C + c ];
          if (g == 0.0f) continue;
          for (int64_t kh=0; kh<kH; ++kh) {
            int64_t h_in = h_start + kh*rate_h; if (h_in<0 || h_in>=H) continue;
            for (int64_t kw=0; kw<kW; ++kw) {
              int64_t w_in = w_start + kw*rate_w; if (w_in<0 || w_in>=W) continue;
              float val = X[idx_in(n,h_in,w_in,c)] + F[idx_f(kh,kw,c)];
              if (std::fabs(val - m) <= eps) {
                dF[idx_f(kh,kw,c)] += g;
              }
            }
          }
        }
      }
    }
  }
  TF_DeleteStatus(s);
}

// Erosion2D - CPU fallback (NHWC, float)
extern "C" void* MPSErosion2D_Create(TF_OpKernelConstruction* ctx) { return MPSDilation2D_Create(ctx); }
extern "C" void MPSErosion2D_Delete(void* kernel) { MPSDilation2D_Delete(kernel); }
extern "C" void MPSErosion2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* a = reinterpret_cast<MPSMorphAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input=nullptr; TF_Tensor* filter=nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &filter, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT || TF_TensorType(filter) != TF_FLOAT) { TF_SetStatus(s, TF_UNIMPLEMENTED, "Erosion2D CPU fallback supports float only"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_NumDims(input)!=4 || TF_NumDims(filter)!=3) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Erosion2D expects input [N,H,W,C] and filter [kH,kW,C]"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int64_t N=TF_Dim(input,0), H=TF_Dim(input,1), W=TF_Dim(input,2), C=TF_Dim(input,3);
  int64_t kH=TF_Dim(filter,0), kW=TF_Dim(filter,1), Cf=TF_Dim(filter,2);
  if (Cf != C) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Erosion2D filter channels must match input channels"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int64_t stride_h = (a->strides.size()>=4) ? a->strides[1] : 1; int64_t stride_w = (a->strides.size()>=4) ? a->strides[2] : 1;
  int64_t rate_h   = (a->rates.size()>=4)   ? a->rates[1]   : 1; int64_t rate_w   = (a->rates.size()>=4)   ? a->rates[2]   : 1;
  int64_t eff_kH = (kH-1)*rate_h + 1; int64_t eff_kW = (kW-1)*rate_w + 1;
  auto same = (a->padding == "SAME");
  int64_t H_out = same ? ( (H + stride_h - 1) / stride_h ) : ( (H >= eff_kH) ? ( (H - eff_kH)/stride_h + 1 ) : 0 );
  int64_t W_out = same ? ( (W + stride_w - 1) / stride_w ) : ( (W >= eff_kW) ? ( (W - eff_kW)/stride_w + 1 ) : 0 );
  int64_t pad_h = std::max<int64_t>(0, (H_out - 1)*stride_h + eff_kH - H);
  int64_t pad_w = std::max<int64_t>(0, (W_out - 1)*stride_w + eff_kW - W);
  int64_t pad_top = same ? pad_h/2 : 0; int64_t pad_left = same ? pad_w/2 : 0;

  int64_t out_dims[4] = {N, H_out, W_out, C}; int64_t out_elems = N*H_out*W_out*C;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, out_elems*sizeof(float), s);
  if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const float* X = (const float*)TF_TensorData(input); const float* F = (const float*)TF_TensorData(filter); float* Y = (float*)TF_TensorData(output);

  auto idx_in = [H,W,C](int64_t n,int64_t h,int64_t w,int64_t c){ return ((n*H + h)*W + w)*C + c; };
  auto idx_f  = [kW,C](int64_t kh,int64_t kw,int64_t c){ return (kh*kW + kw)*C + c; };

  for (int64_t n=0; n<N; ++n) {
    for (int64_t oh=0; oh<H_out; ++oh) {
      int64_t h_start = oh*stride_h - pad_top;
      for (int64_t ow=0; ow<W_out; ++ow) {
        int64_t w_start = ow*stride_w - pad_left;
        for (int64_t c=0; c<C; ++c) {
          float m = std::numeric_limits<float>::infinity();
          for (int64_t kh=0; kh<kH; ++kh) {
            int64_t h_in = h_start + kh*rate_h; if (h_in < 0 || h_in >= H) continue;
            for (int64_t kw=0; kw<kW; ++kw) {
              int64_t w_in = w_start + kw*rate_w; if (w_in < 0 || w_in >= W) continue;
              float val = X[idx_in(n,h_in,w_in,c)] - F[idx_f(kh,kw,c)];
              if (val < m) m = val;
            }
          }
          Y[ ((n*H_out + oh)*W_out + ow)*C + c ] = m;
        }
      }
    }
  }
  TF_DeleteStatus(s);
}

// ============================================================================
// LOCAL RESPONSE NORMALIZATION
// ============================================================================

// LocalResponseNormalization - MPSGraph implementation
struct MPSLRNContext { int32_t depth_radius=5; float bias=1.0f; float alpha=1e-4f; float beta=0.75f; };
extern "C" void* MPSLocalResponseNormalization_Create(TF_OpKernelConstruction* ctx) {
  auto* k = new MPSLRNContext();
  TF_Status* s = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "depth_radius", &k->depth_radius, s);
  TF_OpKernelConstruction_GetAttrFloat(ctx, "bias", &k->bias, s);
  TF_OpKernelConstruction_GetAttrFloat(ctx, "alpha", &k->alpha, s);
  TF_OpKernelConstruction_GetAttrFloat(ctx, "beta", &k->beta, s);
  TF_DeleteStatus(s);
  return k;
}
extern "C" void MPSLocalResponseNormalization_Delete(void* kernel) { delete static_cast<MPSLRNContext*>(kernel); }
extern "C" void MPSLocalResponseNormalization_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<MPSLRNContext*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  if (TF_TensorType(input) != TF_FLOAT) {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "LRN MPSGraph supports float only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }

  int64_t N = TF_Dim(input, 0), H = TF_Dim(input, 1), W = TF_Dim(input, 2), C = TF_Dim(input, 3);
  int64_t dims[4] = {N,H,W,C}; int64_t nelems = N*H*W*C;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, 4, nelems * sizeof(float), s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  auto* mpsCtx = GetContext();
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H), @(W), @(C)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // LRN formula: y = x / (bias + alpha * sum(x^2 in window))^beta
    // MPSGraph crossChannelNormalization: out = input / (alpha + beta * sum_squares)^epsilon
    // Map: alpha_mps=bias, beta_mps=alpha/(2r+1), epsilon_mps=beta
    float kernelSize = 2 * k->depth_radius + 1;
    MPSGraphTensor* outputTensor = [graph normalizationWithTensor:inputTensor
                                                             alpha:k->bias
                                                              beta:(k->alpha / kernelSize)
                                                           epsilon:k->beta
                                                        kernelSize:(NSUInteger)kernelSize
                                                              name:@"lrn"];

    size_t bytes = nelems * sizeof(float);
    id<MTLBuffer> inputBuffer = [mpsCtx->device newBufferWithBytes:TF_TensorData(input) length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [mpsCtx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer shape:@[@(N),@(H),@(W),@(C)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer shape:@[@(N),@(H),@(W),@(C)] dataType:MPSDataTypeFloat32];

    id<MTLCommandBuffer> cb = [mpsCtx->commandQueue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{inputTensor: inputData} targetTensors:@[outputTensor] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), outputBuffer.contents, bytes);
    
    [inputBuffer release]; [outputBuffer release]; [inputData release]; [outputData release]; [graph release];
  }
  TF_DeleteStatus(s);
}

// LocalResponseNormalizationGrad - MPSGraph implementation  
extern "C" void* MPSLocalResponseNormalizationGrad_Create(TF_OpKernelConstruction* ctx) {
  return MPSLocalResponseNormalization_Create(ctx);
}
extern "C" void MPSLocalResponseNormalizationGrad_Delete(void* kernel) { MPSLocalResponseNormalization_Delete(kernel); }
extern "C" void MPSLocalResponseNormalizationGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<MPSLRNContext*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* dY_t = nullptr; TF_Tensor* X_t = nullptr; TF_Tensor* Y_t = nullptr;
  TF_GetInput(ctx, 0, &dY_t, s);
  TF_GetInput(ctx, 1, &X_t, s);
  TF_GetInput(ctx, 2, &Y_t, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  if (TF_TensorType(X_t) != TF_FLOAT) {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "LRNGrad MPSGraph supports float only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }

  int64_t N = TF_Dim(X_t, 0), H = TF_Dim(X_t, 1), W = TF_Dim(X_t, 2), C = TF_Dim(X_t, 3);
  int64_t dims[4] = {N,H,W,C}; int64_t nelems = N*H*W*C;
  TF_Tensor* dX_t = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, 4, nelems * sizeof(float), s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  auto* mpsCtx = GetContext();
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSGraphTensor* dY = [graph placeholderWithShape:@[@(N), @(H), @(W), @(C)] dataType:MPSDataTypeFloat32 name:@"dY"];
    MPSGraphTensor* X  = [graph placeholderWithShape:@[@(N), @(H), @(W), @(C)] dataType:MPSDataTypeFloat32 name:@"X"];
    MPSGraphTensor* Y  = [graph placeholderWithShape:@[@(N), @(H), @(W), @(C)] dataType:MPSDataTypeFloat32 name:@"Y"];
    
    float kernelSize = 2 * k->depth_radius + 1;
    MPSGraphTensor* dX = [graph normalizationGradientWithIncomingGradientTensor:dY
                                                                   sourceTensor:X
                                                                   outputTensor:Y
                                                                          alpha:k->bias
                                                                           beta:(k->alpha / kernelSize)
                                                                        epsilon:k->beta
                                                                     kernelSize:(NSUInteger)kernelSize
                                                                           name:@"lrn_grad"];

    size_t bytes = nelems * sizeof(float);
    id<MTLBuffer> dY_buf = [mpsCtx->device newBufferWithBytes:TF_TensorData(dY_t) length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> X_buf  = [mpsCtx->device newBufferWithBytes:TF_TensorData(X_t) length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y_buf  = [mpsCtx->device newBufferWithBytes:TF_TensorData(Y_t) length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dX_buf = [mpsCtx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    MPSGraphTensorData* dY_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:dY_buf shape:@[@(N),@(H),@(W),@(C)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* X_data  = [[MPSGraphTensorData alloc] initWithMTLBuffer:X_buf  shape:@[@(N),@(H),@(W),@(C)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* Y_data  = [[MPSGraphTensorData alloc] initWithMTLBuffer:Y_buf  shape:@[@(N),@(H),@(W),@(C)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* dX_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:dX_buf shape:@[@(N),@(H),@(W),@(C)] dataType:MPSDataTypeFloat32];

    id<MTLCommandBuffer> cb = [mpsCtx->commandQueue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{dY: dY_data, X: X_data, Y: Y_data} targetTensors:@[dX] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(dX_t), dX_buf.contents, bytes);
    
    [dY_buf release]; [X_buf release]; [Y_buf release]; [dX_buf release];
    [dY_data release]; [X_data release]; [Y_data release]; [dX_data release]; [graph release];
  }
  TF_DeleteStatus(s);
}

// ============================================================================
// FRACTIONAL POOLING
// ============================================================================

// FractionalMaxPool
extern "C" void* MPSFractionalMaxPool_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSFractionalMaxPool_Delete(void* kernel) {}
extern "C" void MPSFractionalMaxPool_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FractionalMaxPool - variable pooling regions");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// FractionalMaxPoolGrad
extern "C" void* MPSFractionalMaxPoolGrad_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSFractionalMaxPoolGrad_Delete(void* kernel) {}
extern "C" void MPSFractionalMaxPoolGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FractionalMaxPoolGrad - gradient operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// FractionalAvgPool
extern "C" void* MPSFractionalAvgPool_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSFractionalAvgPool_Delete(void* kernel) {}
extern "C" void MPSFractionalAvgPool_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FractionalAvgPool - variable pooling regions");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// FractionalAvgPoolGrad
extern "C" void* MPSFractionalAvgPoolGrad_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSFractionalAvgPoolGrad_Delete(void* kernel) {}
extern "C" void MPSFractionalAvgPoolGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FractionalAvgPoolGrad - gradient operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
