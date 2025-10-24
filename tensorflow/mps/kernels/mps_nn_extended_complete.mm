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
// MORPHOLOGICAL OPERATIONS
// ============================================================================

// Dilation2D - CPU fallback (NHWC, float)
namespace { struct MPSMorphAttrs { std::vector<int64_t> strides; std::vector<int64_t> rates; std::string padding; }; }
extern "C" void* MPSDilation2D_Create(TF_OpKernelConstruction* ctx) {
  auto* a = new MPSMorphAttrs();
  TF_Status* s = TF_NewStatus();
  int64_t* strides_data=nullptr; int strides_len=0; TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s); if (strides_data && strides_len>0) a->strides.assign(strides_data, strides_data+strides_len);
  int64_t* rates_data=nullptr; int rates_len=0; TF_OpKernelConstruction_GetAttrInt64List(ctx, "rates", &rates_data, &rates_len, s); if (rates_data && rates_len>0) a->rates.assign(rates_data, rates_data+rates_len);
  char* padding_data=nullptr; size_t padding_len=0; TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s); if (padding_data) a->padding.assign(padding_data, padding_len);
  TF_DeleteStatus(s);
  return a;
}
extern "C" void MPSDilation2D_Delete(void* kernel) { delete reinterpret_cast<MPSMorphAttrs*>(kernel); }
extern "C" void MPSDilation2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* a = reinterpret_cast<MPSMorphAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input=nullptr; TF_Tensor* filter=nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &filter, s); if (TF_GetCode(s)!=TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT || TF_TensorType(filter) != TF_FLOAT) { TF_SetStatus(s, TF_UNIMPLEMENTED, "Dilation2D CPU fallback supports float only"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_NumDims(input)!=4 || TF_NumDims(filter)!=3) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Dilation2D expects input [N,H,W,C] and filter [kH,kW,C]"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int64_t N=TF_Dim(input,0), H=TF_Dim(input,1), W=TF_Dim(input,2), C=TF_Dim(input,3);
  int64_t kH=TF_Dim(filter,0), kW=TF_Dim(filter,1), Cf=TF_Dim(filter,2);
  if (Cf != C) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Dilation2D filter channels must match input channels"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
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
          float m = -std::numeric_limits<float>::infinity();
          for (int64_t kh=0; kh<kH; ++kh) {
            int64_t h_in = h_start + kh*rate_h; if (h_in < 0 || h_in >= H) continue;
            for (int64_t kw=0; kw<kW; ++kw) {
              int64_t w_in = w_start + kw*rate_w; if (w_in < 0 || w_in >= W) continue;
              float val = X[idx_in(n,h_in,w_in,c)] + F[idx_f(kh,kw,c)];
              if (val > m) m = val;
            }
          }
          Y[ ((n*H_out + oh)*W_out + ow)*C + c ] = m;
        }
      }
    }
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

// LocalResponseNormalization - CPU fallback
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
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  // Expect NHWC
  const int64_t N = TF_Dim(input, 0);
  const int64_t H = TF_Dim(input, 1);
  const int64_t W = TF_Dim(input, 2);
  const int64_t C = TF_Dim(input, 3);
  const int64_t dims[4] = {N,H,W,C};
  const int64_t nelems = N*H*W*C;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, 4, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  const float* x = (const float*)TF_TensorData(input);
  float* y = (float*)TF_TensorData(output);
  const int r = k->depth_radius; const float bias=k->bias, alpha=k->alpha, beta=k->beta; const float alpha_over_n = alpha / (2*r+1);

  // For each position, compute across channels
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t w = 0; w < W; ++w) {
        int64_t base = ((n*H + h)*W + w)*C;
        // Precompute cumulative sum of squares across channels for sliding window
        std::vector<float> cum(C+1, 0.0f);
        for (int64_t c = 0; c < C; ++c) {
          float v = x[base + c];
          cum[c+1] = cum[c] + v*v;
        }
        for (int64_t c = 0; c < C; ++c) {
          int64_t c0 = std::max<int64_t>(0, c - r);
          int64_t c1 = std::min<int64_t>(C - 1, c + r);
          float sumsq = cum[c1+1] - cum[c0];
          float S = bias + alpha_over_n * sumsq;
          y[base + c] = x[base + c] * std::pow(S, -beta);
        }
      }
    }
  }
  TF_DeleteStatus(status);
}

// LocalResponseNormalizationGrad - CPU fallback
extern "C" void* MPSLocalResponseNormalizationGrad_Create(TF_OpKernelConstruction* ctx) {
  // Reuse same attributes as forward op
  return MPSLocalResponseNormalization_Create(ctx);
}
extern "C" void MPSLocalResponseNormalizationGrad_Delete(void* kernel) { MPSLocalResponseNormalization_Delete(kernel); }
extern "C" void MPSLocalResponseNormalizationGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<MPSLRNContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  // Inputs: input_grads (dY), input_image (X), output_image (Y)
  TF_Tensor* dY_t = nullptr; TF_Tensor* X_t = nullptr; TF_Tensor* Y_t = nullptr;
  TF_GetInput(ctx, 0, &dY_t, status);
  TF_GetInput(ctx, 1, &X_t, status);
  TF_GetInput(ctx, 2, &Y_t, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  const int64_t N = TF_Dim(X_t, 0), H = TF_Dim(X_t, 1), W = TF_Dim(X_t, 2), C = TF_Dim(X_t, 3);
  const int64_t dims[4] = {N,H,W,C}; const int64_t nelems = N*H*W*C;
  TF_Tensor* dX_t = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, 4, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  const float* dY = (const float*)TF_TensorData(dY_t);
  const float* X  = (const float*)TF_TensorData(X_t);
  const float* Y  = (const float*)TF_TensorData(Y_t);
  float* dX       = (float*)TF_TensorData(dX_t);

  const int r = k->depth_radius; const float bias=k->bias, alpha=k->alpha, beta=k->beta; const float alpha_over_n = alpha / (2*r+1);
  // Initialize dX to 0
  std::fill(dX, dX + nelems, 0.0f);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t w = 0; w < W; ++w) {
        int64_t base = ((n*H + h)*W + w)*C;
        // Precompute sumsq and S per channel
        std::vector<float> cum(C+1, 0.0f);
        for (int64_t c = 0; c < C; ++c) { float v = X[base+c]; cum[c+1] = cum[c] + v*v; }
        std::vector<float> S(C);
        for (int64_t c = 0; c < C; ++c) {
          int64_t c0 = std::max<int64_t>(0, c - r);
          int64_t c1 = std::min<int64_t>(C - 1, c + r);
          float sumsq = cum[c1+1] - cum[c0];
          S[c] = bias + alpha_over_n * sumsq;
        }
        // Gradient accumulation
        for (int64_t c = 0; c < C; ++c) {
          // Diagonal term: dY_c * S_c^{-beta}
          float diag = dY[base+c] * std::pow(S[c], -beta);
          dX[base+c] += diag;
          // Off-diagonal via S dependence: for each j where c in window of j
          int64_t j0 = std::max<int64_t>(0, c - r);
          int64_t j1 = std::min<int64_t>(C - 1, c + r);
          float x_c = X[base+c];
          for (int64_t j = j0; j <= j1; ++j) {
            // d y_j / d x_c = x_j * (-beta) * S_j^{-beta-1} * (2*alpha/n * x_c)
            float coeff = (-beta) * std::pow(S[j], -beta - 1.0f) * (2.0f * alpha_over_n * x_c);
            dX[base+c] += dY[base+j] * X[base+j] * coeff;
          }
        }
      }
    }
  }
  TF_DeleteStatus(status);
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
