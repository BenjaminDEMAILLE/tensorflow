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

// Dilation2D
extern "C" void* MPSDilation2D_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSDilation2D_Delete(void* kernel) {}
extern "C" void MPSDilation2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Dilation2D - morphological operation needs custom implementation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// Dilation2DBackpropInput
extern "C" void* MPSDilation2DBackpropInput_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSDilation2DBackpropInput_Delete(void* kernel) {}
extern "C" void MPSDilation2DBackpropInput_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Dilation2DBackpropInput gradient not implemented");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// Dilation2DBackpropFilter
extern "C" void* MPSDilation2DBackpropFilter_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSDilation2DBackpropFilter_Delete(void* kernel) {}
extern "C" void MPSDilation2DBackpropFilter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Dilation2DBackpropFilter gradient not implemented");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// Erosion2D
extern "C" void* MPSErosion2D_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSErosion2D_Delete(void* kernel) {}
extern "C" void MPSErosion2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Erosion2D - morphological operation needs custom implementation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ============================================================================
// LOCAL RESPONSE NORMALIZATION
// ============================================================================

// LocalResponseNormalization
extern "C" void* MPSLocalResponseNormalization_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSLocalResponseNormalization_Delete(void* kernel) {}
extern "C" void MPSLocalResponseNormalization_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LRN - needs MPS normalization kernel");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// LocalResponseNormalizationGrad
extern "C" void* MPSLocalResponseNormalizationGrad_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSLocalResponseNormalizationGrad_Delete(void* kernel) {}
extern "C" void MPSLocalResponseNormalizationGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LRNGrad - gradient operation");
  TF_OpKernelContext_Failure(ctx, status);
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
