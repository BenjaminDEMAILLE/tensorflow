/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Extended NN Operations
// BiasAdd, Conv2DBackprop, FusedConv2D, QuantizedConv2D, DepthToSpace, SpaceToDepth, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// BiasAdd
struct MPSBiasAddAttrs {
  char data_format[8];
};

void* MPSBiasAdd_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSBiasAddAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", attrs->data_format, 8, s);
  if (TF_GetCode(s) != TF_OK) {
    strcpy(attrs->data_format, "NHWC");
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSBiasAdd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSBiasAddAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* value = nullptr;
  TF_Tensor* bias = nullptr;
  
  TF_GetInput(ctx, 0, &value, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &bias, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // value: [batch, height, width, channels] or [batch, channels, height, width]
  // bias: [channels]
  int num_dims = TF_NumDims(value);
  int64_t* dims = new int64_t[num_dims];
  size_t total_elements = 1;
  for (int i = 0; i < num_dims; i++) {
    dims[i] = TF_Dim(value, i);
    total_elements *= dims[i];
  }
  
  int64_t bias_size = TF_Dim(bias, 0);
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    NSMutableArray* valueShape = [NSMutableArray array];
    for (int i = 0; i < num_dims; i++) {
      [valueShape addObject:@(dims[i])];
    }
    
    MPSGraphTensor* valueTensor = [graph placeholderWithShape:valueShape
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"value"];
    
    MPSGraphTensor* biasTensor = [graph placeholderWithShape:@[@(bias_size)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:@"bias"];
    
    // Broadcast and add
    MPSGraphTensor* output = [graph additionWithPrimaryTensor:valueTensor
                                              secondaryTensor:biasTensor
                                                         name:@"biasadd"];
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* value_data = static_cast<float*>(TF_TensorData(value));
    float* bias_data = static_cast<float*>(TF_TensorData(bias));
    
    size_t value_bytes = total_elements * sizeof(float);
    size_t bias_bytes = bias_size * sizeof(float);
    
    id<MTLBuffer> valueBuffer = [device newBufferWithBytes:value_data
                                                    length:value_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> biasBuffer = [device newBufferWithBytes:bias_data
                                                   length:bias_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:value_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* valueData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:valueBuffer
                    shape:valueShape
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* biasData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:biasBuffer
                    shape:@[@(bias_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:valueShape
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{
      valueTensor: valueData,
      biasTensor: biasData
    };
    
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, num_dims, value_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      delete[] dims;
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    float* out_data = static_cast<float*>(TF_TensorData(tf_output));
    memcpy(out_data, [outputBuffer contents], value_bytes);
  }
  
  delete[] dims;
  TF_DeleteStatus(s);
}

void MPSBiasAdd_Delete(void* kernel) {
  delete static_cast<MPSBiasAddAttrs*>(kernel);
}

// DepthToSpace
struct MPSDepthToSpaceAttrs {
  int block_size;
};

void* MPSDepthToSpace_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSDepthToSpaceAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "block_size", &attrs->block_size, s);
  if (TF_GetCode(s) != TF_OK) attrs->block_size = 2;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSDepthToSpace_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSDepthToSpaceAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // input: [batch, height, width, channels]
  int64_t batch = TF_Dim(input, 0);
  int64_t height = TF_Dim(input, 1);
  int64_t width = TF_Dim(input, 2);
  int64_t channels = TF_Dim(input, 3);
  
  int bs = attrs->block_size;
  int64_t out_h = height * bs;
  int64_t out_w = width * bs;
  int64_t out_c = channels / (bs * bs);
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(height), @(width), @(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // DepthToSpace operation
    MPSGraphTensor* output = [graph depthToSpace2DTensor:inputTensor
                                              widthAxis:2
                                             heightAxis:1
                                              depthAxis:3
                                              blockSize:bs
                                     usePixelShuffleOrder:NO
                                                     name:@"depth_to_space"];
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* input_data = static_cast<float*>(TF_TensorData(input));
    size_t input_bytes = batch * height * width * channels * sizeof(float);
    size_t output_bytes = batch * out_h * out_w * out_c * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch), @(height), @(width), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(batch), @(out_h), @(out_w), @(out_c)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: inputData};
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    int64_t out_dims[] = {batch, out_h, out_w, out_c};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, output_bytes, s);
    
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

void MPSDepthToSpace_Delete(void* kernel) {
  delete static_cast<MPSDepthToSpaceAttrs*>(kernel);
}

// SpaceToDepth
void* MPSSpaceToDepth_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSDepthToSpaceAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "block_size", &attrs->block_size, s);
  if (TF_GetCode(s) != TF_OK) attrs->block_size = 2;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSSpaceToDepth_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSDepthToSpaceAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t batch = TF_Dim(input, 0);
  int64_t height = TF_Dim(input, 1);
  int64_t width = TF_Dim(input, 2);
  int64_t channels = TF_Dim(input, 3);
  
  int bs = attrs->block_size;
  int64_t out_h = height / bs;
  int64_t out_w = width / bs;
  int64_t out_c = channels * bs * bs;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(height), @(width), @(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // SpaceToDepth operation
    MPSGraphTensor* output = [graph spaceToDepth2DTensor:inputTensor
                                              widthAxis:2
                                             heightAxis:1
                                              depthAxis:3
                                              blockSize:bs
                                     usePixelShuffleOrder:NO
                                                     name:@"space_to_depth"];
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* input_data = static_cast<float*>(TF_TensorData(input));
    size_t input_bytes = batch * height * width * channels * sizeof(float);
    size_t output_bytes = batch * out_h * out_w * out_c * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch), @(height), @(width), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(batch), @(out_h), @(out_w), @(out_c)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: inputData};
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    int64_t out_dims[] = {batch, out_h, out_w, out_c};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, output_bytes, s);
    
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

void MPSSpaceToDepth_Delete(void* kernel) {
  delete static_cast<MPSDepthToSpaceAttrs*>(kernel);
}

void RegisterNNExtendedOps(const char* platform_name, TF_Status* status) {
  // BiasAdd
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("BiasAdd", platform_name,
                                                &MPSBiasAdd_Create,
                                                &MPSBiasAdd_Compute,
                                                &MPSBiasAdd_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSBiasAdd", kb, status);
  }
  
  // DepthToSpace
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("DepthToSpace", platform_name,
                                                &MPSDepthToSpace_Create,
                                                &MPSDepthToSpace_Compute,
                                                &MPSDepthToSpace_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSDepthToSpace", kb, status);
  }
  
  // SpaceToDepth
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("SpaceToDepth", platform_name,
                                                &MPSSpaceToDepth_Create,
                                                &MPSSpaceToDepth_Compute,
                                                &MPSSpaceToDepth_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSpaceToDepth", kb, status);
  }
}

// Additional NN operations stubs
void* MPSConv2DBackpropInput_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSConv2DBackpropInput_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2DBackpropInput not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSConv2DBackpropInput_Delete(void* kernel) {}

void* MPSConv2DBackpropFilter_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSConv2DBackpropFilter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2DBackpropFilter not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSConv2DBackpropFilter_Delete(void* kernel) {}

void* MPSFusedBatchNormGrad_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFusedBatchNormGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FusedBatchNormGrad not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFusedBatchNormGrad_Delete(void* kernel) {}

void* MPSFusedBatchNormGradV2_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFusedBatchNormGradV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FusedBatchNormGradV2 not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFusedBatchNormGradV2_Delete(void* kernel) {}

void* MPSFusedBatchNormGradV3_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFusedBatchNormGradV3_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FusedBatchNormGradV3 not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFusedBatchNormGradV3_Delete(void* kernel) {}

void* MPSSpaceToBatchND_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSSpaceToBatchND_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SpaceToBatchND not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSSpaceToBatchND_Delete(void* kernel) {}

void* MPSBatchToSpaceND_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSBatchToSpaceND_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "BatchToSpaceND not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSBatchToSpaceND_Delete(void* kernel) {}

void* MPSDilation2D_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSDilation2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Dilation2D not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSDilation2D_Delete(void* kernel) {}

void* MPSDilation2DBackpropInput_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSDilation2DBackpropInput_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Dilation2DBackpropInput not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSDilation2DBackpropInput_Delete(void* kernel) {}

void* MPSDilation2DBackpropFilter_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSDilation2DBackpropFilter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Dilation2DBackpropFilter not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSDilation2DBackpropFilter_Delete(void* kernel) {}

void* MPSErosion2D_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSErosion2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Erosion2D not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSErosion2D_Delete(void* kernel) {}

void* MPSMaxPool3D_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSMaxPool3D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "MaxPool3D not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSMaxPool3D_Delete(void* kernel) {}

void* MPSMaxPool3DGrad_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSMaxPool3DGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "MaxPool3DGrad not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSMaxPool3DGrad_Delete(void* kernel) {}

void* MPSAvgPool3D_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSAvgPool3D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "AvgPool3D not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSAvgPool3D_Delete(void* kernel) {}

void* MPSAvgPool3DGrad_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSAvgPool3DGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "AvgPool3DGrad not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSAvgPool3DGrad_Delete(void* kernel) {}

void* MPSLocalResponseNormalization_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLocalResponseNormalization_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "LocalResponseNormalization not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLocalResponseNormalization_Delete(void* kernel) {}

void* MPSLocalResponseNormalizationGrad_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLocalResponseNormalizationGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "LocalResponseNormalizationGrad not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLocalResponseNormalizationGrad_Delete(void* kernel) {}

void* MPSFractionalMaxPool_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFractionalMaxPool_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FractionalMaxPool not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFractionalMaxPool_Delete(void* kernel) {}

void* MPSFractionalMaxPoolGrad_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFractionalMaxPoolGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FractionalMaxPoolGrad not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFractionalMaxPoolGrad_Delete(void* kernel) {}

void* MPSFractionalAvgPool_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFractionalAvgPool_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FractionalAvgPool not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFractionalAvgPool_Delete(void* kernel) {}

void* MPSFractionalAvgPoolGrad_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFractionalAvgPoolGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FractionalAvgPoolGrad not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFractionalAvgPoolGrad_Delete(void* kernel) {}

}  // namespace mps
}  // namespace tensorflow

