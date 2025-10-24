/* Copyright 2025 The TensorFlow Authors.

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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/mps/ops/mps_ops_registry.h"
#include "tensorflow/mps/utils/mps_utils.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

#include <memory>
#include <vector>
#include <algorithm>

using namespace tensorflow::mps;

// Forward declarations for stream structures
struct MPSDevice;
struct MPSStreamStruct;
struct MPSStream;

// ============================================================================
// All NN Ops Implementations (Conv2D, DepthwiseConv2D, MaxPool, AvgPool, Softmax, FusedBatchNormV3, Swish, Gelu)
// ============================================================================

// ===== MPS Conv2D kernel (float, NHWC only) =====
namespace {
struct MPSConv2DAttrs {
  std::vector<int64_t> strides;
  std::string padding;
  std::vector<int64_t> dilations;
  std::string data_format;
};
}

extern "C" void* MPSConv2D_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSConv2DAttrs();
  TF_Status* s = TF_NewStatus();
  
  // Get strides
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  // Get padding
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  // Get dilations
  int64_t* dilations_data = nullptr;
  int dilations_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "dilations", &dilations_data, &dilations_len, s);
  if (TF_GetCode(s) == TF_OK && dilations_data && dilations_len > 0) {
    attrs->dilations.assign(dilations_data, dilations_data + dilations_len);
  }
  
  // Get data_format
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSConv2D_Delete(void* kernel) {
  auto* attrs = reinterpret_cast<MPSConv2DAttrs*>(kernel);
  delete attrs;
}

extern "C" void MPSConv2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = reinterpret_cast<MPSConv2DAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_Tensor* filter = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &filter, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_TensorType(filter)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D[MPS] input and filter must have same dtype");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D[MPS] supports float32, float16, and bfloat16");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? sizeof(float) : sizeof(uint16_t);
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  // Only support NHWC for now
  if (attrs->data_format != "NHWC") {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2D[MPS] only supports NHWC format");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  // Input: [N, H, W, C_in], Filter: [kH, kW, C_in, C_out]
  if (TF_NumDims(input) != 4 || TF_NumDims(filter) != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D[MPS] expects 4D input and filter");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C_in = TF_Dim(input, 3);
  
  int64_t kH = TF_Dim(filter, 0);
  int64_t kW = TF_Dim(filter, 1);
  int64_t C_out = TF_Dim(filter, 3);
  
  if (TF_Dim(filter, 2) != C_in) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D filter channels mismatch");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  // Get strides (NHWC: [1, stride_h, stride_w, 1])
  int64_t stride_h = (attrs->strides.size() >= 4) ? attrs->strides[1] : 1;
  int64_t stride_w = (attrs->strides.size() >= 4) ? attrs->strides[2] : 1;
  
  // Get dilations (NHWC: [1, dil_h, dil_w, 1])
  int64_t dil_h = (attrs->dilations.size() >= 4) ? attrs->dilations[1] : 1;
  int64_t dil_w = (attrs->dilations.size() >= 4) ? attrs->dilations[2] : 1;
  
  // Compute output size based on padding
  int64_t H_out = 0, W_out = 0;
  int64_t pad_top = 0, pad_left = 0;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h_total = std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in);
    int64_t pad_w_total = std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in);
    pad_top = pad_h_total / 2;
    pad_left = pad_w_total / 2;
  } else if (attrs->padding == "VALID") {
    H_out = (H_in - (kH - 1) * dil_h) / stride_h;
    W_out = (W_in - (kW - 1) * dil_w) / stride_w;
    if (H_out <= 0 || W_out <= 0) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D output dimensions must be positive");
      TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
    }
    pad_top = 0;
    pad_left = 0;
  } else {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2D[MPS] only supports SAME/VALID padding");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  int64_t out_dims[4] = {N, H_out, W_out, C_out};
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C_out * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_dims, 4, out_bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Try GPU via MPSGraph
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2D[MPS] requires stream (host fallback not implemented)");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Input: [N, H, W, C_in] NHWC
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    // Filter: [kH, kW, C_in, C_out] -> needs to be [C_out, C_in, kH, kW] for MPSGraph
    // We'll transpose on host before feeding
    MPSGraphTensor* filterTensor = [graph placeholderWithShape:@[@(C_out), @(C_in), @(kH), @(kW)]
                                                        dataType:mps_dtype
                                                            name:@"filter"];
    
    // Create convolution descriptor
    MPSGraphConvolution2DOpDescriptor* desc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    desc.strideInX = (NSUInteger)stride_w;
    desc.strideInY = (NSUInteger)stride_h;
    desc.dilationRateInX = (NSUInteger)dil_w;
    desc.dilationRateInY = (NSUInteger)dil_h;
    desc.paddingLeft = (NSUInteger)pad_left;
    desc.paddingRight = (NSUInteger)std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in - pad_top);
    desc.paddingTop = (NSUInteger)pad_top;
    desc.paddingBottom = (NSUInteger)std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in - pad_left);
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    desc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;  // [Out, In, H, W]
    
    // Perform convolution
    MPSGraphTensor* outputTensor = [graph convolution2DWithSourceTensor:inputTensor
                                                          weightsTensor:filterTensor
                                                             descriptor:desc
                                                                   name:@"conv2d"];
    
    // Prepare input buffer
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C_in * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    // Transpose filter: [kH, kW, C_in, C_out] -> [C_out, C_in, kH, kW]
    size_t filter_elem_count = (size_t)C_out * (size_t)C_in * (size_t)kH * (size_t)kW;
    size_t filter_bytes = filter_elem_count * elem_size;
    std::vector<uint8_t> filter_transposed(filter_bytes);
    
    if (is_half) {
      const uint16_t* filter_data = (const uint16_t*)TF_TensorData(filter);
      uint16_t* filter_out = (uint16_t*)filter_transposed.data();
      for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t ci = 0; ci < C_in; ++ci) {
          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              int64_t src_idx = ((kh * kW + kw) * C_in + ci) * C_out + co;
              int64_t dst_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
              filter_out[dst_idx] = filter_data[src_idx];
            }
          }
        }
      }
    } else {
      const float* filter_data = (const float*)TF_TensorData(filter);
      float* filter_out = (float*)filter_transposed.data();
      for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t ci = 0; ci < C_in; ++ci) {
          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              int64_t src_idx = ((kh * kW + kw) * C_in + ci) * C_out + co;
              int64_t dst_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
              filter_out[dst_idx] = filter_data[src_idx];
            }
          }
        }
      }
    }
    id<MTLBuffer> filterBuffer = [dev newBufferWithBytes:filter_transposed.data()
                                                   length:filter_bytes
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* filterData = [[MPSGraphTensorData alloc] initWithMTLBuffer:filterBuffer
                                                                              shape:@[@(C_out), @(C_in), @(kH), @(kW)]
                                                                           dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C_out)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData, filterTensor: filterData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS DepthwiseConv2dNative kernel (float, half, NHWC only) =====
namespace {
struct MPSDepthwiseConv2DAttrs {
  std::vector<int64_t> strides;
  std::string padding;
  std::vector<int64_t> dilations;
  std::string data_format;
};
}

extern "C" void* MPSDepthwiseConv2D_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSDepthwiseConv2DAttrs();
  TF_Status* s = TF_NewStatus();
  
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  int64_t* dilations_data = nullptr;
  int dilations_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "dilations", &dilations_data, &dilations_len, s);
  if (TF_GetCode(s) == TF_OK && dilations_data && dilations_len > 0) {
    attrs->dilations.assign(dilations_data, dilations_data + dilations_len);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSDepthwiseConv2D_Delete(void* kernel_ptr) {
  delete static_cast<MPSDepthwiseConv2DAttrs*>(kernel_ptr);
}

extern "C" void MPSDepthwiseConv2D_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSDepthwiseConv2DAttrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_Tensor* filter = nullptr;
  TF_GetInput(ctx, 1, &filter, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS DepthwiseConv2dNative: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int64_t num_dims_input = TF_NumDims(input);
  int64_t num_dims_filter = TF_NumDims(filter);
  if (num_dims_input != 4 || num_dims_filter != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS DepthwiseConv2dNative: input/filter must be 4D");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Input: [N, H, W, C_in] NHWC
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C_in = TF_Dim(input, 3);
  
  // Filter: [kH, kW, C_in, depth_multiplier]
  int64_t kH = TF_Dim(filter, 0);
  int64_t kW = TF_Dim(filter, 1);
  int64_t filter_c = TF_Dim(filter, 2);
  int64_t depth_mult = TF_Dim(filter, 3);
  
  if (filter_c != C_in) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS DepthwiseConv2dNative: filter channels != input channels");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  int64_t C_out = C_in * depth_mult;
  
  // Get strides
  int64_t stride_h = 1, stride_w = 1;
  if (attrs->strides.size() == 4) {
    stride_h = attrs->strides[1];
    stride_w = attrs->strides[2];
  }
  
  // Get dilations
  int64_t dil_h = 1, dil_w = 1;
  if (attrs->dilations.size() == 4) {
    dil_h = attrs->dilations[1];
    dil_w = attrs->dilations[2];
  }
  
  // Compute padding
  int64_t pad_top = 0, pad_left = 0;
  int64_t H_out, W_out;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h = std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in);
    int64_t pad_w = std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in);
    pad_top = pad_h / 2;
    pad_left = pad_w / 2;
  } else {
    H_out = (H_in - (kH - 1) * dil_h - 1) / stride_h + 1;
    W_out = (W_in - (kW - 1) * dil_w - 1) / stride_w + 1;
  }
  
  int64_t output_dims[4] = {N, H_out, W_out, C_out};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, output_dims, 4, C_out * H_out * W_out * N * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C_out * elem_size;
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Input: [N, H, W, C_in]
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    // Filter: [kH, kW, C_in, depth_mult] -> needs to be [kH, kW, C_in, depth_mult] for depthwise
    MPSGraphTensor* filterTensor = [graph placeholderWithShape:@[@(kH), @(kW), @(C_in), @(depth_mult)]
                                                        dataType:mps_dtype
                                                            name:@"filter"];
    
    // Create depthwise convolution descriptor
    MPSGraphDepthwiseConvolution3DOpDescriptor* desc = [[MPSGraphDepthwiseConvolution3DOpDescriptor alloc] init];
    desc.strides = @[@1, @(stride_h), @(stride_w)];
    desc.dilationRates = @[@1, @(dil_h), @(dil_w)];
    desc.paddingValues = @[@0, @(pad_top), @(pad_left), @0,
                          @(std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in - pad_top)),
                          @(std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in - pad_left))];
    desc.paddingStyle = MPSGraphPaddingStyleExplicit;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    desc.weightsLayout = MPSGraphTensorNamedDataLayoutHWIO;  // [H, W, In, Out]
    
    // Perform depthwise convolution
    MPSGraphTensor* outputTensor = [graph depthwiseConvolution3DWithSourceTensor:inputTensor
                                                                   weightsTensor:filterTensor
                                                                      descriptor:desc
                                                                            name:@"depthwise_conv"];
    
    // Prepare buffers
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C_in * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    size_t filter_bytes = (size_t)kH * (size_t)kW * (size_t)C_in * (size_t)depth_mult * elem_size;
    id<MTLBuffer> filterBuffer = [dev newBufferWithBytes:TF_TensorData(filter)
                                                   length:filter_bytes
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* filterData = [[MPSGraphTensorData alloc] initWithMTLBuffer:filterBuffer
                                                                              shape:@[@(kH), @(kW), @(C_in), @(depth_mult)]
                                                                           dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C_out)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData, filterTensor: filterData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS Maximum/Minimum/Sigmoid/Tanh kernels (half) =====
namespace {
static id<MTLComputePipelineState> g_max_h_pipeline=nil, g_min_h_pipeline=nil, g_sigmoid_h_pipeline=nil, g_tanh_h_pipeline=nil;
static dispatch_once_t g_max_h_once, g_min_h_once, g_sigmoid_h_once, g_tanh_h_once;

static void EnsureMaxHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_max_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void max_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2)]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = max(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS MaximumHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"max_h_k"];
    g_max_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_max_h_pipeline) { NSLog(@"MPS MaximumHalf: pipeline error: %@", err); }
  });
}
static void EnsureMinHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_min_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void min_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2]]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = min(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS MinimumHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"min_h_k"];
    g_min_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_min_h_pipeline) { NSLog(@"MPS MinimumHalf: pipeline error: %@", err); }
  });
}
static void EnsureSigmoidHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_sigmoid_h_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void sigmoid_h_k(const device half* in [[buffer(0)]],\n"
                       @"                        device half* out [[buffer(1)]],\n"
                       @"                        uint gid [[thread_position_in_grid]]) {\n"
                       @"  half x = in[gid];\n"
                       @"  out[gid] = (half)(1.0h / (1.0h + exp(-x)));\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS SigmoidHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"sigmoid_h_k"];
    g_sigmoid_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_sigmoid_h_pipeline) { NSLog(@"MPS SigmoidHalf: pipeline error: %@", err); }
  });
}
static void EnsureTanhHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_tanh_h_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void tanh_h_k(const device half* in [[buffer(0)]],\n"
                       @"                     device half* out [[buffer(1)]],\n"
                       @"                     uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = tanh(in[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS TanhHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"tanh_h_k"];
    g_tanh_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_tanh_h_pipeline) { NSLog(@"MPS TanhHalf: pipeline error: %@", err); }
  });
}
}

extern "C" void MPSMaximumHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a=nullptr, *b=nullptr;
  TF_GetInput(ctx,0,&a,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  TF_GetInput(ctx,1,&b,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(a)!=TF_HALF||TF_TensorType(b)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Maximum[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_a=TF_NumDims(a),nd_b=TF_NumDims(b); int64_t nelems_a=1,nelems_b=1;
  for(int i=0;i<nd_a;++i){int64_t d=TF_Dim(a,i);nelems_a*=(d<0?0:d);} for(int i=0;i<nd_b;++i){int64_t d=TF_Dim(b,i);nelems_b*=(d<0?0:d);}
  bool same_shape=(nd_a==nd_b); if(same_shape)for(int i=0;i<nd_a;++i)if(TF_Dim(a,i)!=TF_Dim(b,i)){same_shape=false;break;}
  int nd=nd_a; int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(same_shape){
    if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} int64_t nelems=1;
    for(int i=0;i<nd;++i){int64_t d=TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(uint16_t);
    TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
    SP_Stream cstream=TF_GetStream(ctx,s);
    if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
      const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
      for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va>vb?va:vb);}
      TF_DeleteStatus(s);return;}
    auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureMaxHalfPipeline(dev);
    if(!g_max_h_pipeline){const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out); for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va>vb?va:vb);} TF_DeleteStatus(s);return;}
    const void* ha=TF_TensorData(a),*hb=TF_TensorData(b);void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes);memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_max_h_pipeline];[enc setBuffer:ba offset:0 atIndex:0];[enc setBuffer:bb offset:0 atIndex:1];[enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes);TF_DeleteStatus(s);return;
  }
  bool a_scalar=(nelems_a==1),b_scalar=(nelems_b==1);
  if(!(a_scalar||b_scalar)){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Maximum shapes must match or one be scalar");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_out=a_scalar?nd_b:nd_a;if(nd_out>8){dyn.reset(new int64_t[nd_out]);dims=dyn.get();}
  int64_t nelems=1;for(int i=0;i<nd_out;++i){int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd_out,bytes,s);if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
  if(a_scalar){float av=HalfToFloat(pa[0]);for(int64_t i=0;i<nelems;++i){float vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(av>vb?av:vb);}}
  else{float bv=HalfToFloat(pb[0]);for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]);po[i]=FloatToHalf(va>bv?va:bv);}}
  TF_DeleteStatus(s);
}

extern "C" void MPSMinimumHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a=nullptr, *b=nullptr;
  TF_GetInput(ctx,0,&a,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  TF_GetInput(ctx,1,&b,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(a)!=TF_HALF||TF_TensorType(b)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Minimum[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_a=TF_NumDims(a),nd_b=TF_NumDims(b); int64_t nelems_a=1,nelems_b=1;
  for(int i=0;i<nd_a;++i){int64_t d=TF_Dim(a,i);nelems_a*=(d<0?0:d);} for(int i=0;i<nd_b;++i){int64_t d=TF_Dim(b,i);nelems_b*=(d<0?0:d);}
  bool same_shape=(nd_a==nd_b); if(same_shape)for(int i=0;i<nd_a;++i)if(TF_Dim(a,i)!=TF_Dim(b,i)){same_shape=false;break;}
  int nd=nd_a; int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(same_shape){
    if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} int64_t nelems=1;
    for(int i=0;i<nd;++i){int64_t d=TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(uint16_t);
    TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
    SP_Stream cstream=TF_GetStream(ctx,s);
    if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
      const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
      for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va<vb?va:vb);}
      TF_DeleteStatus(s);return;}
    auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureMinHalfPipeline(dev);
    if(!g_min_h_pipeline){const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out); for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va<vb?va:vb);} TF_DeleteStatus(s);return;}
    const void* ha=TF_TensorData(a),*hb=TF_TensorData(b);void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes);memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_min_h_pipeline];[enc setBuffer:ba offset:0 atIndex:0];[enc setBuffer:bb offset:0 atIndex:1];[enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes);TF_DeleteStatus(s);return;
  }
  bool a_scalar=(nelems_a==1),b_scalar=(nelems_b==1);
  if(!(a_scalar||b_scalar)){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Minimum shapes must match or one be scalar");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_out=a_scalar?nd_b:nd_a;if(nd_out>8){dyn.reset(new int64_t[nd_out]);dims=dyn.get();}
  int64_t nelems=1;for(int i=0;i<nd_out;++i){int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd_out,bytes,s);if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
  if(a_scalar){float av=HalfToFloat(pa[0]);for(int64_t i=0;i<nelems;++i){float vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(av<vb?av:vb);}}
  else{float bv=HalfToFloat(pb[0]);for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]);po[i]=FloatToHalf(va<bv?va:bv);}}
  TF_DeleteStatus(s);
}

extern "C" void MPSSigmoidHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input=nullptr;TF_GetInput(ctx,0,&input,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(input)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Sigmoid[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd=TF_NumDims(input);int64_t nelems=1;int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} for(int i=0;i<nd;++i){int64_t d=TF_Dim(input,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* output=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  SP_Stream cstream=TF_GetStream(ctx,s);
  if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
    const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output);
    for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(1.0f/(1.0f+expf(-x)));}
    TF_DeleteStatus(s);return;}
  auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureSigmoidHalfPipeline(dev);
  if(!g_sigmoid_h_pipeline){const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output); for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(1.0f/(1.0f+expf(-x)));} TF_DeleteStatus(s);return;}
  const void* in_host=TF_TensorData(input);void* out_host=TF_TensorData(output);
  id<MTLBuffer> inb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],outb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents,in_host,bytes);
  id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
  [enc setComputePipelineState:g_sigmoid_h_pipeline];[enc setBuffer:inb offset:0 atIndex:0];[enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];
  [cb commit];[cb waitUntilCompleted];memcpy(out_host,outb.contents,bytes);TF_DeleteStatus(s);
}

extern "C" void MPSTanhHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input=nullptr;TF_GetInput(ctx,0,&input,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(input)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Tanh[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd=TF_NumDims(input);int64_t nelems=1;int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} for(int i=0;i<nd;++i){int64_t d=TF_Dim(input,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* output=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  SP_Stream cstream=TF_GetStream(ctx,s);
  if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
    const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output);
    for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(tanhf(x));}
    TF_DeleteStatus(s);return;}
  auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureTanhHalfPipeline(dev);
  if(!g_tanh_h_pipeline){const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output); for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(tanhf(x));} TF_DeleteStatus(s);return;}
  const void* in_host=TF_TensorData(input);void* out_host=TF_TensorData(output);
  id<MTLBuffer> inb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],outb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents,in_host,bytes);
  id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
  [enc setComputePipelineState:g_tanh_h_pipeline];[enc setBuffer:inb offset:0 atIndex:0];[enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];
  [cb commit];[cb waitUntilCompleted];memcpy(out_host,outb.contents,bytes);TF_DeleteStatus(s);
}

extern "C" void MPSTanh_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Tanh[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int nd = TF_NumDims(input); int64_t nelems = 1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems * sizeof(float);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) out[i] = tanhf(in[i]);
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureTanhPipeline(dev);
  if (!g_tanh_pipeline) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) out[i] = tanhf(in[i]);
    TF_DeleteStatus(s); return;
  }
  const void* in_host = TF_TensorData(input); void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_tanh_pipeline]; [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted]; memcpy(out_host, outb.contents, bytes); TF_DeleteStatus(s);
}

// ===== MPS Softmax kernel (float, half, bfloat16 via MPSGraph) =====
namespace {
struct MPSSoftmaxAttrs {
  // Softmax is typically along last dimension, but we'll support any axis
  // For now, default to -1 (last dimension)
};
}

extern "C" void* MPSSoftmax_Create(TF_OpKernelConstruction* ctx) {
  return new MPSSoftmaxAttrs();
}

extern "C" void MPSSoftmax_Delete(void* kernel_ptr) {
  delete static_cast<MPSSoftmaxAttrs*>(kernel_ptr);
}

extern "C" void MPSSoftmax_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* logits = nullptr;
  TF_GetInput(ctx, 0, &logits, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(logits);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Softmax: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int nd = TF_NumDims(logits);
  if (nd < 1) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Softmax: input must have at least 1 dimension");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Get shape
  std::vector<int64_t> shape(nd);
  int64_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    shape[i] = TF_Dim(logits, i);
    total_elems *= shape[i];
  }
  
  // Allocate output with same shape
  size_t out_bytes = total_elems * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, out_bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Create NSArray for shape
    NSMutableArray* shapeArray = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [shapeArray addObject:@(shape[i])];
    }
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:shapeArray
                                                      dataType:mps_dtype
                                                          name:@"logits"];
    
    // Softmax along last axis (-1)
    MPSGraphTensor* outputTensor = [graph softMaxWithTensor:inputTensor
                                                        axis:-1
                                                        name:@"softmax"];
    
    size_t input_bytes = total_elems * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(logits)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:shapeArray
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:shapeArray
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS FusedBatchNormV3 kernel (float, half, bfloat16 via MPSGraph) =====
namespace {
struct MPSFusedBatchNormV3Attrs {
  float epsilon;
  bool is_training;
  std::string data_format;
};
}

extern "C" void* MPSFusedBatchNormV3_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSFusedBatchNormV3Attrs();
  TF_Status* s = TF_NewStatus();
  
  float epsilon = 0.0001f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "epsilon", &epsilon, s);
  if (TF_GetCode(s) == TF_OK) {
    attrs->epsilon = epsilon;
  }
  
  TF_Bool is_training = false;
  TF_OpKernelConstruction_GetAttrBool(ctx, "is_training", &is_training, s);
  if (TF_GetCode(s) == TF_OK) {
    attrs->is_training = (is_training != 0);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  } else {
    attrs->data_format = "NHWC";
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSFusedBatchNormV3_Delete(void* kernel_ptr) {
  delete static_cast<MPSFusedBatchNormV3Attrs*>(kernel_ptr);
}

extern "C" void MPSFusedBatchNormV3_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSFusedBatchNormV3Attrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  // Get inputs: x, scale, offset, mean, variance
  TF_Tensor* x = nullptr;
  TF_Tensor* scale = nullptr;
  TF_Tensor* offset = nullptr;
  TF_Tensor* mean = nullptr;
  TF_Tensor* variance = nullptr;
  
  TF_GetInput(ctx, 0, &x, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &scale, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &offset, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 3, &mean, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 4, &variance, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(x);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS FusedBatchNormV3: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int nd = TF_NumDims(x);
  if (nd != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS FusedBatchNormV3: input must be 4D (NHWC or NCHW)");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Get shape [N, H, W, C] for NHWC
  std::vector<int64_t> shape(nd);
  int64_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    shape[i] = TF_Dim(x, i);
    total_elems *= shape[i];
  }
  
  int64_t channels = (attrs->data_format == "NHWC") ? shape[3] : shape[1];
  
  // Allocate outputs: y, batch_mean, batch_variance, saved_mean, saved_variance
  TF_Tensor* y = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, total_elems * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  std::vector<int64_t> stats_shape = {channels};
  TF_Tensor* batch_mean = TF_AllocateOutput(ctx, 1, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_Tensor* batch_variance = TF_AllocateOutput(ctx, 2, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_Tensor* saved_mean = TF_AllocateOutput(ctx, 3, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_Tensor* saved_variance = TF_AllocateOutput(ctx, 4, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    NSMutableArray* shapeArray = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [shapeArray addObject:@(shape[i])];
    }
    
    NSArray* statsShapeArray = @[@(channels)];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:shapeArray
                                                      dataType:mps_dtype
                                                          name:@"input"];
    MPSGraphTensor* scaleTensor = [graph placeholderWithShape:statsShapeArray
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"scale"];
    MPSGraphTensor* offsetTensor = [graph placeholderWithShape:statsShapeArray
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"offset"];
    MPSGraphTensor* meanTensor = [graph placeholderWithShape:statsShapeArray
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"mean"];
    MPSGraphTensor* varianceTensor = [graph placeholderWithShape:statsShapeArray
                                                         dataType:MPSDataTypeFloat32
                                                             name:@"variance"];
    
    // Normalize: y = scale * (x - mean) / sqrt(variance + epsilon) + offset
    // Use MPSGraph batch normalization
    NSUInteger axis = (attrs->data_format == "NHWC") ? 3 : 1;
    
    MPSGraphTensor* normalizedTensor = [graph normalizationWithTensor:inputTensor
                                                           meanTensor:meanTensor
                                                       varianceTensor:varianceTensor
                                                          gammaTensor:scaleTensor
                                                           betaTensor:offsetTensor
                                                              epsilon:attrs->epsilon
                                                                 name:@"batch_norm"];
    
    // Prepare buffers
    size_t input_bytes = total_elems * elem_size;
    size_t stats_bytes = channels * 4;
    
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(x)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scaleBuffer = [dev newBufferWithBytes:TF_TensorData(scale)
                                                  length:stats_bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsetBuffer = [dev newBufferWithBytes:TF_TensorData(offset)
                                                   length:stats_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> meanBuffer = [dev newBufferWithBytes:TF_TensorData(mean)
                                                 length:stats_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> varianceBuffer = [dev newBufferWithBytes:TF_TensorData(variance)
                                                     length:stats_bytes
                                                    options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:input_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:shapeArray
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* scaleData = [[MPSGraphTensorData alloc] initWithMTLBuffer:scaleBuffer
                                                                             shape:statsShapeArray
                                                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* offsetData = [[MPSGraphTensorData alloc] initWithMTLBuffer:offsetBuffer
                                                                              shape:statsShapeArray
                                                                           dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* meanData = [[MPSGraphTensorData alloc] initWithMTLBuffer:meanBuffer
                                                                            shape:statsShapeArray
                                                                         dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* varianceData = [[MPSGraphTensorData alloc] initWithMTLBuffer:varianceBuffer
                                                                                shape:statsShapeArray
                                                                             dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:shapeArray
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData,
                                    scaleTensor: scaleData,
                                    offsetTensor: offsetData,
                                    meanTensor: meanData,
                                    varianceTensor: varianceData}
                   targetTensors:@[normalizedTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(y), outputBuffer.contents, input_bytes);
    
    // Copy statistics (in training mode, these would be computed; for now, copy inputs)
    memcpy(TF_TensorData(batch_mean), TF_TensorData(mean), stats_bytes);
    memcpy(TF_TensorData(batch_variance), TF_TensorData(variance), stats_bytes);
    memcpy(TF_TensorData(saved_mean), TF_TensorData(mean), stats_bytes);
    memcpy(TF_TensorData(saved_variance), TF_TensorData(variance), stats_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS Swish activation kernel (float, half, bfloat16) =====
// Swish(x) = x * sigmoid(x)
namespace {
struct MPSSwishAttrs {};
}

extern "C" void* MPSSwish_Create(TF_OpKernelConstruction* ctx) {
  return new MPSSwishAttrs();
}

extern "C" void MPSSwish_Delete(void* kernel_ptr) {
  delete static_cast<MPSSwishAttrs*>(kernel_ptr);
}

extern "C" void MPSSwish_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Swish: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int nd = TF_NumDims(input);
  std::vector<int64_t> shape(nd);
  int64_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    shape[i] = TF_Dim(input, i);
    total_elems *= shape[i];
  }
  
  size_t bytes = total_elems * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    NSMutableArray* shapeArray = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [shapeArray addObject:@(shape[i])];
    }
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:shapeArray
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    // Swish(x) = x * sigmoid(x)
    MPSGraphTensor* sigmoidTensor = [graph sigmoidWithTensor:inputTensor name:@"sigmoid"];
    MPSGraphTensor* outputTensor = [graph multiplicationWithPrimaryTensor:inputTensor
                                                          secondaryTensor:sigmoidTensor
                                                                     name:@"swish"];
    
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:shapeArray
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:shapeArray
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, bytes);
  }
}

// ===== MPS MaxPool kernel (float, half, NHWC only) =====
namespace {
struct MPSMaxPoolAttrs {
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::string padding;
  std::string data_format;
};
}

extern "C" void* MPSMaxPool_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSMaxPoolAttrs();
  TF_Status* s = TF_NewStatus();
  
  int64_t* ksize_data = nullptr;
  int ksize_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "ksize", &ksize_data, &ksize_len, s);
  if (TF_GetCode(s) == TF_OK && ksize_data && ksize_len > 0) {
    attrs->ksize.assign(ksize_data, ksize_data + ksize_len);
  }
  
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSMaxPool_Delete(void* kernel_ptr) {
  delete static_cast<MPSMaxPoolAttrs*>(kernel_ptr);
}

extern "C" void MPSMaxPool_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSMaxPoolAttrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS MaxPool: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int64_t num_dims = TF_NumDims(input);
  if (num_dims != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS MaxPool: input must be 4D");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Input: [N, H, W, C] NHWC
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C = TF_Dim(input, 3);
  
  // Get ksize and strides
  int64_t kH = 1, kW = 1;
  if (attrs->ksize.size() == 4) {
    kH = attrs->ksize[1];
    kW = attrs->ksize[2];
  }
  
  int64_t stride_h = 1, stride_w = 1;
  if (attrs->strides.size() == 4) {
    stride_h = attrs->strides[1];
    stride_w = attrs->strides[2];
  }
  
  // Compute output dimensions and padding
  int64_t pad_top = 0, pad_left = 0;
  int64_t H_out, W_out;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h = std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in);
    int64_t pad_w = std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in);
    pad_top = pad_h / 2;
    pad_left = pad_w / 2;
  } else {
    H_out = (H_in - kH) / stride_h + 1;
    W_out = (W_in - kW) / stride_w + 1;
  }
  
  int64_t output_dims[4] = {N, H_out, W_out, C};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, output_dims, 4, C * H_out * W_out * N * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C * elem_size;
  
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor alloc] init];
    desc.kernelWidth = (NSUInteger)kW;
    desc.kernelHeight = (NSUInteger)kH;
    desc.strideInX = (NSUInteger)stride_w;
    desc.strideInY = (NSUInteger)stride_h;
    desc.paddingLeft = (NSUInteger)pad_left;
    desc.paddingRight = (NSUInteger)std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in - pad_left);
    desc.paddingTop = (NSUInteger)pad_top;
    desc.paddingBottom = (NSUInteger)std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in - pad_top);
    desc.paddingStyle = MPSGraphPaddingStyleExplicit;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    
    MPSGraphTensor* outputTensor = [graph maxPooling2DWithSourceTensor:inputTensor
                                                             descriptor:desc
                                                                   name:@"maxpool"];
    
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS AvgPool kernel (float, half, NHWC only) =====
namespace {
struct MPSAvgPoolAttrs {
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::string padding;
  std::string data_format;
};
}

extern "C" void* MPSAvgPool_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSAvgPoolAttrs();
  TF_Status* s = TF_NewStatus();
  
  int64_t* ksize_data = nullptr;
  int ksize_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "ksize", &ksize_data, &ksize_len, s);
  if (TF_GetCode(s) == TF_OK && ksize_data && ksize_len > 0) {
    attrs->ksize.assign(ksize_data, ksize_data + ksize_len);
  }
  
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSAvgPool_Delete(void* kernel_ptr) {
  delete static_cast<MPSAvgPoolAttrs*>(kernel_ptr);
}

extern "C" void MPSAvgPool_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSAvgPoolAttrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS AvgPool: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int64_t num_dims = TF_NumDims(input);
  if (num_dims != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS AvgPool: input must be 4D");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Input: [N, H, W, C] NHWC
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C = TF_Dim(input, 3);
  
  // Get ksize and strides
  int64_t kH = 1, kW = 1;
  if (attrs->ksize.size() == 4) {
    kH = attrs->ksize[1];
    kW = attrs->ksize[2];
  }
  
  int64_t stride_h = 1, stride_w = 1;
  if (attrs->strides.size() == 4) {
    stride_h = attrs->strides[1];
    stride_w = attrs->strides[2];
  }
  
  // Compute output dimensions and padding
  int64_t pad_top = 0, pad_left = 0;
  int64_t H_out, W_out;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h = std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in);
    int64_t pad_w = std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in);
    pad_top = pad_h / 2;
    pad_left = pad_w / 2;
  } else {
    H_out = (H_in - kH) / stride_h + 1;
    W_out = (W_in - kW) / stride_w + 1;
  }
  
  int64_t output_dims[4] = {N, H_out, W_out, C};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, output_dims, 4, C * H_out * W_out * N * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C * elem_size;
  
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor alloc] init];
    desc.kernelWidth = (NSUInteger)kW;
    desc.kernelHeight = (NSUInteger)kH;
    desc.strideInX = (NSUInteger)stride_w;
    desc.strideInY = (NSUInteger)stride_h;
    desc.paddingLeft = (NSUInteger)pad_left;
    desc.paddingRight = (NSUInteger)std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in - pad_left);
    desc.paddingTop = (NSUInteger)pad_top;
    desc.paddingBottom = (NSUInteger)std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in - pad_top);
    desc.paddingStyle = MPSGraphPaddingStyleExplicit;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    
    MPSGraphTensor* outputTensor = [graph avgPooling2DWithSourceTensor:inputTensor
                                                             descriptor:desc
                                                                   name:@"avgpool"];
    
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

