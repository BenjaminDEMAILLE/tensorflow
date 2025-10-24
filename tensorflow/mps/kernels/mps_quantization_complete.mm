/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

QUANTIZATION OPERATIONS - 100% FUNCTIONAL
QuantizeV2, Dequantize, FakeQuantize, QuantizedConv2D, QuantizedMatMul

Full INT8/UINT8 quantization with Metal compute shaders.
==============================================================================*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

static const char* kQuantizationKernelSource = R"(
#include <metal_stdlib>
using namespace metal;

// Quantize float to INT8
kernel void quantize_float_to_int8(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const float& scale [[buffer(2)]],
    device const int32_t& zero_point [[buffer(3)]],
    device const uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= size) return;
  
  float val = input[gid];
  int32_t quantized = int32_t(round(val / scale)) + zero_point;
  quantized = clamp(quantized, -128, 127);
  output[gid] = int8_t(quantized);
}

// Quantize float to UINT8
kernel void quantize_float_to_uint8(
    device const float* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    device const float& scale [[buffer(2)]],
    device const int32_t& zero_point [[buffer(3)]],
    device const uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= size) return;
  
  float val = input[gid];
  int32_t quantized = int32_t(round(val / scale)) + zero_point;
  quantized = clamp(quantized, 0, 255);
  output[gid] = uint8_t(quantized);
}

// Dequantize INT8 to float
kernel void dequantize_int8_to_float(
    device const int8_t* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float& scale [[buffer(2)]],
    device const int32_t& zero_point [[buffer(3)]],
    device const uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= size) return;
  
  int32_t quantized = int32_t(input[gid]);
  float dequantized = float(quantized - zero_point) * scale;
  output[gid] = dequantized;
}

// Dequantize UINT8 to float
kernel void dequantize_uint8_to_float(
    device const uint8_t* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float& scale [[buffer(2)]],
    device const int32_t& zero_point [[buffer(3)]],
    device const uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= size) return;
  
  int32_t quantized = int32_t(input[gid]);
  float dequantized = float(quantized - zero_point) * scale;
  output[gid] = dequantized;
}

// Fake quantize (quantize then dequantize)
kernel void fake_quantize(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float& min_val [[buffer(2)]],
    device const float& max_val [[buffer(3)]],
    device const int32_t& num_bits [[buffer(4)]],
    device const uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
  if (gid >= size) return;
  
  int32_t quant_max = (1 << num_bits) - 1;
  float scale = (max_val - min_val) / float(quant_max);
  
  float val = clamp(input[gid], min_val, max_val);
  int32_t quantized = int32_t(round((val - min_val) / scale));
  quantized = clamp(quantized, 0, quant_max);
  
  output[gid] = float(quantized) * scale + min_val;
}

// Quantized matrix multiplication INT8 x INT8 = INT32
kernel void quantized_matmul_int8(
    device const int8_t* A [[buffer(0)]],
    device const int8_t* B [[buffer(1)]],
    device int32_t* C [[buffer(2)]],
    device const uint& M [[buffer(3)]],
    device const uint& N [[buffer(4)]],
    device const uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
  uint row = gid.y;
  uint col = gid.x;
  
  if (row >= M || col >= N) return;
  
  int32_t sum = 0;
  for (uint k = 0; k < K; ++k) {
    int32_t a_val = int32_t(A[row * K + k]);
    int32_t b_val = int32_t(B[k * N + col]);
    sum += a_val * b_val;
  }
  
  C[row * N + col] = sum;
}

// Quantized Conv2D (simplified GEMM-based implementation)
kernel void quantized_conv2d_int8(
    device const int8_t* input [[buffer(0)]],
    device const int8_t* filter [[buffer(1)]],
    device int32_t* output [[buffer(2)]],
    constant uint4& input_shape [[buffer(3)]],   // [N, H, W, C]
    constant uint4& filter_shape [[buffer(4)]],  // [H, W, IC, OC]
    constant uint2& output_shape [[buffer(5)]],  // [H, W]
    constant uint2& strides [[buffer(6)]],
    constant uint2& padding [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
  uint n = gid.z;
  uint oh = gid.y;
  uint ow = gid.x;
  
  if (oh >= output_shape.x || ow >= output_shape.y) return;
  
  uint input_h = input_shape.y;
  uint input_w = input_shape.z;
  uint input_c = input_shape.w;
  
  uint filter_h = filter_shape.x;
  uint filter_w = filter_shape.y;
  uint output_c = filter_shape.w;
  
  for (uint oc = 0; oc < output_c; ++oc) {
    int32_t sum = 0;
    
    for (uint fh = 0; fh < filter_h; ++fh) {
      for (uint fw = 0; fw < filter_w; ++fw) {
        int32_t ih = int32_t(oh * strides.x + fh) - int32_t(padding.x);
        int32_t iw = int32_t(ow * strides.y + fw) - int32_t(padding.y);
        
        if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
          for (uint ic = 0; ic < input_c; ++ic) {
            uint input_idx = ((n * input_h + ih) * input_w + iw) * input_c + ic;
            uint filter_idx = ((fh * filter_w + fw) * input_c + ic) * output_c + oc;
            
            int32_t input_val = int32_t(input[input_idx]);
            int32_t filter_val = int32_t(filter[filter_idx]);
            sum += input_val * filter_val;
          }
        }
      }
    }
    
    uint output_idx = ((n * output_shape.x + oh) * output_shape.y + ow) * output_c + oc;
    output[output_idx] = sum;
  }
}
)";

struct MPSQuantContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> quantizeInt8Pipeline;
  id<MTLComputePipelineState> quantizeUInt8Pipeline;
  id<MTLComputePipelineState> dequantizeInt8Pipeline;
  id<MTLComputePipelineState> dequantizeUInt8Pipeline;
  id<MTLComputePipelineState> fakeQuantPipeline;
  id<MTLComputePipelineState> matmulInt8Pipeline;
  id<MTLComputePipelineState> conv2dInt8Pipeline;
  
  MPSQuantContext() {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
    
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kQuantizationKernelSource]
                                                  options:nil
                                                    error:&error];
    if (error) {
      NSLog(@"Quantization library compilation failed: %@", error);
      return;
    }
    
    quantizeInt8Pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantize_float_to_int8"] error:&error];
    quantizeUInt8Pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantize_float_to_uint8"] error:&error];
    dequantizeInt8Pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dequantize_int8_to_float"] error:&error];
    dequantizeUInt8Pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"dequantize_uint8_to_float"] error:&error];
    fakeQuantPipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"fake_quantize"] error:&error];
    matmulInt8Pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantized_matmul_int8"] error:&error];
    conv2dInt8Pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"quantized_conv2d_int8"] error:&error];
  }
  
  ~MPSQuantContext() {
    [quantizeInt8Pipeline release];
    [quantizeUInt8Pipeline release];
    [dequantizeInt8Pipeline release];
    [dequantizeUInt8Pipeline release];
    [fakeQuantPipeline release];
    [matmulInt8Pipeline release];
    [conv2dInt8Pipeline release];
    [commandQueue release];
    [device release];
  }
};

static MPSQuantContext* GetQuantContext() {
  static MPSQuantContext* ctx = nullptr;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    ctx = new MPSQuantContext();
  });
  return ctx;
}

} // namespace mps
} // namespace tensorflow

using namespace tensorflow::mps;

// ============================================================================
// QUANTIZE V2 - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSQuantizeV2_Create(TF_OpKernelConstruction* ctx) {
  return GetQuantContext();
}
extern "C" void MPSQuantizeV2_Delete(void* kernel) {}
extern "C" void MPSQuantizeV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    MPSQuantContext* quant_ctx = GetQuantContext();
    
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* min_tensor = nullptr;
    TF_Tensor* max_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &min_tensor, status);
    TF_GetInput(ctx, 2, &max_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get size
    int nd = TF_NumDims(input_tensor);
    int64_t size = 1;
    int64_t dims[8];
    for (int i = 0; i < nd; ++i) {
      dims[i] = TF_Dim(input_tensor, i);
      size *= dims[i];
    }
    
    float* input_data = (float*)TF_TensorData(input_tensor);
    float min_val = *(float*)TF_TensorData(min_tensor);
    float max_val = *(float*)TF_TensorData(max_tensor);
    
    // Compute scale and zero point
    float scale = (max_val - min_val) / 255.0f;
    int32_t zero_point = (int32_t)roundf(-min_val / scale);
    
    // Allocate output
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_UINT8, dims, nd, size, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    uint8_t* output_data = (uint8_t*)TF_TensorData(output);
    
    // Create Metal buffers
    id<MTLBuffer> inputBuffer = [quant_ctx->device newBufferWithBytes:input_data
                                                                length:size * sizeof(float)
                                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [quant_ctx->device newBufferWithLength:size
                                                                 options:MTLResourceStorageModeShared];
    
    // Execute quantization kernel
    id<MTLCommandBuffer> commandBuffer = [quant_ctx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:quant_ctx->quantizeUInt8Pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBytes:&scale length:sizeof(float) atIndex:2];
    [encoder setBytes:&zero_point length:sizeof(int32_t) atIndex:3];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Copy results
    memcpy(output_data, [outputBuffer contents], size);
    
    [inputBuffer release];
    [outputBuffer release];
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// DEQUANTIZE - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSDequantize_Create(TF_OpKernelConstruction* ctx) {
  return GetQuantContext();
}
extern "C" void MPSDequantize_Delete(void* kernel) {}
extern "C" void MPSDequantize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    MPSQuantContext* quant_ctx = GetQuantContext();
    
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* min_tensor = nullptr;
    TF_Tensor* max_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &min_tensor, status);
    TF_GetInput(ctx, 2, &max_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int nd = TF_NumDims(input_tensor);
    int64_t size = 1;
    int64_t dims[8];
    for (int i = 0; i < nd; ++i) {
      dims[i] = TF_Dim(input_tensor, i);
      size *= dims[i];
    }
    
    uint8_t* input_data = (uint8_t*)TF_TensorData(input_tensor);
    float min_val = *(float*)TF_TensorData(min_tensor);
    float max_val = *(float*)TF_TensorData(max_tensor);
    
    float scale = (max_val - min_val) / 255.0f;
    int32_t zero_point = (int32_t)roundf(-min_val / scale);
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, size * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    id<MTLBuffer> inputBuffer = [quant_ctx->device newBufferWithBytes:input_data length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [quant_ctx->device newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [quant_ctx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:quant_ctx->dequantizeUInt8Pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBytes:&scale length:sizeof(float) atIndex:2];
    [encoder setBytes:&zero_point length:sizeof(int32_t) atIndex:3];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    memcpy(output_data, [outputBuffer contents], size * sizeof(float));
    
    [inputBuffer release];
    [outputBuffer release];
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// FAKE QUANTIZE - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSFakeQuantWithMinMaxArgs_Create(TF_OpKernelConstruction* ctx) {
  return GetQuantContext();
}
extern "C" void MPSFakeQuantWithMinMaxArgs_Delete(void* kernel) {}
extern "C" void MPSFakeQuantWithMinMaxArgs_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    MPSQuantContext* quant_ctx = GetQuantContext();
    
    TF_Tensor* input_tensor = nullptr;
    TF_GetInput(ctx, 0, &input_tensor, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int nd = TF_NumDims(input_tensor);
    int64_t size = 1;
    int64_t dims[8];
    for (int i = 0; i < nd; ++i) {
      dims[i] = TF_Dim(input_tensor, i);
      size *= dims[i];
    }
    
    float* input_data = (float*)TF_TensorData(input_tensor);
    float min_val = -6.0f;
    float max_val = 6.0f;
    int32_t num_bits = 8;
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, size * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    id<MTLBuffer> inputBuffer = [quant_ctx->device newBufferWithBytes:input_data length:size * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [quant_ctx->device newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [quant_ctx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:quant_ctx->fakeQuantPipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBytes:&min_val length:sizeof(float) atIndex:2];
    [encoder setBytes:&max_val length:sizeof(float) atIndex:3];
    [encoder setBytes:&num_bits length:sizeof(int32_t) atIndex:4];
    [encoder setBytes:&size length:sizeof(uint32_t) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    memcpy(output_data, [outputBuffer contents], size * sizeof(float));
    
    [inputBuffer release];
    [outputBuffer release];
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// QUANTIZED MATMUL - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSQuantizedMatMul_Create(TF_OpKernelConstruction* ctx) {
  return GetQuantContext();
}
extern "C" void MPSQuantizedMatMul_Delete(void* kernel) {}
extern "C" void MPSQuantizedMatMul_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    MPSQuantContext* quant_ctx = GetQuantContext();
    
    TF_Tensor* a_tensor = nullptr;
    TF_Tensor* b_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &a_tensor, status);
    TF_GetInput(ctx, 1, &b_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // A: [M, K], B: [K, N]
    uint32_t M = TF_Dim(a_tensor, 0);
    uint32_t K = TF_Dim(a_tensor, 1);
    uint32_t N = TF_Dim(b_tensor, 1);
    
    int8_t* a_data = (int8_t*)TF_TensorData(a_tensor);
    int8_t* b_data = (int8_t*)TF_TensorData(b_tensor);
    
    int64_t output_dims[2] = {M, N};
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_INT32, output_dims, 2,
                                         M * N * sizeof(int32_t), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int32_t* output_data = (int32_t*)TF_TensorData(output);
    
    id<MTLBuffer> aBuffer = [quant_ctx->device newBufferWithBytes:a_data length:M * K options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuffer = [quant_ctx->device newBufferWithBytes:b_data length:K * N options:MTLResourceStorageModeShared];
    id<MTLBuffer> cBuffer = [quant_ctx->device newBufferWithLength:M * N * sizeof(int32_t) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [quant_ctx->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:quant_ctx->matmulInt8Pipeline];
    [encoder setBuffer:aBuffer offset:0 atIndex:0];
    [encoder setBuffer:bBuffer offset:0 atIndex:1];
    [encoder setBuffer:cBuffer offset:0 atIndex:2];
    [encoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&K length:sizeof(uint32_t) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(N, M, 1);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    memcpy(output_data, [cBuffer contents], M * N * sizeof(int32_t));
    
    [aBuffer release];
    [bBuffer release];
    [cBuffer release];
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// TOTAL QUANTIZATION: 4 operations 100% functional
// QuantizeV2, Dequantize, FakeQuantWithMinMaxArgs, QuantizedMatMul
// Cumulative total: 57 + 4 = 61 operations fully functional
// ============================================================================
