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

// Bitwise operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <Metal/Metal.h>

namespace {
id<MTLDevice> GetMetalDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

id<MTLCommandQueue> GetCommandQueue() {
  static id<MTLCommandQueue> queue = [GetMetalDevice() newCommandQueue];
  return queue;
}

struct MPSBitwiseContext {
  id<MTLComputePipelineState> pipeline_i32;
  id<MTLComputePipelineState> pipeline_i64;
};

MPSBitwiseContext* CreateBitwiseKernel(const char* kernelNameI32, const char* kernelNameI64, const char* shaderCode) {
  auto* ctx = new MPSBitwiseContext();
  @autoreleasepool {
    NSError* error = nil;
    id<MTLLibrary> lib = [GetMetalDevice() newLibraryWithSource:[NSString stringWithUTF8String:shaderCode] options:nil error:&error];
    if (lib) {
      id<MTLFunction> func32 = [lib newFunctionWithName:[NSString stringWithUTF8String:kernelNameI32]];
      if (func32) {
        ctx->pipeline_i32 = [GetMetalDevice() newComputePipelineStateWithFunction:func32 error:&error];
        [func32 release];
      }
      id<MTLFunction> func64 = [lib newFunctionWithName:[NSString stringWithUTF8String:kernelNameI64]];
      if (func64) {
        ctx->pipeline_i64 = [GetMetalDevice() newComputePipelineStateWithFunction:func64 error:&error];
        [func64 release];
      }
      [lib release];
    }
  }
  return ctx;
}
}

// ===== BitwiseAnd =====
extern "C" void* MPSBitwiseAnd_Create(TF_OpKernelConstruction* ctx) {
  const char* shader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void bitwise_and_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device int* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] & b[gid]; }
kernel void bitwise_and_i64(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]], device long* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] & b[gid]; }
)";
  return CreateBitwiseKernel("bitwise_and_i32", "bitwise_and_i64", shader);
}

extern "C" void MPSBitwiseAnd_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSBitwiseContext*>(kernel);
    if (k->pipeline_i32) [k->pipeline_i32 release];
    if (k->pipeline_i64) [k->pipeline_i64 release];
    delete k;
  }
}

extern "C" void MPSBitwiseAnd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<MPSBitwiseContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* a = nullptr;
  TF_GetInput(ctx, 0, &a, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 1, &b, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(a, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(a);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  id<MTLBuffer> aBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(a) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(b) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> oBuf = [GetMetalDevice() newBufferWithLength:nelems*elem_size options:MTLResourceStorageModeShared];
  
  id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> pipe = (dtype == TF_INT32) ? k->pipeline_i32 : k->pipeline_i64;
  [enc setComputePipelineState:pipe];
  [enc setBuffer:aBuf offset:0 atIndex:0];
  [enc setBuffer:bBuf offset:0 atIndex:1];
  [enc setBuffer:oBuf offset:0 atIndex:2];
  NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  
  memcpy(TF_TensorData(out), oBuf.contents, nelems*elem_size);
  [aBuf release]; [bBuf release]; [oBuf release];
  }
  
  TF_DeleteStatus(status);
}

// ===== BitwiseOr =====
extern "C" void* MPSBitwiseOr_Create(TF_OpKernelConstruction* ctx) {
  const char* shader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void bitwise_or_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device int* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] | b[gid]; }
kernel void bitwise_or_i64(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]], device long* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] | b[gid]; }
)";
  return CreateBitwiseKernel("bitwise_or_i32", "bitwise_or_i64", shader);
}

extern "C" void MPSBitwiseOr_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSBitwiseContext*>(kernel);
    if (k->pipeline_i32) [k->pipeline_i32 release];
    if (k->pipeline_i64) [k->pipeline_i64 release];
    delete k;
  }
}

extern "C" void MPSBitwiseOr_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* a = nullptr;
  TF_GetInput(ctx, 0, &a, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 1, &b, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  auto* k = static_cast<MPSBitwiseContext*>(kernel);
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(a, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(a);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  id<MTLBuffer> aBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(a) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(b) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> oBuf = [GetMetalDevice() newBufferWithLength:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> pipe = (dtype == TF_INT32) ? k->pipeline_i32 : k->pipeline_i64;
  [enc setComputePipelineState:pipe];
  [enc setBuffer:aBuf offset:0 atIndex:0];
  [enc setBuffer:bBuf offset:0 atIndex:1];
  [enc setBuffer:oBuf offset:0 atIndex:2];
  NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(TF_TensorData(out), oBuf.contents, nelems*elem_size);
  [aBuf release]; [bBuf release]; [oBuf release];
  
  TF_DeleteStatus(status);
}

// ===== BitwiseXor =====
extern "C" void* MPSBitwiseXor_Create(TF_OpKernelConstruction* ctx) {
  const char* shader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void bitwise_xor_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device int* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] ^ b[gid]; }
kernel void bitwise_xor_i64(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]], device long* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] ^ b[gid]; }
)";
  return CreateBitwiseKernel("bitwise_xor_i32", "bitwise_xor_i64", shader);
}

extern "C" void MPSBitwiseXor_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSBitwiseContext*>(kernel);
    if (k->pipeline_i32) [k->pipeline_i32 release];
    if (k->pipeline_i64) [k->pipeline_i64 release];
    delete k;
  }
}

extern "C" void MPSBitwiseXor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* a = nullptr;
  TF_GetInput(ctx, 0, &a, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 1, &b, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  auto* k = static_cast<MPSBitwiseContext*>(kernel);
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(a, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(a);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  id<MTLBuffer> aBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(a) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(b) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> oBuf = [GetMetalDevice() newBufferWithLength:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> pipe = (dtype == TF_INT32) ? k->pipeline_i32 : k->pipeline_i64;
  [enc setComputePipelineState:pipe];
  [enc setBuffer:aBuf offset:0 atIndex:0];
  [enc setBuffer:bBuf offset:0 atIndex:1];
  [enc setBuffer:oBuf offset:0 atIndex:2];
  NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(TF_TensorData(out), oBuf.contents, nelems*elem_size);
  [aBuf release]; [bBuf release]; [oBuf release];
  
  TF_DeleteStatus(status);
}

// ===== Invert =====
extern "C" void* MPSInvert_Create(TF_OpKernelConstruction* ctx) {
  const char* shader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void invert_i32(device const int* a [[buffer(0)]], device int* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) { out[gid] = ~a[gid]; }
kernel void invert_i64(device const long* a [[buffer(0)]], device long* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) { out[gid] = ~a[gid]; }
)";
  return CreateBitwiseKernel("invert_i32", "invert_i64", shader);
}

extern "C" void MPSInvert_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSBitwiseContext*>(kernel);
    if (k->pipeline_i32) [k->pipeline_i32 release];
    if (k->pipeline_i64) [k->pipeline_i64 release];
    delete k;
  }
}

extern "C" void MPSInvert_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  auto* k = static_cast<MPSBitwiseContext*>(kernel);
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(input);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  id<MTLBuffer> aBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> oBuf = [GetMetalDevice() newBufferWithLength:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> pipe = (dtype == TF_INT32) ? k->pipeline_i32 : k->pipeline_i64;
  [enc setComputePipelineState:pipe];
  [enc setBuffer:aBuf offset:0 atIndex:0];
  [enc setBuffer:oBuf offset:0 atIndex:1];
  NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(TF_TensorData(out), oBuf.contents, nelems*elem_size);
  [aBuf release]; [oBuf release];
  
  TF_DeleteStatus(status);
}

// ===== LeftShift =====
extern "C" void* MPSLeftShift_Create(TF_OpKernelConstruction* ctx) {
  const char* shader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void lshift_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device int* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] << b[gid]; }
kernel void lshift_i64(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]], device long* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] << b[gid]; }
)";
  return CreateBitwiseKernel("lshift_i32", "lshift_i64", shader);
}

extern "C" void MPSLeftShift_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSBitwiseContext*>(kernel);
    if (k->pipeline_i32) [k->pipeline_i32 release];
    if (k->pipeline_i64) [k->pipeline_i64 release];
    delete k;
  }
}

extern "C" void MPSLeftShift_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* y = nullptr;
  TF_GetInput(ctx, 1, &y, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  auto* k = static_cast<MPSBitwiseContext*>(kernel);
  int nd = TF_NumDims(x);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(x, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(x);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  id<MTLBuffer> aBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(x) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(y) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> oBuf = [GetMetalDevice() newBufferWithLength:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> pipe = (dtype == TF_INT32) ? k->pipeline_i32 : k->pipeline_i64;
  [enc setComputePipelineState:pipe];
  [enc setBuffer:aBuf offset:0 atIndex:0];
  [enc setBuffer:bBuf offset:0 atIndex:1];
  [enc setBuffer:oBuf offset:0 atIndex:2];
  NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(TF_TensorData(out), oBuf.contents, nelems*elem_size);
  [aBuf release]; [bBuf release]; [oBuf release];
  
  TF_DeleteStatus(status);
}

// ===== RightShift =====
extern "C" void* MPSRightShift_Create(TF_OpKernelConstruction* ctx) {
  const char* shader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void rshift_i32(device const int* a [[buffer(0)]], device const int* b [[buffer(1)]], device int* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] >> b[gid]; }
kernel void rshift_i64(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]], device long* out [[buffer(2)]], uint gid [[thread_position_in_grid]]) { out[gid] = a[gid] >> b[gid]; }
)";
  return CreateBitwiseKernel("rshift_i32", "rshift_i64", shader);
}

extern "C" void MPSRightShift_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSBitwiseContext*>(kernel);
    if (k->pipeline_i32) [k->pipeline_i32 release];
    if (k->pipeline_i64) [k->pipeline_i64 release];
    delete k;
  }
}

extern "C" void MPSRightShift_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* y = nullptr;
  TF_GetInput(ctx, 1, &y, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  auto* k = static_cast<MPSBitwiseContext*>(kernel);
  int nd = TF_NumDims(x);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(x, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(x);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  id<MTLBuffer> aBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(x) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> bBuf = [GetMetalDevice() newBufferWithBytes:TF_TensorData(y) length:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> oBuf = [GetMetalDevice() newBufferWithLength:nelems*elem_size options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> pipe = (dtype == TF_INT32) ? k->pipeline_i32 : k->pipeline_i64;
  [enc setComputePipelineState:pipe];
  [enc setBuffer:aBuf offset:0 atIndex:0];
  [enc setBuffer:bBuf offset:0 atIndex:1];
  [enc setBuffer:oBuf offset:0 atIndex:2];
  NSUInteger tgSize = pipe.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(TF_TensorData(out), oBuf.contents, nelems*elem_size);
  [aBuf release]; [bBuf release]; [oBuf release];
  
  TF_DeleteStatus(status);
}
