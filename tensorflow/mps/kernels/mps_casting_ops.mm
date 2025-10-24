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

// Casting and type conversion operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include <cstring>

// ===== Cast (comprehensive type conversion) =====
struct MPSCastContext {
  TF_DataType src_type;
  TF_DataType dst_type;
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> castPipeline;
};

extern "C" void* MPSCast_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* kernel_ctx = new MPSCastContext;
  
  TF_OpKernelConstruction_GetAttrType(ctx, "SrcT", &kernel_ctx->src_type, status);
  TF_OpKernelConstruction_GetAttrType(ctx, "DstT", &kernel_ctx->dst_type, status);
  
  @autoreleasepool {
    kernel_ctx->device = MTLCreateSystemDefaultDevice();
    kernel_ctx->commandQueue = [kernel_ctx->device newCommandQueue];
    
    NSError* error = nil;
    NSString* shaderSource = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void cast_float_to_int32(device const float* in [[buffer(0)]],
                                 device int* out [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
  out[gid] = (int)in[gid];
}

kernel void cast_int32_to_float(device const int* in [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
  out[gid] = (float)in[gid];
}

kernel void cast_float_to_int64(device const float* in [[buffer(0)]],
                                 device long* out [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
  out[gid] = (long)in[gid];
}

kernel void cast_int64_to_float(device const long* in [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
  out[gid] = (float)in[gid];
}

kernel void cast_bool_to_float(device const bool* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
  out[gid] = in[gid] ? 1.0f : 0.0f;
}

kernel void cast_float_to_bool(device const float* in [[buffer(0)]],
                                device bool* out [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
  out[gid] = (in[gid] != 0.0f);
}
)";
    
    id<MTLLibrary> lib = [kernel_ctx->device newLibraryWithSource:shaderSource options:nil error:&error];
    if (lib) {
      NSString* kernelName = @"";
      if (kernel_ctx->src_type == TF_FLOAT && kernel_ctx->dst_type == TF_INT32) {
        kernelName = @"cast_float_to_int32";
      } else if (kernel_ctx->src_type == TF_INT32 && kernel_ctx->dst_type == TF_FLOAT) {
        kernelName = @"cast_int32_to_float";
      } else if (kernel_ctx->src_type == TF_FLOAT && kernel_ctx->dst_type == TF_INT64) {
        kernelName = @"cast_float_to_int64";
      } else if (kernel_ctx->src_type == TF_INT64 && kernel_ctx->dst_type == TF_FLOAT) {
        kernelName = @"cast_int64_to_float";
      } else if (kernel_ctx->src_type == TF_BOOL && kernel_ctx->dst_type == TF_FLOAT) {
        kernelName = @"cast_bool_to_float";
      } else if (kernel_ctx->src_type == TF_FLOAT && kernel_ctx->dst_type == TF_BOOL) {
        kernelName = @"cast_float_to_bool";
      }
      
      if (kernelName.length > 0) {
        id<MTLFunction> func = [lib newFunctionWithName:kernelName];
        kernel_ctx->castPipeline = [kernel_ctx->device newComputePipelineStateWithFunction:func error:&error];
        [func release];
      }
      [lib release];
    }
  }
  
  if (TF_GetCode(status) != TF_OK) {
    if (kernel_ctx->castPipeline) [kernel_ctx->castPipeline release];
    if (kernel_ctx->commandQueue) [kernel_ctx->commandQueue release];
    if (kernel_ctx->device) [kernel_ctx->device release];
    delete kernel_ctx;
    kernel_ctx = nullptr;
  }
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSCast_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSCastContext*>(kernel);
    if (k->castPipeline) [k->castPipeline release];
    if (k->commandQueue) [k->commandQueue release];
    if (k->device) [k->device release];
    delete k;
  }
}

extern "C" void MPSCast_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSCastContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  int nd = TF_NumDims(input);
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(input, i); nelems *= dims[i]; }
  
  size_t dst_size = TF_DataTypeSize(kernel_ctx->dst_type);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, kernel_ctx->dst_type, dims, nd, nelems * dst_size, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  if (!kernel_ctx->castPipeline) {
    TF_SetStatus(status, TF_UNIMPLEMENTED, "Cast: Unsupported type conversion");
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  size_t src_size = TF_DataTypeSize(kernel_ctx->src_type);
  id<MTLBuffer> inBuf = [kernel_ctx->device newBufferWithBytes:TF_TensorData(input) length:nelems*src_size options:MTLResourceStorageModeShared];
  id<MTLBuffer> outBuf = [kernel_ctx->device newBufferWithLength:nelems*dst_size options:MTLResourceStorageModeShared];
  
  id<MTLCommandBuffer> cb = [kernel_ctx->commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
  [encoder setComputePipelineState:kernel_ctx->castPipeline];
  [encoder setBuffer:inBuf offset:0 atIndex:0];
  [encoder setBuffer:outBuf offset:0 atIndex:1];
  NSUInteger tgSize = kernel_ctx->castPipeline.maxTotalThreadsPerThreadgroup;
  [encoder dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [encoder endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  
  memcpy(TF_TensorData(output), outBuf.contents, nelems*dst_size);
  [inBuf release]; [outBuf release];
  }
  
  TF_DeleteStatus(status);
}

// ===== Bitcast =====
extern "C" void* MPSBitcast_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBitcast_Delete(void* kernel) {}

extern "C" void MPSBitcast_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Bitcast requires type reinterpretation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ComplexAbs =====
extern "C" void* MPSComplexAbs_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSComplexAbs_Delete(void* kernel) {}

extern "C" void MPSComplexAbs_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ComplexAbs requires complex number support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Real =====
extern "C" void* MPSReal_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSReal_Delete(void* kernel) {}

extern "C" void MPSReal_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Real requires complex number support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Imag =====
extern "C" void* MPSImag_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSImag_Delete(void* kernel) {}

extern "C" void MPSImag_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Imag requires complex number support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Complex =====
extern "C" void* MPSComplex_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSComplex_Delete(void* kernel) {}

extern "C" void MPSComplex_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Complex requires complex number support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Conj =====
extern "C" void* MPSConj_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSConj_Delete(void* kernel) {}

extern "C" void MPSConj_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Conj requires complex number support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SaturateCast =====
extern "C" void* MPSSaturateCast_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSaturateCast_Delete(void* kernel) {}

extern "C" void MPSSaturateCast_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SaturateCast requires clamping logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
