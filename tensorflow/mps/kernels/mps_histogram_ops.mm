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

// Histogram and binning operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include <algorithm>
#include <vector>
#include <Metal/Metal.h>

namespace {
struct MPSHistogramGPUContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
};

MPSHistogramGPUContext* CreateHistogramPipeline() {
  auto* c = new MPSHistogramGPUContext();
  @autoreleasepool {
    c->device = MTLCreateSystemDefaultDevice();
    c->queue = [c->device newCommandQueue];
    NSError* error = nil;
    NSString* src = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void histogram_fixed_width(device const float* values [[buffer(0)]],
                                  device atomic_int* hist [[buffer(1)]],
                                  constant float& min_val [[buffer(2)]],
                                  constant float& max_val [[buffer(3)]],
                                  constant int& nbins [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
  float v = values[gid];
  if (v < min_val || v > max_val || nbins <= 0) return;
  float bw = (max_val - min_val) / float(nbins);
  // Handle edge case max -> last bin
  int bin = (v == max_val) ? (nbins - 1) : int(floor((v - min_val) / bw));
  bin = clamp(bin, 0, nbins - 1);
  atomic_fetch_add_explicit(&(hist[bin]), 1, memory_order_relaxed);
}
    )";
    id<MTLLibrary> lib = [c->device newLibraryWithSource:src options:nil error:&error];
    if (lib) {
      id<MTLFunction> f = [lib newFunctionWithName:@"histogram_fixed_width"];
      c->pipeline = [c->device newComputePipelineStateWithFunction:f error:&error];
      [f release];
      [lib release];
    }
  }
  return c;
}
}

// ===== HistogramFixedWidth =====
typedef struct {
  int32_t nbins;
  MPSHistogramGPUContext* gpu;
} MPSHistogramContext;

extern "C" void* MPSHistogramFixedWidth_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* kernel_ctx = new MPSHistogramContext;
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "nbins", &kernel_ctx->nbins, status);
  kernel_ctx->gpu = CreateHistogramPipeline();
  
  if (TF_GetCode(status) != TF_OK) {
    delete kernel_ctx;
    kernel_ctx = nullptr;
  }
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSHistogramFixedWidth_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSHistogramContext*>(kernel);
    if (k->gpu) {
      if (k->gpu->pipeline) [k->gpu->pipeline release];
      if (k->gpu->queue) [k->gpu->queue release];
      if (k->gpu->device) [k->gpu->device release];
      delete k->gpu;
    }
    delete k;
  }
}

extern "C" void MPSHistogramFixedWidth_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSHistogramContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  // Get inputs: values and value_range
  TF_Tensor* values = nullptr;
  TF_Tensor* value_range = nullptr;
  TF_GetInput(ctx, 0, &values, status);
  TF_GetInput(ctx, 1, &value_range, status);
  
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  // Get data
  const float* vals = (const float*)TF_TensorData(values);
  const float* range = (const float*)TF_TensorData(value_range);
  
  int nd_vals = TF_NumDims(values);
  int64_t nelems = 1;
  for (int i = 0; i < nd_vals; ++i) {
    nelems *= TF_Dim(values, i);
  }
  
  float min_val = range[0];
  float max_val = range[1];
  int32_t nbins = kernel_ctx->nbins;
  
  // Allocate output histogram
  int64_t out_dims[1] = {nbins};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_INT32, out_dims, 1, nbins * sizeof(int32_t), status);
  int32_t* hist = (int32_t*)TF_TensorData(output);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  // Zero-initialize
  for (int32_t i = 0; i < nbins; ++i) hist[i] = 0;
  
  // GPU compute
  @autoreleasepool {
    auto* gpu = kernel_ctx->gpu;
    size_t values_bytes = nelems * sizeof(float);
    id<MTLBuffer> vbuf = [gpu->device newBufferWithBytes:vals length:values_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> hbuf = [gpu->device newBufferWithBytes:hist length:nbins*sizeof(int32_t) options:MTLResourceStorageModeShared];
    float min_v = min_val, max_v = max_val;
    int nb = nbins;
    id<MTLBuffer> minBuf = [gpu->device newBufferWithBytes:&min_v length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> maxBuf = [gpu->device newBufferWithBytes:&max_v length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> nbBuf = [gpu->device newBufferWithBytes:&nb length:sizeof(int) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> cb = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gpu->pipeline];
    [enc setBuffer:vbuf offset:0 atIndex:0];
    [enc setBuffer:hbuf offset:0 atIndex:1];
    [enc setBuffer:minBuf offset:0 atIndex:2];
    [enc setBuffer:maxBuf offset:0 atIndex:3];
    [enc setBuffer:nbBuf offset:0 atIndex:4];
    NSUInteger tg = gpu->pipeline.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreads:MTLSizeMake(nelems,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(hist, hbuf.contents, nbins*sizeof(int32_t));
    [vbuf release]; [hbuf release]; [minBuf release]; [maxBuf release]; [nbBuf release];
  }
  
  TF_DeleteStatus(status);
}

// ===== Bincount =====
extern "C" void* MPSBincount_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBincount_Delete(void* kernel) {}

extern "C" void MPSBincount_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Bincount requires CPU counting logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DenseBincount =====
extern "C" void* MPSDenseBincount_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDenseBincount_Delete(void* kernel) {}

extern "C" void MPSDenseBincount_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DenseBincount requires CPU counting logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SparseBincount =====
extern "C" void* MPSSparseBincount_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSparseBincount_Delete(void* kernel) {}

extern "C" void MPSSparseBincount_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SparseBincount requires sparse tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RaggedBincount =====
extern "C" void* MPSRaggedBincount_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRaggedBincount_Delete(void* kernel) {}

extern "C" void MPSRaggedBincount_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RaggedBincount requires ragged tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
