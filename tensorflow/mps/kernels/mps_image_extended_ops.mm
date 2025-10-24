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

// Extended image operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

// ===== ExtractImagePatches =====
struct MPSExtractPatchesCtx {
  std::vector<int32_t> ksizes;
  std::vector<int32_t> strides;
  std::vector<int32_t> rates;
  std::string padding;
};

extern "C" void* MPSExtractPatches_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSExtractPatchesCtx();
  TF_Status* status = TF_NewStatus();
  
  int32_t ksizes[4];
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "ksizes", ksizes, 4, status);
  if (TF_GetCode(status) == TF_OK) {
    kernel_ctx->ksizes.assign(ksizes, ksizes + 4);
  }
  
  int32_t strides[4];
  TF_OpKernelConstruction_GetAttrInt32List(ctx, "strides", strides, 4, status);
  if (TF_GetCode(status) == TF_OK) {
    kernel_ctx->strides.assign(strides, strides + 4);
  }
  
  char padding_buf[32];
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", padding_buf, 32, status);
  if (TF_GetCode(status) == TF_OK) {
    kernel_ctx->padding = padding_buf;
  }
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSExtractPatches_Delete(void* kernel) {
  delete reinterpret_cast<MPSExtractPatchesCtx*>(kernel);
}

extern "C" void MPSExtractPatches_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ExtractImagePatches not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RGBToHSV =====
struct MPSRGBToHSVContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> pipeline;
};

extern "C" void* MPSRGBToHSV_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSRGBToHSVContext;
  
  @autoreleasepool {
    kernel_ctx->device = MTLCreateSystemDefaultDevice();
    kernel_ctx->commandQueue = [kernel_ctx->device newCommandQueue];
    
    NSError* error = nil;
    NSString* source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void rgb_to_hsv(device const float* rgb [[buffer(0)]],
                        device float* hsv [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  uint idx = gid * 3;
  float r = rgb[idx];
  float g = rgb[idx + 1];
  float b = rgb[idx + 2];
  
  float maxc = max(max(r, g), b);
  float minc = min(min(r, g), b);
  float delta = maxc - minc;
  
  // Value
  float v = maxc;
  
  // Saturation
  float s = (maxc > 0.0f) ? (delta / maxc) : 0.0f;
  
  // Hue
  float h = 0.0f;
  if (delta > 0.0f) {
    if (maxc == r) {
      h = (g - b) / delta;
      if (g < b) h += 6.0f;
    } else if (maxc == g) {
      h = 2.0f + (b - r) / delta;
    } else {
      h = 4.0f + (r - g) / delta;
    }
    h /= 6.0f;
  }
  
  hsv[idx] = h;
  hsv[idx + 1] = s;
  hsv[idx + 2] = v;
}
)";
    
    id<MTLLibrary> lib = [kernel_ctx->device newLibraryWithSource:source options:nil error:&error];
    if (lib) {
      id<MTLFunction> func = [lib newFunctionWithName:@"rgb_to_hsv"];
      kernel_ctx->pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:func error:&error];
      [func release];
      [lib release];
    }
  }
  
  return kernel_ctx;
}

extern "C" void MPSRGBToHSV_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSRGBToHSVContext*>(kernel);
    if (k->pipeline) [k->pipeline release];
    if (k->commandQueue) [k->commandQueue release];
    if (k->device) [k->device release];
    delete k;
  }
}

extern "C" void MPSRGBToHSV_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSRGBToHSVContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(input);
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(input, i); nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (!kernel_ctx->pipeline) {
    TF_SetStatus(status, TF_INTERNAL, "RGBToHSV: Pipeline not created");
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  int64_t num_pixels = nelems / 3;
  id<MTLBuffer> rgbBuf = [kernel_ctx->device newBufferWithBytes:TF_TensorData(input) length:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> hsvBuf = [kernel_ctx->device newBufferWithLength:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  
  id<MTLCommandBuffer> cb = [kernel_ctx->commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:kernel_ctx->pipeline];
  [enc setBuffer:rgbBuf offset:0 atIndex:0];
  [enc setBuffer:hsvBuf offset:0 atIndex:1];
  NSUInteger tgSize = kernel_ctx->pipeline.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(num_pixels, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  
  memcpy(TF_TensorData(output), hsvBuf.contents, nelems*sizeof(float));
  [rgbBuf release]; [hsvBuf release];
  }
  
  TF_DeleteStatus(status);
}

// ===== HSVToRGB =====
struct MPSHSVToRGBContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> pipeline;
};

extern "C" void* MPSHSVToRGB_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSHSVToRGBContext;
  
  @autoreleasepool {
    kernel_ctx->device = MTLCreateSystemDefaultDevice();
    kernel_ctx->commandQueue = [kernel_ctx->device newCommandQueue];
    
    NSError* error = nil;
    NSString* source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void hsv_to_rgb(device const float* hsv [[buffer(0)]],
                        device float* rgb [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  uint idx = gid * 3;
  float h = hsv[idx];
  float s = hsv[idx + 1];
  float v = hsv[idx + 2];
  
  float r, g, b;
  
  if (s == 0.0f) {
    r = g = b = v;
  } else {
    h *= 6.0f;
    int i = (int)floor(h);
    float f = h - float(i);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
  }
  
  rgb[idx] = r;
  rgb[idx + 1] = g;
  rgb[idx + 2] = b;
}
)";
    
    id<MTLLibrary> lib = [kernel_ctx->device newLibraryWithSource:source options:nil error:&error];
    if (lib) {
      id<MTLFunction> func = [lib newFunctionWithName:@"hsv_to_rgb"];
      kernel_ctx->pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:func error:&error];
      [func release];
      [lib release];
    }
  }
  
  return kernel_ctx;
}

extern "C" void MPSHSVToRGB_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSHSVToRGBContext*>(kernel);
    if (k->pipeline) [k->pipeline release];
    if (k->commandQueue) [k->commandQueue release];
    if (k->device) [k->device release];
    delete k;
  }
}

extern "C" void MPSHSVToRGB_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSHSVToRGBContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(input);
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(input, i); nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (!kernel_ctx->pipeline) {
    TF_SetStatus(status, TF_INTERNAL, "HSVToRGB: Pipeline not created");
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  int64_t num_pixels = nelems / 3;
  id<MTLBuffer> hsvBuf = [kernel_ctx->device newBufferWithBytes:TF_TensorData(input) length:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> rgbBuf = [kernel_ctx->device newBufferWithLength:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  
  id<MTLCommandBuffer> cb = [kernel_ctx->commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:kernel_ctx->pipeline];
  [enc setBuffer:hsvBuf offset:0 atIndex:0];
  [enc setBuffer:rgbBuf offset:0 atIndex:1];
  NSUInteger tgSize = kernel_ctx->pipeline.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(num_pixels, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  
  memcpy(TF_TensorData(output), rgbBuf.contents, nelems*sizeof(float));
  [hsvBuf release]; [rgbBuf release];
  }
  
  TF_DeleteStatus(status);
}

// ===== AdjustContrast =====
struct MPSAdjustContrastContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> pipeline;
};

extern "C" void* MPSAdjustContrast_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSAdjustContrastContext;
  
  @autoreleasepool {
    kernel_ctx->device = MTLCreateSystemDefaultDevice();
    kernel_ctx->commandQueue = [kernel_ctx->device newCommandQueue];
    
    NSError* error = nil;
    NSString* source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void adjust_contrast(device const float* images [[buffer(0)]],
                              device const float* factor [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& channels [[buffer(3)]],
                              uint3 gid [[thread_position_in_grid]]) {
  uint idx = gid.x * channels + gid.z;
  float mean = 0.0f;
  for (uint c = 0; c < channels; c++) {
    mean += images[gid.x * channels + c];
  }
  mean /= float(channels);
  output[idx] = mean + (images[idx] - mean) * factor[0];
}
)";
    
    id<MTLLibrary> lib = [kernel_ctx->device newLibraryWithSource:source options:nil error:&error];
    if (lib) {
      id<MTLFunction> func = [lib newFunctionWithName:@"adjust_contrast"];
      kernel_ctx->pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:func error:&error];
      [func release];
      [lib release];
    }
  }
  
  return kernel_ctx;
}

extern "C" void MPSAdjustContrast_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSAdjustContrastContext*>(kernel);
    if (k->pipeline) [k->pipeline release];
    if (k->commandQueue) [k->commandQueue release];
    if (k->device) [k->device release];
    delete k;
  }
}

extern "C" void MPSAdjustContrast_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSAdjustContrastContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* images = nullptr;
  TF_GetInput(ctx, 0, &images, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* factor = nullptr;
  TF_GetInput(ctx, 1, &factor, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(images);
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(images, i); nelems *= dims[i]; }
  
  uint channels = (nd > 0) ? dims[nd - 1] : 1;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (!kernel_ctx->pipeline) {
    TF_SetStatus(status, TF_INTERNAL, "AdjustContrast: Pipeline not created");
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  id<MTLBuffer> imgBuf = [kernel_ctx->device newBufferWithBytes:TF_TensorData(images) length:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> facBuf = [kernel_ctx->device newBufferWithBytes:TF_TensorData(factor) length:sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> outBuf = [kernel_ctx->device newBufferWithLength:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> chBuf = [kernel_ctx->device newBufferWithBytes:&channels length:sizeof(uint) options:MTLResourceStorageModeShared];
  
  id<MTLCommandBuffer> cb = [kernel_ctx->commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:kernel_ctx->pipeline];
  [enc setBuffer:imgBuf offset:0 atIndex:0];
  [enc setBuffer:facBuf offset:0 atIndex:1];
  [enc setBuffer:outBuf offset:0 atIndex:2];
  [enc setBuffer:chBuf offset:0 atIndex:3];
  NSUInteger tgSize = kernel_ctx->pipeline.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems/channels, 1, channels) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  
  memcpy(TF_TensorData(output), outBuf.contents, nelems*sizeof(float));
  [imgBuf release]; [facBuf release]; [outBuf release]; [chBuf release];
  }
  
  TF_DeleteStatus(status);
}

// ===== AdjustBrightness =====
struct MPSAdjustBrightnessContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLComputePipelineState> pipeline;
};

extern "C" void* MPSAdjustBrightness_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSAdjustBrightnessContext;
  
  @autoreleasepool {
    kernel_ctx->device = MTLCreateSystemDefaultDevice();
    kernel_ctx->commandQueue = [kernel_ctx->device newCommandQueue];
    
    NSError* error = nil;
    NSString* source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void adjust_brightness(device const float* images [[buffer(0)]],
                                device const float* delta [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  output[gid] = images[gid] + delta[0];
}
)";
    
    id<MTLLibrary> lib = [kernel_ctx->device newLibraryWithSource:source options:nil error:&error];
    if (lib) {
      id<MTLFunction> func = [lib newFunctionWithName:@"adjust_brightness"];
      kernel_ctx->pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:func error:&error];
      [func release];
      [lib release];
    }
  }
  
  return kernel_ctx;
}

extern "C" void MPSAdjustBrightness_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSAdjustBrightnessContext*>(kernel);
    if (k->pipeline) [k->pipeline release];
    if (k->commandQueue) [k->commandQueue release];
    if (k->device) [k->device release];
    delete k;
  }
}

extern "C" void MPSAdjustBrightness_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSAdjustBrightnessContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* images = nullptr;
  TF_GetInput(ctx, 0, &images, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* delta = nullptr;
  TF_GetInput(ctx, 1, &delta, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(images);
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = TF_Dim(images, i); nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (!kernel_ctx->pipeline) {
    TF_SetStatus(status, TF_INTERNAL, "AdjustBrightness: Pipeline not created");
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  id<MTLBuffer> imgBuf = [kernel_ctx->device newBufferWithBytes:TF_TensorData(images) length:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> delBuf = [kernel_ctx->device newBufferWithBytes:TF_TensorData(delta) length:sizeof(float) options:MTLResourceStorageModeShared];
  id<MTLBuffer> outBuf = [kernel_ctx->device newBufferWithLength:nelems*sizeof(float) options:MTLResourceStorageModeShared];
  
  id<MTLCommandBuffer> cb = [kernel_ctx->commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:kernel_ctx->pipeline];
  [enc setBuffer:imgBuf offset:0 atIndex:0];
  [enc setBuffer:delBuf offset:0 atIndex:1];
  [enc setBuffer:outBuf offset:0 atIndex:2];
  NSUInteger tgSize = kernel_ctx->pipeline.maxTotalThreadsPerThreadgroup;
  [enc dispatchThreads:MTLSizeMake(nelems, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  
  memcpy(TF_TensorData(output), outBuf.contents, nelems*sizeof(float));
  [imgBuf release]; [delBuf release]; [outBuf release];
  }
  
  TF_DeleteStatus(status);
}

// ===== AdjustSaturation =====
extern "C" void* MPSAdjustSaturation_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSAdjustSaturation_Delete(void* kernel) {}

extern "C" void MPSAdjustSaturation_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "AdjustSaturation not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== AdjustHue =====
extern "C" void* MPSAdjustHue_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSAdjustHue_Delete(void* kernel) {}

extern "C" void MPSAdjustHue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "AdjustHue not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodeJpeg =====
extern "C" void* MPSDecodeJpeg_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodeJpeg_Delete(void* kernel) {}

extern "C" void MPSDecodeJpeg_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodeJpeg CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodePng =====
extern "C" void* MPSDecodePng_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodePng_Delete(void* kernel) {}

extern "C" void MPSDecodePng_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodePng CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== EncodeJpeg =====
extern "C" void* MPSEncodeJpeg_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSEncodeJpeg_Delete(void* kernel) {}

extern "C" void MPSEncodeJpeg_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "EncodeJpeg CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
