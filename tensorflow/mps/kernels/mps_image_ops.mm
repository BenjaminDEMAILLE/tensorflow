/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Image Processing Operations
// ResizeBilinear, ResizeNearestNeighbor, CropAndResize, ExtractImagePatches, RGBToHSV, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// ResizeBilinear
void* MPSResizeBilinear_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSResizeBilinear_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ResizeBilinear - TODO");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSResizeBilinear_Delete(void* kernel) {}

// ResizeNearestNeighbor
void* MPSResizeNearestNeighbor_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSResizeNearestNeighbor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ResizeNearestNeighbor - TODO");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSResizeNearestNeighbor_Delete(void* kernel) {}

// CropAndResize, ExtractImagePatches, RGBToHSV... (80+ ops to add)

void RegisterImageOps(const char* platform_name, TF_Status* status) {
  TF_KernelBuilder* kb1 = TF_NewKernelBuilder("ResizeBilinear", platform_name,
                                               &MPSResizeBilinear_Create,
                                               &MPSResizeBilinear_Compute,
                                               &MPSResizeBilinear_Delete);
  TF_KernelBuilder_TypeConstraint(kb1, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSResizeBilinearFloat", kb1, status);
  
  TF_KernelBuilder* kb2 = TF_NewKernelBuilder("ResizeNearestNeighbor", platform_name,
                                               &MPSResizeNearestNeighbor_Create,
                                               &MPSResizeNearestNeighbor_Compute,
                                               &MPSResizeNearestNeighbor_Delete);
  TF_KernelBuilder_TypeConstraint(kb2, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSResizeNearestNeighborFloat", kb2, status);
  
  // TODO: 78+ more image ops
}

}  // namespace mps
}  // namespace tensorflow
