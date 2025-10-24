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

void* MPSBiasAdd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSBiasAdd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // BiasAdd: value + bias (broadcast bias across batch/spatial dims)
  // Use MPSGraph broadcast + add
  TF_SetStatus(s, TF_UNIMPLEMENTED, "BiasAdd - TODO (MPSGraph broadcast + add)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSBiasAdd_Delete(void* kernel) {}

void* MPSDepthToSpace_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSDepthToSpace_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // DepthToSpace: rearrange (permute) tensor from depth into blocks of spatial data
  TF_SetStatus(s, TF_UNIMPLEMENTED, "DepthToSpace - TODO (Metal permute kernel)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSDepthToSpace_Delete(void* kernel) {}

void RegisterNNExtendedOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("BiasAdd", platform_name,
                                                &MPSBiasAdd_Create,
                                                &MPSBiasAdd_Compute,
                                                &MPSBiasAdd_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSBiasAdd", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("DepthToSpace", platform_name,
                                                &MPSDepthToSpace_Create,
                                                &MPSDepthToSpace_Compute,
                                                &MPSDepthToSpace_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSDepthToSpace", kb, status);
  }
  // TODO: 98+ more extended NN ops
  // Conv2DBackpropInput, Conv2DBackpropFilter, FusedBatchNormGrad
  // QuantizedConv2D, QuantizedMatMul, SpaceToDepth, SpaceToBatchND
  // BatchToSpaceND, Dilation2D, etc.
}

}  // namespace mps
}  // namespace tensorflow
