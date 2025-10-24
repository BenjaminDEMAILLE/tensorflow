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

// Data manipulation operations (Stack, Unstack, Concat, Pack, Unpack) for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// ===== Stack =====
struct MPSStackCtx {
  int32_t axis;
  int32_t num_inputs;
};

extern "C" void* MPSStack_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSStackCtx();
  TF_Status* status = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "axis", &kernel_ctx->axis, status);
  TF_OpKernelConstruction_GetAttrInt32(ctx, "N", &kernel_ctx->num_inputs, status);
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSStack_Delete(void* kernel) {
  delete reinterpret_cast<MPSStackCtx*>(kernel);
}

extern "C" void MPSStack_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Stack not yet fully implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Unstack =====
extern "C" void* MPSUnstack_Create(TF_OpKernelConstruction* ctx) {
  return MPSStack_Create(ctx);
}

extern "C" void MPSUnstack_Delete(void* kernel) {
  MPSStack_Delete(kernel);
}

extern "C" void MPSUnstack_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Unstack not yet fully implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ConcatV2 =====
extern "C" void* MPSConcatV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSConcatV2_Delete(void* kernel) {}

extern "C" void MPSConcatV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ConcatV2 not yet fully implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Pack =====
extern "C" void* MPSPack_Create(TF_OpKernelConstruction* ctx) {
  return MPSStack_Create(ctx);
}

extern "C" void MPSPack_Delete(void* kernel) {
  MPSStack_Delete(kernel);
}

extern "C" void MPSPack_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Pack not yet fully implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Unpack =====
extern "C" void* MPSUnpack_Create(TF_OpKernelConstruction* ctx) {
  return MPSStack_Create(ctx);
}

extern "C" void MPSUnpack_Delete(void* kernel) {
  MPSStack_Delete(kernel);
}

extern "C" void MPSUnpack_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Unpack not yet fully implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Squeeze =====
extern "C" void* MPSSqueeze_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSqueeze_Delete(void* kernel) {}

extern "C" void MPSSqueeze_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  // Squeeze is a shape manipulation - can be done on CPU
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Squeeze shape manipulation pending");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ExpandDims =====
extern "C" void* MPSExpandDims_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSExpandDims_Delete(void* kernel) {}

extern "C" void MPSExpandDims_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ExpandDims shape manipulation pending");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
