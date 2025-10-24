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

// Random number generation operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <random>

// ===== RandomUniform =====
extern "C" void* MPSRandomUniform_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRandomUniform_Delete(void* kernel) {}

extern "C" void MPSRandomUniform_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RandomUniform not yet implemented on MPS, use CPU fallback");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RandomNormal =====
extern "C" void* MPSRandomNormal_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRandomNormal_Delete(void* kernel) {}

extern "C" void MPSRandomNormal_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RandomNormal not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RandomShuffle =====
extern "C" void* MPSRandomShuffle_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRandomShuffle_Delete(void* kernel) {}

extern "C" void MPSRandomShuffle_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RandomShuffle not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TruncatedNormal =====
extern "C" void* MPSTruncatedNormal_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTruncatedNormal_Delete(void* kernel) {}

extern "C" void MPSTruncatedNormal_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TruncatedNormal not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Multinomial =====
extern "C" void* MPSMultinomial_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMultinomial_Delete(void* kernel) {}

extern "C" void MPSMultinomial_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Multinomial not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
