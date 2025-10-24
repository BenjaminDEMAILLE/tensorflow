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

// Set operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== SetSize =====
extern "C" void* MPSSetSize_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSetSize_Delete(void* kernel) {}

extern "C" void MPSSetSize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SetSize requires set operation support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SetDifference =====
extern "C" void* MPSSetDifference_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSetDifference_Delete(void* kernel) {}

extern "C" void MPSSetDifference_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SetDifference requires set operation support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SetIntersection =====
extern "C" void* MPSSetIntersection_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSetIntersection_Delete(void* kernel) {}

extern "C" void MPSSetIntersection_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SetIntersection requires set operation support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SetUnion =====
extern "C" void* MPSSetUnion_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSetUnion_Delete(void* kernel) {}

extern "C" void MPSSetUnion_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SetUnion requires set operation support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DenseToDenseSetOperation =====
extern "C" void* MPSDenseToDenseSetOperation_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDenseToDenseSetOperation_Delete(void* kernel) {}

extern "C" void MPSDenseToDenseSetOperation_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DenseToDenseSetOperation requires set operation support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SparseToSparseSetOperation =====
extern "C" void* MPSSparseToSparseSetOperation_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSparseToSparseSetOperation_Delete(void* kernel) {}

extern "C" void MPSSparseToSparseSetOperation_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SparseToSparseSetOperation requires set operation support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DenseToSparseSetOperation =====
extern "C" void* MPSDenseToSparseSetOperation_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDenseToSparseSetOperation_Delete(void* kernel) {}

extern "C" void MPSDenseToSparseSetOperation_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DenseToSparseSetOperation requires set operation support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
