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

// Scattering operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== ScatterNd =====
extern "C" void* MPSScatterNd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSScatterNd_Delete(void* kernel) {}

extern "C" void MPSScatterNd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ScatterNd requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ScatterNdAdd =====
extern "C" void* MPSScatterNdAdd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSScatterNdAdd_Delete(void* kernel) {}

extern "C" void MPSScatterNdAdd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ScatterNdAdd requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ScatterNdUpdate =====
extern "C" void* MPSScatterNdUpdate_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSScatterNdUpdate_Delete(void* kernel) {}

extern "C" void MPSScatterNdUpdate_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ScatterNdUpdate requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ScatterNdSub =====
extern "C" void* MPSScatterNdSub_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSScatterNdSub_Delete(void* kernel) {}

extern "C" void MPSScatterNdSub_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ScatterNdSub requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ScatterNdMax =====
extern "C" void* MPSScatterNdMax_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSScatterNdMax_Delete(void* kernel) {}

extern "C" void MPSScatterNdMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ScatterNdMax requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ScatterNdMin =====
extern "C" void* MPSScatterNdMin_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSScatterNdMin_Delete(void* kernel) {}

extern "C" void MPSScatterNdMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ScatterNdMin requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorScatterUpdate =====
extern "C" void* MPSTensorScatterUpdate_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorScatterUpdate_Delete(void* kernel) {}

extern "C" void MPSTensorScatterUpdate_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorScatterUpdate requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorScatterAdd =====
extern "C" void* MPSTensorScatterAdd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorScatterAdd_Delete(void* kernel) {}

extern "C" void MPSTensorScatterAdd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorScatterAdd requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorScatterSub =====
extern "C" void* MPSTensorScatterSub_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorScatterSub_Delete(void* kernel) {}

extern "C" void MPSTensorScatterSub_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorScatterSub requires scatter logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
