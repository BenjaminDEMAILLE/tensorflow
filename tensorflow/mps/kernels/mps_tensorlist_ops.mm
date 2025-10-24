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

// TensorList operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== TensorListReserve =====
extern "C" void* MPSTensorListReserve_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListReserve_Delete(void* kernel) {}

extern "C" void MPSTensorListReserve_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListReserve requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListPushBack =====
extern "C" void* MPSTensorListPushBack_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListPushBack_Delete(void* kernel) {}

extern "C" void MPSTensorListPushBack_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListPushBack requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListPopBack =====
extern "C" void* MPSTensorListPopBack_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListPopBack_Delete(void* kernel) {}

extern "C" void MPSTensorListPopBack_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListPopBack requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListGetItem =====
extern "C" void* MPSTensorListGetItem_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListGetItem_Delete(void* kernel) {}

extern "C" void MPSTensorListGetItem_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListGetItem requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListSetItem =====
extern "C" void* MPSTensorListSetItem_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListSetItem_Delete(void* kernel) {}

extern "C" void MPSTensorListSetItem_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListSetItem requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListStack =====
extern "C" void* MPSTensorListStack_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListStack_Delete(void* kernel) {}

extern "C" void MPSTensorListStack_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListStack requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListFromTensor =====
extern "C" void* MPSTensorListFromTensor_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListFromTensor_Delete(void* kernel) {}

extern "C" void MPSTensorListFromTensor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListFromTensor requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListGather =====
extern "C" void* MPSTensorListGather_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListGather_Delete(void* kernel) {}

extern "C" void MPSTensorListGather_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListGather requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListScatter =====
extern "C" void* MPSTensorListScatter_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListScatter_Delete(void* kernel) {}

extern "C" void MPSTensorListScatter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListScatter requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorListLength =====
extern "C" void* MPSTensorListLength_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorListLength_Delete(void* kernel) {}

extern "C" void MPSTensorListLength_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorListLength requires variant tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
