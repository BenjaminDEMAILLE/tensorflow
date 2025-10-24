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

// Top-K and sorting operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== TopK =====
extern "C" void* MPSTopK_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTopK_Delete(void* kernel) {}

extern "C" void MPSTopK_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TopK requires partial sorting");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TopKV2 =====
extern "C" void* MPSTopKV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTopKV2_Delete(void* kernel) {}

extern "C" void MPSTopKV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TopKV2 requires partial sorting");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== NthElement =====
extern "C" void* MPSNthElement_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSNthElement_Delete(void* kernel) {}

extern "C" void MPSNthElement_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "NthElement requires selection algorithm");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UniqueV2 =====
extern "C" void* MPSUniqueV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUniqueV2_Delete(void* kernel) {}

extern "C" void MPSUniqueV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UniqueV2 requires hash set");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UniqueWithCounts =====
extern "C" void* MPSUniqueWithCounts_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUniqueWithCounts_Delete(void* kernel) {}

extern "C" void MPSUniqueWithCounts_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UniqueWithCounts requires hash map");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ListDiff =====
extern "C" void* MPSListDiff_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSListDiff_Delete(void* kernel) {}

extern "C" void MPSListDiff_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ListDiff requires set operations");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
