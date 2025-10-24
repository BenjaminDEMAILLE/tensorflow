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

// Ragged tensor operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== RaggedTensorToTensor =====
extern "C" void* MPSRaggedTensorToTensor_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRaggedTensorToTensor_Delete(void* kernel) {}

extern "C" void MPSRaggedTensorToTensor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RaggedTensorToTensor requires ragged tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RaggedTensorFromVariant =====
extern "C" void* MPSRaggedTensorFromVariant_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRaggedTensorFromVariant_Delete(void* kernel) {}

extern "C" void MPSRaggedTensorFromVariant_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RaggedTensorFromVariant requires ragged tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RaggedGather =====
extern "C" void* MPSRaggedGather_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRaggedGather_Delete(void* kernel) {}

extern "C" void MPSRaggedGather_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RaggedGather requires ragged tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RaggedRange =====
extern "C" void* MPSRaggedRange_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRaggedRange_Delete(void* kernel) {}

extern "C" void MPSRaggedRange_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RaggedRange requires ragged tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RaggedTensorToSparse =====
extern "C" void* MPSRaggedTensorToSparse_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRaggedTensorToSparse_Delete(void* kernel) {}

extern "C" void MPSRaggedTensorToSparse_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RaggedTensorToSparse requires ragged tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RaggedCross =====
extern "C" void* MPSRaggedCross_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRaggedCross_Delete(void* kernel) {}

extern "C" void MPSRaggedCross_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RaggedCross requires ragged tensor support");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
