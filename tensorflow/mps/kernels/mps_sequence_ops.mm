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

// Sequence operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

// ===== CumSum =====
extern "C" void* MPSCumSum_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCumSum_Delete(void* kernel) {}

extern "C" void MPSCumSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CumSum requires cumulative computation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CumProd =====
extern "C" void* MPSCumProd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCumProd_Delete(void* kernel) {}

extern "C" void MPSCumProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CumProd requires cumulative computation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Cumsum (lowercase variant) =====
extern "C" void* MPSCumsum_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCumsum_Delete(void* kernel) {}

extern "C" void MPSCumsum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Cumsum requires cumulative computation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Cumprod (lowercase variant) =====
extern "C" void* MPSCumprod_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCumprod_Delete(void* kernel) {}

extern "C" void MPSCumprod_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Cumprod requires cumulative computation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Reverse =====
extern "C" void* MPSReverse_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSReverse_Delete(void* kernel) {}

extern "C" void MPSReverse_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Reverse requires axis reversal");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ReverseV2 =====
extern "C" void* MPSReverseV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSReverseV2_Delete(void* kernel) {}

extern "C" void MPSReverseV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ReverseV2 requires axis reversal");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ReverseSequence =====
extern "C" void* MPSReverseSequence_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSReverseSequence_Delete(void* kernel) {}

extern "C" void MPSReverseSequence_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ReverseSequence requires sequence reversal");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
