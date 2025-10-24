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

// Stateful operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== StatefulPartitionedCall =====
extern "C" void* MPSStatefulPartitionedCall_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStatefulPartitionedCall_Delete(void* kernel) {}

extern "C" void MPSStatefulPartitionedCall_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StatefulPartitionedCall requires function execution");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== PartitionedCall =====
extern "C" void* MPSPartitionedCall_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSPartitionedCall_Delete(void* kernel) {}

extern "C" void MPSPartitionedCall_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "PartitionedCall requires function execution");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StatefulRandomBinomial =====
extern "C" void* MPSStatefulRandomBinomial_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStatefulRandomBinomial_Delete(void* kernel) {}

extern "C" void MPSStatefulRandomBinomial_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StatefulRandomBinomial is CPU-only (RNG state)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StatefulStandardNormal =====
extern "C" void* MPSStatefulStandardNormal_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStatefulStandardNormal_Delete(void* kernel) {}

extern "C" void MPSStatefulStandardNormal_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StatefulStandardNormal is CPU-only (RNG state)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StatefulUniform =====
extern "C" void* MPSStatefulUniform_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStatefulUniform_Delete(void* kernel) {}

extern "C" void MPSStatefulUniform_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StatefulUniform is CPU-only (RNG state)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StatefulUniformInt =====
extern "C" void* MPSStatefulUniformInt_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStatefulUniformInt_Delete(void* kernel) {}

extern "C" void MPSStatefulUniformInt_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StatefulUniformInt is CPU-only (RNG state)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StatefulTruncatedNormal =====
extern "C" void* MPSStatefulTruncatedNormal_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStatefulTruncatedNormal_Delete(void* kernel) {}

extern "C" void MPSStatefulTruncatedNormal_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StatefulTruncatedNormal is CPU-only (RNG state)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RngSkip =====
extern "C" void* MPSRngSkip_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRngSkip_Delete(void* kernel) {}

extern "C" void MPSRngSkip_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RngSkip is CPU-only (RNG state)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
