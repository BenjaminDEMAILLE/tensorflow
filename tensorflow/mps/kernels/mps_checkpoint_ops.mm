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

// Checkpointing and model persistence operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== Save =====
extern "C" void* MPSSave_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSave_Delete(void* kernel) {}

extern "C" void MPSSave_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Save is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SaveV2 =====
extern "C" void* MPSSaveV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSaveV2_Delete(void* kernel) {}

extern "C" void MPSSaveV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SaveV2 is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Restore =====
extern "C" void* MPSRestore_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRestore_Delete(void* kernel) {}

extern "C" void MPSRestore_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Restore is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RestoreV2 =====
extern "C" void* MPSRestoreV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRestoreV2_Delete(void* kernel) {}

extern "C" void MPSRestoreV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RestoreV2 is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SaveSlices =====
extern "C" void* MPSSaveSlices_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSaveSlices_Delete(void* kernel) {}

extern "C" void MPSSaveSlices_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SaveSlices is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RestoreSlice =====
extern "C" void* MPSRestoreSlice_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRestoreSlice_Delete(void* kernel) {}

extern "C" void MPSRestoreSlice_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RestoreSlice is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ShardedFilename =====
extern "C" void* MPSShardedFilename_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSShardedFilename_Delete(void* kernel) {}

extern "C" void MPSShardedFilename_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ShardedFilename is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ShardedFilespec =====
extern "C" void* MPSShardedFilespec_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSShardedFilespec_Delete(void* kernel) {}

extern "C" void MPSShardedFilespec_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ShardedFilespec is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== MergeV2Checkpoints =====
extern "C" void* MPSMergeV2Checkpoints_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMergeV2Checkpoints_Delete(void* kernel) {}

extern "C" void MPSMergeV2Checkpoints_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MergeV2Checkpoints is CPU-only operation (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
