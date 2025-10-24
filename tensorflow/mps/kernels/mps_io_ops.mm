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

// IO operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== ReadFile =====
extern "C" void* MPSReadFile_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSReadFile_Delete(void* kernel) {}

extern "C" void MPSReadFile_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ReadFile is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== WriteFile =====
extern "C" void* MPSWriteFile_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSWriteFile_Delete(void* kernel) {}

extern "C" void MPSWriteFile_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "WriteFile is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== MatchingFiles =====
extern "C" void* MPSMatchingFiles_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMatchingFiles_Delete(void* kernel) {}

extern "C" void MPSMatchingFiles_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MatchingFiles is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DeleteFile =====
extern "C" void* MPSDeleteFile_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDeleteFile_Delete(void* kernel) {}

extern "C" void MPSDeleteFile_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DeleteFile is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== FileExists =====
extern "C" void* MPSFileExists_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSFileExists_Delete(void* kernel) {}

extern "C" void MPSFileExists_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FileExists is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CreateDirectory =====
extern "C" void* MPSCreateDirectory_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCreateDirectory_Delete(void* kernel) {}

extern "C" void MPSCreateDirectory_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CreateDirectory is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DeleteDirectory =====
extern "C" void* MPSDeleteDirectory_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDeleteDirectory_Delete(void* kernel) {}

extern "C" void MPSDeleteDirectory_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DeleteDirectory is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RenameFile =====
extern "C" void* MPSRenameFile_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRenameFile_Delete(void* kernel) {}

extern "C" void MPSRenameFile_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RenameFile is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CopyFile =====
extern "C" void* MPSCopyFile_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCopyFile_Delete(void* kernel) {}

extern "C" void MPSCopyFile_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CopyFile is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== FileSize =====
extern "C" void* MPSFileSize_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSFileSize_Delete(void* kernel) {}

extern "C" void MPSFileSize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FileSize is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
