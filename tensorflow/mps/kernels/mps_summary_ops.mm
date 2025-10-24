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

// Summary and logging operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== ScalarSummary =====
extern "C" void* MPSScalarSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSScalarSummary_Delete(void* kernel) {}

extern "C" void MPSScalarSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ScalarSummary is CPU-only (logging)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== HistogramSummary =====
extern "C" void* MPSHistogramSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSHistogramSummary_Delete(void* kernel) {}

extern "C" void MPSHistogramSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "HistogramSummary is CPU-only (logging)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ImageSummary =====
extern "C" void* MPSImageSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSImageSummary_Delete(void* kernel) {}

extern "C" void MPSImageSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ImageSummary is CPU-only (logging)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== AudioSummary =====
extern "C" void* MPSAudioSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSAudioSummary_Delete(void* kernel) {}

extern "C" void MPSAudioSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "AudioSummary is CPU-only (logging)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== MergeSummary =====
extern "C" void* MPSMergeSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMergeSummary_Delete(void* kernel) {}

extern "C" void MPSMergeSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MergeSummary is CPU-only (logging)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TensorSummary =====
extern "C" void* MPSTensorSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTensorSummary_Delete(void* kernel) {}

extern "C" void MPSTensorSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TensorSummary is CPU-only (logging)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SummaryWriter =====
extern "C" void* MPSSummaryWriter_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSummaryWriter_Delete(void* kernel) {}

extern "C" void MPSSummaryWriter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SummaryWriter is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CreateSummaryFileWriter =====
extern "C" void* MPSCreateSummaryFileWriter_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCreateSummaryFileWriter_Delete(void* kernel) {}

extern "C" void MPSCreateSummaryFileWriter_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CreateSummaryFileWriter is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
