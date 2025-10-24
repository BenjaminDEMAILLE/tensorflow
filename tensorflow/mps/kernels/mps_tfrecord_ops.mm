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

// TFRecord operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== TFRecordReader =====
extern "C" void* MPSTFRecordReader_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTFRecordReader_Delete(void* kernel) {}

extern "C" void MPSTFRecordReader_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TFRecordReader is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TFRecordReaderV2 =====
extern "C" void* MPSTFRecordReaderV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTFRecordReaderV2_Delete(void* kernel) {}

extern "C" void MPSTFRecordReaderV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TFRecordReaderV2 is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TFRecordDataset =====
extern "C" void* MPSTFRecordDataset_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTFRecordDataset_Delete(void* kernel) {}

extern "C" void MPSTFRecordDataset_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TFRecordDataset is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== FixedLengthRecordReader =====
extern "C" void* MPSFixedLengthRecordReader_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSFixedLengthRecordReader_Delete(void* kernel) {}

extern "C" void MPSFixedLengthRecordReader_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FixedLengthRecordReader is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== TextLineReader =====
extern "C" void* MPSTextLineReader_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSTextLineReader_Delete(void* kernel) {}

extern "C" void MPSTextLineReader_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "TextLineReader is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== WholeFileReader =====
extern "C" void* MPSWholeFileReader_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSWholeFileReader_Delete(void* kernel) {}

extern "C" void MPSWholeFileReader_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "WholeFileReader is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== IdentityReader =====
extern "C" void* MPSIdentityReader_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSIdentityReader_Delete(void* kernel) {}

extern "C" void MPSIdentityReader_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "IdentityReader is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ReaderRead =====
extern "C" void* MPSReaderRead_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSReaderRead_Delete(void* kernel) {}

extern "C" void MPSReaderRead_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ReaderRead is CPU-only (file I/O)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
