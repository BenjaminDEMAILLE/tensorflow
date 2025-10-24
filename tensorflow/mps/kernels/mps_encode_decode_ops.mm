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

// Encode/Decode operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== EncodeBase64 =====
extern "C" void* MPSEncodeBase64_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSEncodeBase64_Delete(void* kernel) {}

extern "C" void MPSEncodeBase64_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "EncodeBase64 is CPU-only (string encoding)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodeBase64 =====
extern "C" void* MPSDecodeBase64_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodeBase64_Delete(void* kernel) {}

extern "C" void MPSDecodeBase64_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodeBase64 is CPU-only (string decoding)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== EncodeProto =====
extern "C" void* MPSEncodeProto_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSEncodeProto_Delete(void* kernel) {}

extern "C" void MPSEncodeProto_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "EncodeProto is CPU-only (protobuf encoding)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodeProto =====
extern "C" void* MPSDecodeProto_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodeProto_Delete(void* kernel) {}

extern "C" void MPSDecodeProto_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodeProto is CPU-only (protobuf decoding)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodeRaw =====
extern "C" void* MPSDecodeRaw_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodeRaw_Delete(void* kernel) {}

extern "C" void MPSDecodeRaw_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodeRaw is CPU-only (raw bytes decoding)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodeCompressed =====
extern "C" void* MPSDecodeCompressed_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodeCompressed_Delete(void* kernel) {}

extern "C" void MPSDecodeCompressed_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodeCompressed is CPU-only (decompression)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodeCSV =====
extern "C" void* MPSDecodeCSV_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodeCSV_Delete(void* kernel) {}

extern "C" void MPSDecodeCSV_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodeCSV is CPU-only (CSV parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodeJSON =====
extern "C" void* MPSDecodeJSON_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodeJSON_Delete(void* kernel) {}

extern "C" void MPSDecodeJSON_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodeJSON is CPU-only (JSON parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
