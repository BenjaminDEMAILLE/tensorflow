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

// Parsing operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== ParseExample =====
extern "C" void* MPSParseExample_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSParseExample_Delete(void* kernel) {}

extern "C" void MPSParseExample_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ParseExample is CPU-only (parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ParseExampleV2 =====
extern "C" void* MPSParseExampleV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSParseExampleV2_Delete(void* kernel) {}

extern "C" void MPSParseExampleV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ParseExampleV2 is CPU-only (parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ParseSingleExample =====
extern "C" void* MPSParseSingleExample_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSParseSingleExample_Delete(void* kernel) {}

extern "C" void MPSParseSingleExample_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ParseSingleExample is CPU-only (parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ParseSequenceExample =====
extern "C" void* MPSParseSequenceExample_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSParseSequenceExample_Delete(void* kernel) {}

extern "C" void MPSParseSequenceExample_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ParseSequenceExample is CPU-only (parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ParseSingleSequenceExample =====
extern "C" void* MPSParseSingleSequenceExample_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSParseSingleSequenceExample_Delete(void* kernel) {}

extern "C" void MPSParseSingleSequenceExample_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ParseSingleSequenceExample is CPU-only (parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ParseTensor =====
extern "C" void* MPSParseTensor_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSParseTensor_Delete(void* kernel) {}

extern "C" void MPSParseTensor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ParseTensor is CPU-only (parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SerializeTensor =====
extern "C" void* MPSSerializeTensor_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSerializeTensor_Delete(void* kernel) {}

extern "C" void MPSSerializeTensor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SerializeTensor is CPU-only (serialization)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== DecodePaddedRaw =====
extern "C" void* MPSDecodePaddedRaw_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDecodePaddedRaw_Delete(void* kernel) {}

extern "C" void MPSDecodePaddedRaw_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DecodePaddedRaw is CPU-only (parsing)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
