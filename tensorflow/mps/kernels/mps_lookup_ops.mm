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

// Lookup table operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== HashTable =====
extern "C" void* MPSHashTable_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSHashTable_Delete(void* kernel) {}

extern "C" void MPSHashTable_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "HashTable requires resource management");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== HashTableV2 =====
extern "C" void* MPSHashTableV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSHashTableV2_Delete(void* kernel) {}

extern "C" void MPSHashTableV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "HashTableV2 requires resource management");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== MutableHashTable =====
extern "C" void* MPSMutableHashTable_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMutableHashTable_Delete(void* kernel) {}

extern "C" void MPSMutableHashTable_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MutableHashTable requires resource management");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== LookupTableFind =====
extern "C" void* MPSLookupTableFind_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSLookupTableFind_Delete(void* kernel) {}

extern "C" void MPSLookupTableFind_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LookupTableFind requires resource management");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== LookupTableInsert =====
extern "C" void* MPSLookupTableInsert_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSLookupTableInsert_Delete(void* kernel) {}

extern "C" void MPSLookupTableInsert_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LookupTableInsert requires resource management");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== LookupTableSize =====
extern "C" void* MPSLookupTableSize_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSLookupTableSize_Delete(void* kernel) {}

extern "C" void MPSLookupTableSize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LookupTableSize requires resource management");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== InitializeTable =====
extern "C" void* MPSInitializeTable_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSInitializeTable_Delete(void* kernel) {}

extern "C" void MPSInitializeTable_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "InitializeTable requires resource management");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== InitializeTableFromTextFile =====
extern "C" void* MPSInitializeTableFromTextFile_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSInitializeTableFromTextFile_Delete(void* kernel) {}

extern "C" void MPSInitializeTableFromTextFile_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "InitializeTableFromTextFile requires file I/O");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
