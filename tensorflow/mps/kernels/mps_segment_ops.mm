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

// Segment reduction operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== SegmentSum =====
extern "C" void* MPSSegmentSum_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSegmentSum_Delete(void* kernel) {}

extern "C" void MPSSegmentSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SegmentSum requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SegmentMean =====
extern "C" void* MPSSegmentMean_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSegmentMean_Delete(void* kernel) {}

extern "C" void MPSSegmentMean_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SegmentMean requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SegmentMax =====
extern "C" void* MPSSegmentMax_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSegmentMax_Delete(void* kernel) {}

extern "C" void MPSSegmentMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SegmentMax requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SegmentMin =====
extern "C" void* MPSSegmentMin_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSegmentMin_Delete(void* kernel) {}

extern "C" void MPSSegmentMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SegmentMin requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SegmentProd =====
extern "C" void* MPSSegmentProd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSegmentProd_Delete(void* kernel) {}

extern "C" void MPSSegmentProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SegmentProd requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnsortedSegmentSum =====
extern "C" void* MPSUnsortedSegmentSum_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnsortedSegmentSum_Delete(void* kernel) {}

extern "C" void MPSUnsortedSegmentSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnsortedSegmentSum requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnsortedSegmentMax =====
extern "C" void* MPSUnsortedSegmentMax_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnsortedSegmentMax_Delete(void* kernel) {}

extern "C" void MPSUnsortedSegmentMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnsortedSegmentMax requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnsortedSegmentMin =====
extern "C" void* MPSUnsortedSegmentMin_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnsortedSegmentMin_Delete(void* kernel) {}

extern "C" void MPSUnsortedSegmentMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnsortedSegmentMin requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnsortedSegmentProd =====
extern "C" void* MPSUnsortedSegmentProd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnsortedSegmentProd_Delete(void* kernel) {}

extern "C" void MPSUnsortedSegmentProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnsortedSegmentProd requires segment reduction");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
