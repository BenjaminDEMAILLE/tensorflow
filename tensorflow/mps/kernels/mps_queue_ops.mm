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

// Queue operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== FIFOQueue =====
extern "C" void* MPSFIFOQueue_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSFIFOQueue_Delete(void* kernel) {}

extern "C" void MPSFIFOQueue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FIFOQueue requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== FIFOQueueV2 =====
extern "C" void* MPSFIFOQueueV2_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSFIFOQueueV2_Delete(void* kernel) {}

extern "C" void MPSFIFOQueueV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FIFOQueueV2 requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== PaddingFIFOQueue =====
extern "C" void* MPSPaddingFIFOQueue_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSPaddingFIFOQueue_Delete(void* kernel) {}

extern "C" void MPSPaddingFIFOQueue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "PaddingFIFOQueue requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== PriorityQueue =====
extern "C" void* MPSPriorityQueue_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSPriorityQueue_Delete(void* kernel) {}

extern "C" void MPSPriorityQueue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "PriorityQueue requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== QueueEnqueue =====
extern "C" void* MPSQueueEnqueue_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSQueueEnqueue_Delete(void* kernel) {}

extern "C" void MPSQueueEnqueue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "QueueEnqueue requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== QueueDequeue =====
extern "C" void* MPSQueueDequeue_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSQueueDequeue_Delete(void* kernel) {}

extern "C" void MPSQueueDequeue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "QueueDequeue requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== QueueSize =====
extern "C" void* MPSQueueSize_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSQueueSize_Delete(void* kernel) {}

extern "C" void MPSQueueSize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "QueueSize requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== QueueClose =====
extern "C" void* MPSQueueClose_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSQueueClose_Delete(void* kernel) {}

extern "C" void MPSQueueClose_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "QueueClose requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== QueueIsClosed =====
extern "C" void* MPSQueueIsClosed_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSQueueIsClosed_Delete(void* kernel) {}

extern "C" void MPSQueueIsClosed_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "QueueIsClosed requires queue management infrastructure");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
