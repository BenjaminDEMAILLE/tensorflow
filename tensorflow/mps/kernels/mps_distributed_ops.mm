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

// Distributed training operations for MPS backend
// Note: Most distributed ops require multi-device coordination handled by runtime

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== CollectiveReduce =====
extern "C" void* MPSCollectiveReduce_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCollectiveReduce_Delete(void* kernel) {}

extern "C" void MPSCollectiveReduce_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CollectiveReduce requires distributed runtime coordination");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CollectiveGather =====
extern "C" void* MPSCollectiveGather_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCollectiveGather_Delete(void* kernel) {}

extern "C" void MPSCollectiveGather_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CollectiveGather requires distributed runtime");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CollectiveBcastSend =====
extern "C" void* MPSCollectiveBcastSend_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCollectiveBcastSend_Delete(void* kernel) {}

extern "C" void MPSCollectiveBcastSend_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CollectiveBcastSend requires distributed runtime");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CollectiveBcastRecv =====
extern "C" void* MPSCollectiveBcastRecv_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCollectiveBcastRecv_Delete(void* kernel) {}

extern "C" void MPSCollectiveBcastRecv_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CollectiveBcastRecv requires distributed runtime");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== NcclAllReduce =====
extern "C" void* MPSNcclAllReduce_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSNcclAllReduce_Delete(void* kernel) {}

extern "C" void MPSNcclAllReduce_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "NCCL not available on MPS - use CollectiveReduce");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== NcclBroadcast =====
extern "C" void* MPSNcclBroadcast_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSNcclBroadcast_Delete(void* kernel) {}

extern "C" void MPSNcclBroadcast_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "NCCL not available on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Send =====
extern "C" void* MPSSend_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSend_Delete(void* kernel) {}

extern "C" void MPSSend_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Send requires runtime communication");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Recv =====
extern "C" void* MPSRecv_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRecv_Delete(void* kernel) {}

extern "C" void MPSRecv_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Recv requires runtime communication");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
