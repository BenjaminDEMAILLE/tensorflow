/* Copyright 2025 The TensorFlow Authors.

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

#include "tensorflow/mps/ops/mps_ops_registry.h"
#include "tensorflow/c/kernels.h"

// ============================================================================
// Logical Operations (3 ops)
// ============================================================================
// LogicalAnd, LogicalOr, LogicalNot

namespace tensorflow {
namespace mps {

// TODO: Extract from mps_pluggable_device_plugin.mm

void* MPSLogicalAnd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLogicalAnd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "LogicalAnd not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLogicalAnd_Delete(void* kernel) {}

void* MPSLogicalOr_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLogicalOr_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "LogicalOr not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLogicalOr_Delete(void* kernel) {}

void* MPSLogicalNot_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLogicalNot_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "LogicalNot not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLogicalNot_Delete(void* kernel) {}

void* MPSLogicalXor_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLogicalXor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "LogicalXor not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLogicalXor_Delete(void* kernel) {}

void* MPSEqual_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSEqual_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Equal not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSEqual_Delete(void* kernel) {}

void* MPSNotEqual_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSNotEqual_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "NotEqual not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSNotEqual_Delete(void* kernel) {}

void* MPSGreater_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSGreater_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Greater not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSGreater_Delete(void* kernel) {}

void* MPSGreaterEqual_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSGreaterEqual_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "GreaterEqual not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSGreaterEqual_Delete(void* kernel) {}

void* MPSLess_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLess_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Less not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLess_Delete(void* kernel) {}

void* MPSLessEqual_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSLessEqual_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "LessEqual not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSLessEqual_Delete(void* kernel) {}

void* MPSSelect_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSSelect_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Select not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSSelect_Delete(void* kernel) {}

void* MPSSelectV2_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSSelectV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SelectV2 not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSSelectV2_Delete(void* kernel) {}

void* MPSWhere_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSWhere_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Where not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSWhere_Delete(void* kernel) {}

void* MPSIsFinite_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSIsFinite_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "IsFinite not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSIsFinite_Delete(void* kernel) {}

void* MPSIsInf_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSIsInf_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "IsInf not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSIsInf_Delete(void* kernel) {}

void* MPSIsNan_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSIsNan_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "IsNan not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSIsNan_Delete(void* kernel) {}

void RegisterLogicalOps(const char* platform_name, TF_Status* status) {
  // TODO: Registration
}

}  // namespace mps
}  // namespace tensorflow

