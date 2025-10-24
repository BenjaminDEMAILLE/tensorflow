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
// Reduction Operations (7 ops)
// ============================================================================
// Sum, Mean, Max, Min, Prod, All, Any

namespace tensorflow {
namespace mps {

// TODO: Extract from mps_pluggable_device_plugin.mm

void RegisterReductionOps(const char* platform_name, TF_Status* status) {
  // TODO: Registration
}

// Additional reduction operations stubs
extern "C" void* MPSReduceSum_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceSum not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceSum_Delete(void* kernel) {}

extern "C" void* MPSReduceMean_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceMean_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceMean not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceMean_Delete(void* kernel) {}

extern "C" void* MPSReduceMax_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceMax not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceMax_Delete(void* kernel) {}

extern "C" void* MPSReduceMin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceMin not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceMin_Delete(void* kernel) {}

extern "C" void* MPSReduceProd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceProd not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceProd_Delete(void* kernel) {}

extern "C" void* MPSReduceAll_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceAll_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceAll not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceAll_Delete(void* kernel) {}

extern "C" void* MPSReduceAny_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceAny_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceAny not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceAny_Delete(void* kernel) {}

extern "C" void* MPSReduceStd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceStd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceStd not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceStd_Delete(void* kernel) {}

extern "C" void* MPSReduceVariance_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceVariance_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceVariance not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceVariance_Delete(void* kernel) {}

extern "C" void* MPSReduceEuclideanNorm_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceEuclideanNorm_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceEuclideanNorm not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceEuclideanNorm_Delete(void* kernel) {}

extern "C" void* MPSReduceLogsumexp_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSReduceLogsumexp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ReduceLogsumexp not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSReduceLogsumexp_Delete(void* kernel) {}

extern "C" void* MPSCumSum_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSCumSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "CumSum not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSCumSum_Delete(void* kernel) {}

extern "C" void* MPSCumProd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSCumProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "CumProd not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSCumProd_Delete(void* kernel) {}

extern "C" void* MPSCumMax_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSCumMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "CumMax not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSCumMax_Delete(void* kernel) {}

extern "C" void* MPSCumMin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSCumMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "CumMin not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSCumMin_Delete(void* kernel) {}

extern "C" void* MPSSegmentSum_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SegmentSum not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSSegmentSum_Delete(void* kernel) {}

extern "C" void* MPSSegmentMean_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentMean_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SegmentMean not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSSegmentMean_Delete(void* kernel) {}

extern "C" void* MPSSegmentMax_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SegmentMax not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSSegmentMax_Delete(void* kernel) {}

extern "C" void* MPSSegmentMin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SegmentMin not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSSegmentMin_Delete(void* kernel) {}

extern "C" void* MPSSegmentProd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SegmentProd not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSSegmentProd_Delete(void* kernel) {}

extern "C" void* MPSUnsortedSegmentSum_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "UnsortedSegmentSum not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSUnsortedSegmentSum_Delete(void* kernel) {}

extern "C" void* MPSUnsortedSegmentMax_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "UnsortedSegmentMax not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSUnsortedSegmentMax_Delete(void* kernel) {}

extern "C" void* MPSUnsortedSegmentMin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "UnsortedSegmentMin not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSUnsortedSegmentMin_Delete(void* kernel) {}

extern "C" void* MPSUnsortedSegmentProd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "UnsortedSegmentProd not implemented for MPS");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
extern "C" void MPSUnsortedSegmentProd_Delete(void* kernel) {}

}  // namespace mps
}  // namespace tensorflow

