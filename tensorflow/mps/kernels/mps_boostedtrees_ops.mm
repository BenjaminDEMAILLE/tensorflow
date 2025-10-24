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

// Boosted trees operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== BoostedTreesCreateEnsemble =====
extern "C" void* MPSBoostedTreesCreateEnsemble_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesCreateEnsemble_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesCreateEnsemble_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesCreateEnsemble is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesPredict =====
extern "C" void* MPSBoostedTreesPredict_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesPredict_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesPredict_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesPredict is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesTrainingPredict =====
extern "C" void* MPSBoostedTreesTrainingPredict_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesTrainingPredict_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesTrainingPredict_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesTrainingPredict is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesUpdateEnsemble =====
extern "C" void* MPSBoostedTreesUpdateEnsemble_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesUpdateEnsemble_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesUpdateEnsemble_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesUpdateEnsemble is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesCalculateBestGainsPerFeature =====
extern "C" void* MPSBoostedTreesCalculateBestGainsPerFeature_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesCalculateBestGainsPerFeature_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesCalculateBestGainsPerFeature_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesCalculateBestGainsPerFeature is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesMakeStatsSummary =====
extern "C" void* MPSBoostedTreesMakeStatsSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesMakeStatsSummary_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesMakeStatsSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesMakeStatsSummary is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesAggregateStats =====
extern "C" void* MPSBoostedTreesAggregateStats_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesAggregateStats_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesAggregateStats_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesAggregateStats is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesQuantileStreamResourceAdd =====
extern "C" void* MPSBoostedTreesQuantileStreamResourceAdd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesQuantileStreamResourceAdd_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesQuantileStreamResourceAdd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesQuantileStreamResourceAdd is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
