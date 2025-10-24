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

// Feature engineering operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== Bucketize =====
extern "C" void* MPSBucketize_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBucketize_Delete(void* kernel) {}

extern "C" void MPSBucketize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Bucketize requires bucketing logic");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CrossedColumn =====
extern "C" void* MPSCrossedColumn_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCrossedColumn_Delete(void* kernel) {}

extern "C" void MPSCrossedColumn_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CrossedColumn is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== QuantileAccumulator =====
extern "C" void* MPSQuantileAccumulator_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSQuantileAccumulator_Delete(void* kernel) {}

extern "C" void MPSQuantileAccumulator_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "QuantileAccumulator is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== FeatureNormalization =====
extern "C" void* MPSFeatureNormalization_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSFeatureNormalization_Delete(void* kernel) {}

extern "C" void MPSFeatureNormalization_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FeatureNormalization requires normalization");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== BoostedTreesBucketize =====
extern "C" void* MPSBoostedTreesBucketize_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBoostedTreesBucketize_Delete(void* kernel) {}

extern "C" void MPSBoostedTreesBucketize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "BoostedTreesBucketize is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
