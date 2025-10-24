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

// Candidate sampling operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== UniformCandidateSampler =====
extern "C" void* MPSUniformCandidateSampler_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUniformCandidateSampler_Delete(void* kernel) {}

extern "C" void MPSUniformCandidateSampler_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UniformCandidateSampler is CPU-only (sampling)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== LogUniformCandidateSampler =====
extern "C" void* MPSLogUniformCandidateSampler_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSLogUniformCandidateSampler_Delete(void* kernel) {}

extern "C" void MPSLogUniformCandidateSampler_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LogUniformCandidateSampler is CPU-only (sampling)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== LearnedUnigramCandidateSampler =====
extern "C" void* MPSLearnedUnigramCandidateSampler_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSLearnedUnigramCandidateSampler_Delete(void* kernel) {}

extern "C" void MPSLearnedUnigramCandidateSampler_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LearnedUnigramCandidateSampler is CPU-only (sampling)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== FixedUnigramCandidateSampler =====
extern "C" void* MPSFixedUnigramCandidateSampler_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSFixedUnigramCandidateSampler_Delete(void* kernel) {}

extern "C" void MPSFixedUnigramCandidateSampler_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "FixedUnigramCandidateSampler is CPU-only (sampling)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== AllCandidateSampler =====
extern "C" void* MPSAllCandidateSampler_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSAllCandidateSampler_Delete(void* kernel) {}

extern "C" void MPSAllCandidateSampler_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "AllCandidateSampler is CPU-only (sampling)");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ComputeAccidentalHits =====
extern "C" void* MPSComputeAccidentalHits_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSComputeAccidentalHits_Delete(void* kernel) {}

extern "C" void MPSComputeAccidentalHits_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ComputeAccidentalHits is CPU-only");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
