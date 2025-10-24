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

// Clip and clamping operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include <algorithm>

// ===== ClipByValue =====
extern "C" void* MPSClipByValue_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSClipByValue_Delete(void* kernel) {}

extern "C" void MPSClipByValue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_Tensor* clip_min = nullptr;
  TF_Tensor* clip_max = nullptr;
  
  TF_GetInput(ctx, 0, &input, status);
  TF_GetInput(ctx, 1, &clip_min, status);
  TF_GetInput(ctx, 2, &clip_max, status);
  
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  int nd = TF_NumDims(input);
  int64_t dims[8];
  int64_t nelems = 1;
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
    nelems *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  
  const float* in = (const float*)TF_TensorData(input);
  const float* min_val = (const float*)TF_TensorData(clip_min);
  const float* max_val = (const float*)TF_TensorData(clip_max);
  float* out = (float*)TF_TensorData(output);
  
  float min_scalar = min_val[0];
  float max_scalar = max_val[0];
  
  // CPU-based clipping
  for (int64_t i = 0; i < nelems; ++i) {
    out[i] = std::max(min_scalar, std::min(max_scalar, in[i]));
  }
  
  TF_DeleteStatus(status);
}

// ===== ClipByNorm =====
extern "C" void* MPSClipByNorm_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSClipByNorm_Delete(void* kernel) {}

extern "C" void MPSClipByNorm_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ClipByNorm requires norm computation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ClipByGlobalNorm =====
extern "C" void* MPSClipByGlobalNorm_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSClipByGlobalNorm_Delete(void* kernel) {}

extern "C" void MPSClipByGlobalNorm_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ClipByGlobalNorm requires global norm computation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
