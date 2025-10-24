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

// Debugging and utility operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <stdio.h>

// ===== Print =====
extern "C" void* MPSPrint_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSPrint_Delete(void* kernel) {}

extern "C" void MPSPrint_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  // Print op typically handled on CPU - just pass through input
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  // Forward input to output
  int nd = TF_NumDims(input);
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
  }
  
  TF_DataType dtype = TF_TensorType(input);
  size_t bytes = TF_TensorByteSize(input);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, dims, nd, bytes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  memcpy(TF_TensorData(output), TF_TensorData(input), bytes);
  
  TF_DeleteStatus(status);
}

// ===== Assert =====
extern "C" void* MPSAssert_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSAssert_Delete(void* kernel) {}

extern "C" void MPSAssert_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  // Assert op checks condition - CPU operation
  TF_Tensor* condition = nullptr;
  TF_GetInput(ctx, 0, &condition, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  // Check if condition is true
  if (TF_TensorType(condition) == TF_BOOL) {
    const bool* cond_data = (const bool*)TF_TensorData(condition);
    if (cond_data && !cond_data[0]) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "Assertion failed");
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
  }
  
  TF_DeleteStatus(status);
}

// ===== DebugIdentity =====
extern "C" void* MPSDebugIdentity_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDebugIdentity_Delete(void* kernel) {}

extern "C" void MPSDebugIdentity_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  // Identity with optional debug logging
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(input);
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
  }
  
  TF_DataType dtype = TF_TensorType(input);
  size_t bytes = TF_TensorByteSize(input);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, dims, nd, bytes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  memcpy(TF_TensorData(output), TF_TensorData(input), bytes);
  
  TF_DeleteStatus(status);
}

// ===== DebugNanCount =====
extern "C" void* MPSDebugNanCount_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDebugNanCount_Delete(void* kernel) {}

extern "C" void MPSDebugNanCount_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  // Count NaNs (CPU operation)
  int64_t nan_count = 0;
  if (TF_TensorType(input) == TF_FLOAT) {
    const float* data = (const float*)TF_TensorData(input);
    size_t num_elems = TF_TensorByteSize(input) / sizeof(float);
    for (size_t i = 0; i < num_elems; ++i) {
      if (isnan(data[i])) {
        nan_count++;
      }
    }
  }
  
  // Output is a scalar int64
  int64_t out_dims[1] = {1};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_INT64, out_dims, 1, sizeof(int64_t), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int64_t* out_data = (int64_t*)TF_TensorData(output);
  out_data[0] = nan_count;
  
  TF_DeleteStatus(status);
}

// ===== DebugNumericSummary =====
extern "C" void* MPSDebugNumericSummary_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSDebugNumericSummary_Delete(void* kernel) {}

extern "C" void MPSDebugNumericSummary_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "DebugNumericSummary not yet implemented on MPS");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StopGradient =====
extern "C" void* MPSStopGradient_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStopGradient_Delete(void* kernel) {}

extern "C" void MPSStopGradient_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  // StopGradient is identity during forward pass
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(input);
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
  }
  
  TF_DataType dtype = TF_TensorType(input);
  size_t bytes = TF_TensorByteSize(input);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, dims, nd, bytes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  memcpy(TF_TensorData(output), TF_TensorData(input), bytes);
  
  TF_DeleteStatus(status);
}
