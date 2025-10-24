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

// Bitwise operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

// ===== BitwiseAnd =====
extern "C" void* MPSBitwiseAnd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBitwiseAnd_Delete(void* kernel) {}

extern "C" void MPSBitwiseAnd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* a = nullptr;
  TF_GetInput(ctx, 0, &a, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 1, &b, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  // CPU fallback for bitwise operations
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(a, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(a);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (dtype == TF_INT32) {
    const int32_t* pa = (const int32_t*)TF_TensorData(a);
    const int32_t* pb = (const int32_t*)TF_TensorData(b);
    int32_t* po = (int32_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = pa[i] & pb[i];
    }
  } else if (dtype == TF_INT64) {
    const int64_t* pa = (const int64_t*)TF_TensorData(a);
    const int64_t* pb = (const int64_t*)TF_TensorData(b);
    int64_t* po = (int64_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = pa[i] & pb[i];
    }
  }
  
  TF_DeleteStatus(status);
}

// ===== BitwiseOr =====
extern "C" void* MPSBitwiseOr_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBitwiseOr_Delete(void* kernel) {}

extern "C" void MPSBitwiseOr_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* a = nullptr;
  TF_GetInput(ctx, 0, &a, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 1, &b, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(a, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(a);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (dtype == TF_INT32) {
    const int32_t* pa = (const int32_t*)TF_TensorData(a);
    const int32_t* pb = (const int32_t*)TF_TensorData(b);
    int32_t* po = (int32_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = pa[i] | pb[i];
    }
  } else if (dtype == TF_INT64) {
    const int64_t* pa = (const int64_t*)TF_TensorData(a);
    const int64_t* pb = (const int64_t*)TF_TensorData(b);
    int64_t* po = (int64_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = pa[i] | pb[i];
    }
  }
  
  TF_DeleteStatus(status);
}

// ===== BitwiseXor =====
extern "C" void* MPSBitwiseXor_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSBitwiseXor_Delete(void* kernel) {}

extern "C" void MPSBitwiseXor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* a = nullptr;
  TF_GetInput(ctx, 0, &a, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 1, &b, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(a, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(a);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (dtype == TF_INT32) {
    const int32_t* pa = (const int32_t*)TF_TensorData(a);
    const int32_t* pb = (const int32_t*)TF_TensorData(b);
    int32_t* po = (int32_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = pa[i] ^ pb[i];
    }
  } else if (dtype == TF_INT64) {
    const int64_t* pa = (const int64_t*)TF_TensorData(a);
    const int64_t* pb = (const int64_t*)TF_TensorData(b);
    int64_t* po = (int64_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = pa[i] ^ pb[i];
    }
  }
  
  TF_DeleteStatus(status);
}

// ===== Invert =====
extern "C" void* MPSInvert_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSInvert_Delete(void* kernel) {}

extern "C" void MPSInvert_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(input);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (dtype == TF_INT32) {
    const int32_t* pin = (const int32_t*)TF_TensorData(input);
    int32_t* pout = (int32_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      pout[i] = ~pin[i];
    }
  } else if (dtype == TF_INT64) {
    const int64_t* pin = (const int64_t*)TF_TensorData(input);
    int64_t* pout = (int64_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      pout[i] = ~pin[i];
    }
  }
  
  TF_DeleteStatus(status);
}

// ===== LeftShift =====
extern "C" void* MPSLeftShift_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSLeftShift_Delete(void* kernel) {}

extern "C" void MPSLeftShift_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* y = nullptr;
  TF_GetInput(ctx, 1, &y, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(x);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(x, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(x);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (dtype == TF_INT32) {
    const int32_t* px = (const int32_t*)TF_TensorData(x);
    const int32_t* py = (const int32_t*)TF_TensorData(y);
    int32_t* po = (int32_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = px[i] << py[i];
    }
  } else if (dtype == TF_INT64) {
    const int64_t* px = (const int64_t*)TF_TensorData(x);
    const int64_t* py = (const int64_t*)TF_TensorData(y);
    int64_t* po = (int64_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = px[i] << py[i];
    }
  }
  
  TF_DeleteStatus(status);
}

// ===== RightShift =====
extern "C" void* MPSRightShift_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRightShift_Delete(void* kernel) {}

extern "C" void MPSRightShift_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* x = nullptr;
  TF_GetInput(ctx, 0, &x, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* y = nullptr;
  TF_GetInput(ctx, 1, &y, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(x);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(x, i);
    nelems *= dims[i];
  }
  
  TF_DataType dtype = TF_TensorType(x);
  size_t elem_size = (dtype == TF_INT32) ? 4 : 8;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype, dims, nd, nelems * elem_size, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  if (dtype == TF_INT32) {
    const int32_t* px = (const int32_t*)TF_TensorData(x);
    const int32_t* py = (const int32_t*)TF_TensorData(y);
    int32_t* po = (int32_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = px[i] >> py[i];
    }
  } else if (dtype == TF_INT64) {
    const int64_t* px = (const int64_t*)TF_TensorData(x);
    const int64_t* py = (const int64_t*)TF_TensorData(y);
    int64_t* po = (int64_t*)TF_TensorData(out);
    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = px[i] >> py[i];
    }
  }
  
  TF_DeleteStatus(status);
}
