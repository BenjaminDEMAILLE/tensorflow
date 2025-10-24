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

// Extended linear algebra operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ===== Cholesky =====
extern "C" void* MPSCholesky_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCholesky_Delete(void* kernel) {}

extern "C" void MPSCholesky_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Cholesky requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== CholeskyGrad =====
extern "C" void* MPSCholeskyGrad_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSCholeskyGrad_Delete(void* kernel) {}

extern "C" void MPSCholeskyGrad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "CholeskyGrad requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== MatrixInverse =====
extern "C" void* MPSMatrixInverse_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMatrixInverse_Delete(void* kernel) {}

extern "C" void MPSMatrixInverse_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MatrixInverse requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== MatrixDeterminant =====
extern "C" void* MPSMatrixDeterminant_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMatrixDeterminant_Delete(void* kernel) {}

extern "C" void MPSMatrixDeterminant_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MatrixDeterminant requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== LogMatrixDeterminant =====
extern "C" void* MPSLogMatrixDeterminant_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSLogMatrixDeterminant_Delete(void* kernel) {}

extern "C" void MPSLogMatrixDeterminant_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "LogMatrixDeterminant requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Qr =====
extern "C" void* MPSQr_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSQr_Delete(void* kernel) {}

extern "C" void MPSQr_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Qr requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Svd =====
extern "C" void* MPSSvd_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSvd_Delete(void* kernel) {}

extern "C" void MPSSvd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Svd requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== Eig =====
extern "C" void* MPSEig_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSEig_Delete(void* kernel) {}

extern "C" void MPSEig_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "Eig requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== SelfAdjointEig =====
extern "C" void* MPSSelfAdjointEig_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSSelfAdjointEig_Delete(void* kernel) {}

extern "C" void MPSSelfAdjointEig_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "SelfAdjointEig requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== MatrixSolve =====
extern "C" void* MPSMatrixSolve_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSMatrixSolve_Delete(void* kernel) {}

extern "C" void MPSMatrixSolve_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "MatrixSolve requires LAPACK/Accelerate");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}
