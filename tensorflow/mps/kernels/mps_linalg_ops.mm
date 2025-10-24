/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Linear Algebra Operations
// MatrixDiag, MatrixSolve, Cholesky, QR, SVD, Eig, MatrixInverse, Det, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

void* MPSMatrixDiag_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSMatrixDiag_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // MatrixDiag: create diagonal matrix from vector
  TF_SetStatus(s, TF_UNIMPLEMENTED, "MatrixDiag - TODO (Metal kernel)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSMatrixDiag_Delete(void* kernel) {}

void* MPSCholesky_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSCholesky_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // Cholesky decomposition: A = L @ L^T
  // Use MPSMatrixDecompositionCholesky
  TF_SetStatus(s, TF_UNIMPLEMENTED, "Cholesky - TODO (MPSMatrixDecompositionCholesky)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSCholesky_Delete(void* kernel) {}

void* MPSQR_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSQR_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // QR decomposition: A = Q @ R
  TF_SetStatus(s, TF_UNIMPLEMENTED, "QR - TODO (Metal QR decomposition)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSQR_Delete(void* kernel) {}

void* MPSSVD_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSSVD_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // Singular Value Decomposition: A = U @ S @ V^T
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SVD - TODO (Metal SVD)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSSVD_Delete(void* kernel) {}

void* MPSMatrixInverse_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSMatrixInverse_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // Matrix inverse: A^-1
  TF_SetStatus(s, TF_UNIMPLEMENTED, "MatrixInverse - TODO (Metal inverse)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSMatrixInverse_Delete(void* kernel) {}

void RegisterLinAlgOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MatrixDiag", platform_name,
                                                &MPSMatrixDiag_Create,
                                                &MPSMatrixDiag_Compute,
                                                &MPSMatrixDiag_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMatrixDiag", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Cholesky", platform_name,
                                                &MPSCholesky_Create,
                                                &MPSCholesky_Compute,
                                                &MPSCholesky_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSCholesky", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Qr", platform_name,
                                                &MPSQR_Create,
                                                &MPSQR_Compute,
                                                &MPSQR_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSQR", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Svd", platform_name,
                                                &MPSSVD_Create,
                                                &MPSSVD_Compute,
                                                &MPSSVD_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSVD", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MatrixInverse", platform_name,
                                                &MPSMatrixInverse_Create,
                                                &MPSMatrixInverse_Compute,
                                                &MPSMatrixInverse_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMatrixInverse", kb, status);
  }
  // TODO: 195+ more linalg ops
  // MatrixSolve, MatrixTriangularSolve, MatrixDeterminant, MatrixSetDiag
  // Eig, SelfAdjointEig, Lu, LogMatrixDeterminant, MatrixSquareRoot
  // MatrixExponential, BandedTriangularSolve, Tridiagonal, etc.
}

}  // namespace mps
}  // namespace tensorflow
