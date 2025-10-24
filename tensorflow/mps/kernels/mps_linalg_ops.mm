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
  
  TF_Tensor* diagonal = nullptr;
  TF_GetInput(ctx, 0, &diagonal, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // diagonal: [batch, n] -> output: [batch, n, n]
  int num_dims = TF_NumDims(diagonal);
  int64_t batch = (num_dims == 2) ? TF_Dim(diagonal, 0) : 1;
  int64_t n = (num_dims == 2) ? TF_Dim(diagonal, 1) : TF_Dim(diagonal, 0);
  
  size_t output_bytes = batch * n * n * sizeof(float);
  int64_t out_dims[] = {batch, n, n};
  TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, (num_dims == 2) ? 3 : 2, output_bytes, s);
  
  if (TF_GetCode(s) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  float* diag_data = static_cast<float*>(TF_TensorData(diagonal));
  float* out_data = static_cast<float*>(TF_TensorData(tf_output));
  
  // Zero out matrix
  memset(out_data, 0, output_bytes);
  
  // Fill diagonal
  for (int64_t b = 0; b < batch; b++) {
    for (int64_t i = 0; i < n; i++) {
      out_data[b * n * n + i * n + i] = diag_data[b * n + i];
    }
  }
  
  TF_DeleteStatus(s);
}
void MPSMatrixDiag_Delete(void* kernel) {}

void* MPSCholesky_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSCholesky_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // input: [batch, n, n] symmetric positive definite matrix
  int64_t batch = TF_Dim(input, 0);
  int64_t n = TF_Dim(input, 1);
  
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    
    float* input_data = static_cast<float*>(TF_TensorData(input));
    
    size_t matrix_bytes = batch * n * n * sizeof(float);
    int64_t out_dims[] = {batch, n, n};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 3, matrix_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    float* out_data = static_cast<float*>(TF_TensorData(tf_output));
    
    // Perform Cholesky decomposition for each batch
    for (int64_t b = 0; b < batch; b++) {
      MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                                        columns:n
                                                                       rowBytes:n * sizeof(float)
                                                                       dataType:MPSDataTypeFloat32];
      
      id<MTLBuffer> inputBuffer = [device newBufferWithBytes:&input_data[b * n * n]
                                                      length:n * n * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
      
      id<MTLBuffer> outputBuffer = [device newBufferWithLength:n * n * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
      
      MPSMatrix* inputMatrix = [[MPSMatrix alloc] initWithBuffer:inputBuffer descriptor:desc];
      MPSMatrix* outputMatrix = [[MPSMatrix alloc] initWithBuffer:outputBuffer descriptor:desc];
      
      MPSMatrixDecompositionCholesky* cholesky = [[MPSMatrixDecompositionCholesky alloc] 
                                                   initWithDevice:device rows:n columns:n];
      
      id<MTLCommandQueue> commandQueue = [device newCommandQueue];
      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      
      [cholesky encodeToCommandBuffer:commandBuffer
                         sourceMatrix:inputMatrix
                   resultMatrix:outputMatrix];
      
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      
      memcpy(&out_data[b * n * n], [outputBuffer contents], n * n * sizeof(float));
    }
  }
  
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
