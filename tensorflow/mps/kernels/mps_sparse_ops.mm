/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Sparse Tensor Operations
// SparseToDense, SparseSoftmax, SparseMatMul, SparseTensorDenseMatMul, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// SparseToDense
void* MPSSparseToDense_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSSparseToDense_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "SparseToDense - TODO");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSSparseToDense_Delete(void* kernel) {}

// SparseMatMul, SparseSoftmax... (100+ ops)

void RegisterSparseOps(const char* platform_name, TF_Status* status) {
  TF_KernelBuilder* kb = TF_NewKernelBuilder("SparseToDense", platform_name,
                                              &MPSSparseToDense_Create,
                                              &MPSSparseToDense_Compute,
                                              &MPSSparseToDense_Delete);
  TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSparseToDenseFloat", kb, status);
  // TODO: 99+ more sparse ops
}

}  // namespace mps
}  // namespace tensorflow
