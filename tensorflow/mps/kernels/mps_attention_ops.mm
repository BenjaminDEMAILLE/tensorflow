/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Attention Operations
// ScaledDotProductAttention, MultiHeadAttention, FusedAttention, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

void* MPSScaledDotProductAttention_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSScaledDotProductAttention_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // Attention: softmax(Q @ K^T / sqrt(d_k)) @ V
  // Use MPSGraph matmul + softmax + matmul
  TF_SetStatus(s, TF_UNIMPLEMENTED, "ScaledDotProductAttention - TODO (MPSGraph matmul+softmax)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSScaledDotProductAttention_Delete(void* kernel) {}

void RegisterAttentionOps(const char* platform_name, TF_Status* status) {
  TF_KernelBuilder* kb = TF_NewKernelBuilder("Einsum", platform_name,
                                              &MPSScaledDotProductAttention_Create,
                                              &MPSScaledDotProductAttention_Compute,
                                              &MPSScaledDotProductAttention_Delete);
  TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSEinsum", kb, status);
  // TODO: 29+ more attention ops (MultiHeadAttention, CrossAttention, etc.)
}

}  // namespace mps
}  // namespace tensorflow
