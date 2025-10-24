/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Embedding Operations
// EmbeddingLookup, EmbeddingLookupSparse, SafeEmbeddingLookupSparse, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

void* MPSEmbeddingLookup_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSEmbeddingLookup_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // EmbeddingLookup: params[ids] - gather operation
  TF_SetStatus(s, TF_UNIMPLEMENTED, "EmbeddingLookup - TODO (Metal gather kernel)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSEmbeddingLookup_Delete(void* kernel) {}

void RegisterEmbeddingOps(const char* platform_name, TF_Status* status) {
  TF_KernelBuilder* kb = TF_NewKernelBuilder("GatherV2", platform_name,
                                              &MPSEmbeddingLookup_Create,
                                              &MPSEmbeddingLookup_Compute,
                                              &MPSEmbeddingLookup_Delete);
  TF_KernelBuilder_TypeConstraint(kb, "Tparams", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSGatherV2Float", kb, status);
  // TODO: 14+ more embedding ops
}

}  // namespace mps
}  // namespace tensorflow
