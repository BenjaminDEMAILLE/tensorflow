/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS String Operations
// StringJoin, StringSplit, StringToHashBucket, AsString, DecodeBase64, etc.

#import <Foundation/Foundation.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

void* MPSStringJoin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSStringJoin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // String ops on GPU not efficient, fallback to CPU
  TF_SetStatus(s, TF_UNIMPLEMENTED, "StringJoin - TODO (CPU fallback recommended)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSStringJoin_Delete(void* kernel) {}

void RegisterStringOps(const char* platform_name, TF_Status* status) {
  // Note: String ops typically run on CPU
  // TODO: 50+ string ops (StringSplit, StringFormat, StringLength, etc.)
  (void)platform_name;
  (void)status;
}

}  // namespace mps
}  // namespace tensorflow
