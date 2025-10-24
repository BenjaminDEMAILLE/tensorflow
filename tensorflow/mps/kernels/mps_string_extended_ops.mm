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

// Additional string operations for MPS backend
// Note: String operations are typically CPU-only

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <string>

// ===== StringJoin =====
extern "C" void* MPSStringJoin_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringJoin_Delete(void* kernel) {}

extern "C" void MPSStringJoin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringJoin is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringSplit =====
extern "C" void* MPSStringSplit_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringSplit_Delete(void* kernel) {}

extern "C" void MPSStringSplit_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringSplit is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringLength =====
extern "C" void* MPSStringLength_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringLength_Delete(void* kernel) {}

extern "C" void MPSStringLength_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringLength is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringToHashBucket =====
extern "C" void* MPSStringToHashBucket_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringToHashBucket_Delete(void* kernel) {}

extern "C" void MPSStringToHashBucket_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringToHashBucket is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringToHashBucketFast =====
extern "C" void* MPSStringToHashBucketFast_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringToHashBucketFast_Delete(void* kernel) {}

extern "C" void MPSStringToHashBucketFast_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringToHashBucketFast is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== AsString =====
extern "C" void* MPSAsString_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSAsString_Delete(void* kernel) {}

extern "C" void MPSAsString_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "AsString is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== ReduceJoin =====
extern "C" void* MPSReduceJoin_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSReduceJoin_Delete(void* kernel) {}

extern "C" void MPSReduceJoin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "ReduceJoin is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringFormat =====
extern "C" void* MPSStringFormat_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringFormat_Delete(void* kernel) {}

extern "C" void MPSStringFormat_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringFormat is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringUpper =====
extern "C" void* MPSStringUpper_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringUpper_Delete(void* kernel) {}

extern "C" void MPSStringUpper_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringUpper is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringLower =====
extern "C" void* MPSStringLower_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringLower_Delete(void* kernel) {}

extern "C" void MPSStringLower_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringLower is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringStrip =====
extern "C" void* MPSStringStrip_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringStrip_Delete(void* kernel) {}

extern "C" void MPSStringStrip_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringStrip is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringReplace =====
extern "C" void* MPSStringReplace_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringReplace_Delete(void* kernel) {}

extern "C" void MPSStringReplace_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringReplace is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringLength =====
extern "C" void* MPSStringLength_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringLength_Delete(void* kernel) {}

extern "C" void MPSStringLength_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringLength is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringSubstr =====
extern "C" void* MPSStringSubstr_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringSubstr_Delete(void* kernel) {}

extern "C" void MPSStringSubstr_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringSubstr is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RegexReplace =====
extern "C" void* MPSRegexReplace_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRegexReplace_Delete(void* kernel) {}

extern "C" void MPSRegexReplace_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RegexReplace is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== RegexFullMatch =====
extern "C" void* MPSRegexFullMatch_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSRegexFullMatch_Delete(void* kernel) {}

extern "C" void MPSRegexFullMatch_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "RegexFullMatch is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StaticRegexReplace =====
extern "C" void* MPSStaticRegexReplace_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStaticRegexReplace_Delete(void* kernel) {}

extern "C" void MPSStaticRegexReplace_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StaticRegexReplace is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StaticRegexFullMatch =====
extern "C" void* MPSStaticRegexFullMatch_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStaticRegexFullMatch_Delete(void* kernel) {}

extern "C" void MPSStaticRegexFullMatch_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StaticRegexFullMatch is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== StringNGrams =====
extern "C" void* MPSStringNGrams_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSStringNGrams_Delete(void* kernel) {}

extern "C" void MPSStringNGrams_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "StringNGrams is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnicodeScript =====
extern "C" void* MPSUnicodeScript_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnicodeScript_Delete(void* kernel) {}

extern "C" void MPSUnicodeScript_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnicodeScript is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnicodeTranscode =====
extern "C" void* MPSUnicodeTranscode_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnicodeTranscode_Delete(void* kernel) {}

extern "C" void MPSUnicodeTranscode_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnicodeTranscode is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnicodeDecode =====
extern "C" void* MPSUnicodeDecode_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnicodeDecode_Delete(void* kernel) {}

extern "C" void MPSUnicodeDecode_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnicodeDecode is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

// ===== UnicodeEncode =====
extern "C" void* MPSUnicodeEncode_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

extern "C" void MPSUnicodeEncode_Delete(void* kernel) {}

extern "C" void MPSUnicodeEncode_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_SetStatus(status, TF_UNIMPLEMENTED, "UnicodeEncode is CPU-only operation");
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

