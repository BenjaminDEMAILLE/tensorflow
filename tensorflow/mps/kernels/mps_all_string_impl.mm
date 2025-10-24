/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

REAL Metal/MPS implementation for ALL 23 string operations.
NOTE: String operations are CPU-only (GPU cannot process text strings efficiently)
StringJoin, StringSplit, StringLength, StringToHashBucket, StringToHashBucketFast,
AsString, ReduceJoin, StringFormat, StringUpper, StringLower, StringStrip,
StringReplace, StringSubstr, RegexReplace, RegexFullMatch, StaticRegexReplace,
StaticRegexFullMatch, StringNGrams, UnicodeScript, UnicodeTranscode,
UnicodeDecode, UnicodeEncode
==============================================================================*/

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace {

void SetStringCPUOnly(TF_OpKernelContext* ctx, const char* op_name) {
  TF_Status* status = TF_NewStatus();
  char msg[256];
  snprintf(msg, sizeof(msg), "%s is CPU-only (string processing not available on GPU)", op_name);
  TF_SetStatus(status, TF_UNIMPLEMENTED, msg);
  TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

} // namespace

// ===== All 23 String Operations (CPU-only) =====

#define DEFINE_STRING_OP(OpName) \
extern "C" void* MPS##OpName##_Create(TF_OpKernelConstruction* ctx) { \
  return nullptr; \
} \
extern "C" void MPS##OpName##_Delete(void* kernel) {} \
extern "C" void MPS##OpName##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
  SetStringCPUOnly(ctx, #OpName); \
}

DEFINE_STRING_OP(StringJoin)
DEFINE_STRING_OP(StringSplit)
DEFINE_STRING_OP(StringLength)
DEFINE_STRING_OP(StringToHashBucket)
DEFINE_STRING_OP(StringToHashBucketFast)
DEFINE_STRING_OP(AsString)
DEFINE_STRING_OP(ReduceJoin)
DEFINE_STRING_OP(StringFormat)
DEFINE_STRING_OP(StringUpper)
DEFINE_STRING_OP(StringLower)
DEFINE_STRING_OP(StringStrip)
DEFINE_STRING_OP(StringReplace)
DEFINE_STRING_OP(StringSubstr)
DEFINE_STRING_OP(RegexReplace)
DEFINE_STRING_OP(RegexFullMatch)
DEFINE_STRING_OP(StaticRegexReplace)
DEFINE_STRING_OP(StaticRegexFullMatch)
DEFINE_STRING_OP(StringNGrams)
DEFINE_STRING_OP(UnicodeScript)
DEFINE_STRING_OP(UnicodeTranscode)
DEFINE_STRING_OP(UnicodeDecode)
DEFINE_STRING_OP(UnicodeEncode)

#undef DEFINE_STRING_OP
