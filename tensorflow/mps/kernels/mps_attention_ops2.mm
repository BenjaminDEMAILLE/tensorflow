// Copyright 2025 The TensorFlow Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

// Macros for repetitive stub generation
#define ATTENTION_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Attention-related ops (Batch 13)
ATTENTION_OP(Attention)
ATTENTION_OP(AttentionV2)
ATTENTION_OP(AttentionV3)
ATTENTION_OP(AttentionV4)
ATTENTION_OP(AttentionV5)
ATTENTION_OP(MultiHeadAttention)
ATTENTION_OP(MultiHeadAttentionV2)
ATTENTION_OP(MultiHeadAttentionV3)
ATTENTION_OP(MultiHeadAttentionV4)
ATTENTION_OP(MultiHeadAttentionV5)
ATTENTION_OP(SelfAttention)
ATTENTION_OP(SelfAttentionV2)
ATTENTION_OP(SelfAttentionV3)
ATTENTION_OP(SelfAttentionV4)
ATTENTION_OP(SelfAttentionV5)
ATTENTION_OP(CrossAttention)
ATTENTION_OP(CrossAttentionV2)
ATTENTION_OP(CrossAttentionV3)
ATTENTION_OP(CrossAttentionV4)
ATTENTION_OP(CrossAttentionV5)
ATTENTION_OP(BahdanauAttention)
ATTENTION_OP(BahdanauAttentionV2)
ATTENTION_OP(BahdanauAttentionV3)
ATTENTION_OP(LuongAttention)
ATTENTION_OP(LuongAttentionV2)
ATTENTION_OP(LuongAttentionV3)
ATTENTION_OP(AdditiveAttention)
ATTENTION_OP(AdditiveAttentionV2)
ATTENTION_OP(DotProductAttention)
ATTENTION_OP(DotProductAttentionV2)
ATTENTION_OP(ScaledDotProductAttention)
ATTENTION_OP(ScaledDotProductAttentionV2)

#undef ATTENTION_OP
