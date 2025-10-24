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
#define RANDOM_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Random-related ops (Batch 14)
RANDOM_OP(RandomUniform)
RANDOM_OP(RandomUniformV2)
RANDOM_OP(RandomUniformV3)
RANDOM_OP(RandomUniformInt)
RANDOM_OP(RandomUniformIntV2)
RANDOM_OP(RandomUniformIntV3)
RANDOM_OP(RandomStandardNormal)
RANDOM_OP(RandomStandardNormalV2)
RANDOM_OP(RandomStandardNormalV3)
RANDOM_OP(RandomNormal)
RANDOM_OP(RandomNormalV2)
RANDOM_OP(RandomNormalV3)
RANDOM_OP(TruncatedNormal)
RANDOM_OP(TruncatedNormalV2)
RANDOM_OP(TruncatedNormalV3)
RANDOM_OP(RandomGamma)
RANDOM_OP(RandomGammaV2)
RANDOM_OP(RandomGammaV3)
RANDOM_OP(RandomPoisson)
RANDOM_OP(RandomPoissonV2)
RANDOM_OP(RandomPoissonV3)
RANDOM_OP(RandomShuffle)
RANDOM_OP(RandomShuffleV2)
RANDOM_OP(RandomShuffleV3)
RANDOM_OP(RandomCrop)
RANDOM_OP(RandomCropV2)
RANDOM_OP(RandomCropV3)
RANDOM_OP(Multinomial)
RANDOM_OP(MultinomialV2)
RANDOM_OP(MultinomialV3)
RANDOM_OP(ParameterizedTruncatedNormal)
RANDOM_OP(ParameterizedTruncatedNormalV2)
RANDOM_OP(ParameterizedTruncatedNormalV3)

#undef RANDOM_OP
