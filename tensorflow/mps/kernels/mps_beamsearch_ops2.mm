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
#define BEAMSEARCH_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// BeamSearch-related ops (Batch 13)
BEAMSEARCH_OP(BeamSearch)
BEAMSEARCH_OP(BeamSearchV2)
BEAMSEARCH_OP(BeamSearchV3)
BEAMSEARCH_OP(BeamSearchV4)
BEAMSEARCH_OP(BeamSearchV5)
BEAMSEARCH_OP(BeamSearchDecoder)
BEAMSEARCH_OP(BeamSearchDecoderV2)
BEAMSEARCH_OP(BeamSearchDecoderV3)
BEAMSEARCH_OP(BeamSearchDecoderV4)
BEAMSEARCH_OP(BeamSearchDecoderV5)
BEAMSEARCH_OP(BeamSearchDecoderV6)
BEAMSEARCH_OP(BeamSearchDecoderV7)
BEAMSEARCH_OP(BeamSearchDecoderV8)
BEAMSEARCH_OP(BeamSearchDecoderV9)
BEAMSEARCH_OP(BeamSearchDecoderV10)
BEAMSEARCH_OP(BeamSearchDecoderV11)
BEAMSEARCH_OP(BeamSearchDecoderV12)
BEAMSEARCH_OP(BeamSearchDecoderV13)
BEAMSEARCH_OP(BeamSearchDecoderV14)
BEAMSEARCH_OP(BeamSearchDecoderV15)
BEAMSEARCH_OP(BeamSearchDecoderV16)
BEAMSEARCH_OP(BeamSearchDecoderV17)
BEAMSEARCH_OP(BeamSearchDecoderV18)
BEAMSEARCH_OP(BeamSearchDecoderV19)
BEAMSEARCH_OP(BeamSearchDecoderV20)
BEAMSEARCH_OP(BeamSearchDecoderV21)
BEAMSEARCH_OP(BeamSearchDecoderV22)
BEAMSEARCH_OP(BeamSearchDecoderV23)
BEAMSEARCH_OP(BeamSearchDecoderV24)
BEAMSEARCH_OP(BeamSearchDecoderV25)
BEAMSEARCH_OP(BeamSearchDecoderV26)
BEAMSEARCH_OP(BeamSearchDecoderV27)

#undef BEAMSEARCH_OP
