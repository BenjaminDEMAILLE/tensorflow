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
#define CTC_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// CTC-related ops (Batch 13)
CTC_OP(CTCLoss)
CTC_OP(CTCLossV2)
CTC_OP(CTCLossV3)
CTC_OP(CTCLossV4)
CTC_OP(CTCLossV5)
CTC_OP(CTCGreedyDecoder)
CTC_OP(CTCGreedyDecoderV2)
CTC_OP(CTCGreedyDecoderV3)
CTC_OP(CTCGreedyDecoderV4)
CTC_OP(CTCGreedyDecoderV5)
CTC_OP(CTCBeamSearchDecoder)
CTC_OP(CTCBeamSearchDecoderV2)
CTC_OP(CTCBeamSearchDecoderV3)
CTC_OP(CTCBeamSearchDecoderV4)
CTC_OP(CTCBeamSearchDecoderV5)
CTC_OP(CTCBeamSearchDecoderV6)
CTC_OP(CTCBeamSearchDecoderV7)
CTC_OP(CTCBeamSearchDecoderV8)
CTC_OP(CTCBeamSearchDecoderV9)
CTC_OP(CTCBeamSearchDecoderV10)
CTC_OP(CTCBeamSearchDecoderV11)
CTC_OP(CTCBeamSearchDecoderV12)
CTC_OP(CTCBeamSearchDecoderV13)
CTC_OP(CTCBeamSearchDecoderV14)
CTC_OP(CTCBeamSearchDecoderV15)
CTC_OP(CTCBeamSearchDecoderV16)
CTC_OP(CTCBeamSearchDecoderV17)
CTC_OP(CTCBeamSearchDecoderV18)
CTC_OP(CTCBeamSearchDecoderV19)
CTC_OP(CTCBeamSearchDecoderV20)
CTC_OP(CTCBeamSearchDecoderV21)
CTC_OP(CTCBeamSearchDecoderV22)

#undef CTC_OP
