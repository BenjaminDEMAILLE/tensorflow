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
#define PARSING_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Parsing-related ops (Batch 12)
PARSING_OP(ParseExample)
PARSING_OP(ParseSequenceExample)
PARSING_OP(ParseSingleSequenceExample)
PARSING_OP(ParseSingleExample)
PARSING_OP(ParseExampleV2)
PARSING_OP(ParseExampleV3)
PARSING_OP(ParseExampleV4)
PARSING_OP(ParseExampleV5)
PARSING_OP(ParseExampleV6)
PARSING_OP(ParseExampleV7)
PARSING_OP(ParseExampleV8)
PARSING_OP(ParseExampleV9)
PARSING_OP(ParseExampleV10)
PARSING_OP(ParseExampleV11)
PARSING_OP(ParseExampleV12)
PARSING_OP(ParseExampleV13)
PARSING_OP(ParseExampleV14)
PARSING_OP(ParseExampleV15)
PARSING_OP(ParseExampleV16)
PARSING_OP(ParseExampleV17)
PARSING_OP(ParseExampleV18)
PARSING_OP(ParseExampleV19)
PARSING_OP(ParseExampleV20)
PARSING_OP(ParseExampleV21)
PARSING_OP(ParseExampleV22)
PARSING_OP(ParseExampleV23)
PARSING_OP(ParseExampleV24)
PARSING_OP(ParseExampleV25)
PARSING_OP(ParseExampleV26)
PARSING_OP(ParseExampleV27)
PARSING_OP(ParseExampleV28)
PARSING_OP(ParseExampleV29)
PARSING_OP(ParseExampleV30)

#undef PARSING_OP
