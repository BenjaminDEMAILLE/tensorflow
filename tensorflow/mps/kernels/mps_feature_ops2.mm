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
#define FEATURE_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Feature-related ops (Batch 12)
FEATURE_OP(FeatureColumn)
FEATURE_OP(FeatureColumnV2)
FEATURE_OP(FeatureColumnV3)
FEATURE_OP(FeatureColumnV4)
FEATURE_OP(FeatureColumnV5)
FEATURE_OP(FeatureColumnV6)
FEATURE_OP(FeatureColumnV7)
FEATURE_OP(FeatureColumnV8)
FEATURE_OP(FeatureColumnV9)
FEATURE_OP(FeatureColumnV10)
FEATURE_OP(FeatureColumnV11)
FEATURE_OP(FeatureColumnV12)
FEATURE_OP(FeatureColumnV13)
FEATURE_OP(FeatureColumnV14)
FEATURE_OP(FeatureColumnV15)
FEATURE_OP(FeatureColumnV16)
FEATURE_OP(FeatureColumnV17)
FEATURE_OP(FeatureColumnV18)
FEATURE_OP(FeatureColumnV19)
FEATURE_OP(FeatureColumnV20)
FEATURE_OP(FeatureColumnV21)
FEATURE_OP(FeatureColumnV22)
FEATURE_OP(FeatureColumnV23)
FEATURE_OP(FeatureColumnV24)
FEATURE_OP(FeatureColumnV25)
FEATURE_OP(FeatureColumnV26)
FEATURE_OP(FeatureColumnV27)
FEATURE_OP(FeatureColumnV28)
FEATURE_OP(FeatureColumnV29)
FEATURE_OP(FeatureColumnV30)

#undef FEATURE_OP
