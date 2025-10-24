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
#define OPTIMIZER_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Optimizer-related ops (Batch 13)
OPTIMIZER_OP(ApplyGradientDescent)
OPTIMIZER_OP(ApplyGradientDescentV2)
OPTIMIZER_OP(ApplyProximalGradientDescent)
OPTIMIZER_OP(ApplyProximalGradientDescentV2)
OPTIMIZER_OP(ApplyAdadelta)
OPTIMIZER_OP(ApplyAdadeltaV2)
OPTIMIZER_OP(ApplyAdagrad)
OPTIMIZER_OP(ApplyAdagradV2)
OPTIMIZER_OP(ApplyAdagradDA)
OPTIMIZER_OP(ApplyAdagradDAV2)
OPTIMIZER_OP(ApplyFtrl)
OPTIMIZER_OP(ApplyFtrlV2)
OPTIMIZER_OP(ApplyMomentum)
OPTIMIZER_OP(ApplyMomentumV2)
OPTIMIZER_OP(ApplyAdam)
OPTIMIZER_OP(ApplyAdamV2)
OPTIMIZER_OP(ApplyAdaMax)
OPTIMIZER_OP(ApplyAdaMaxV2)
OPTIMIZER_OP(ApplyRMSProp)
OPTIMIZER_OP(ApplyRMSPropV2)
OPTIMIZER_OP(ApplyCenteredRMSProp)
OPTIMIZER_OP(ApplyCenteredRMSPropV2)
OPTIMIZER_OP(ApplyAddSign)
OPTIMIZER_OP(ApplyAddSignV2)
OPTIMIZER_OP(ApplyPowerSign)
OPTIMIZER_OP(ApplyPowerSignV2)
OPTIMIZER_OP(SparseApplyAdadelta)
OPTIMIZER_OP(SparseApplyAdadeltaV2)
OPTIMIZER_OP(SparseApplyAdagrad)
OPTIMIZER_OP(SparseApplyAdagradV2)
OPTIMIZER_OP(SparseApplyAdagradDA)
OPTIMIZER_OP(SparseApplyAdagradDAV2)

#undef OPTIMIZER_OP
