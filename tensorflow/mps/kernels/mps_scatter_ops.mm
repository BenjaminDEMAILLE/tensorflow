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
#define SCATTER_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Scatter-related ops (Batch 11)
SCATTER_OP(ScatterNd)
SCATTER_OP(ScatterNdAdd)
SCATTER_OP(ScatterNdSub)
SCATTER_OP(ScatterNdUpdate)
SCATTER_OP(ScatterNdMax)
SCATTER_OP(ScatterNdMin)
SCATTER_OP(ScatterNdMul)
SCATTER_OP(ScatterNdDiv)
SCATTER_OP(ScatterNdMean)
SCATTER_OP(ScatterNdSum)
SCATTER_OP(ScatterNdProd)
SCATTER_OP(ScatterNdSqrt)
SCATTER_OP(ScatterNdLog)
SCATTER_OP(ScatterNdExp)
SCATTER_OP(ScatterNdAbs)
SCATTER_OP(ScatterNdNeg)
SCATTER_OP(ScatterNdSquare)
SCATTER_OP(ScatterNdClip)
SCATTER_OP(ScatterNdFloor)
SCATTER_OP(ScatterNdCeil)
SCATTER_OP(ScatterNdRound)
SCATTER_OP(ScatterNdTrunc)
SCATTER_OP(ScatterNdCumsum)
SCATTER_OP(ScatterNdCumprod)
SCATTER_OP(ScatterNdArgmax)
SCATTER_OP(ScatterNdArgmin)
SCATTER_OP(ScatterNdSoftmax)
SCATTER_OP(ScatterNdLogSoftmax)
SCATTER_OP(ScatterNdRelu)
SCATTER_OP(ScatterNdSigmoid)

#undef SCATTER_OP
