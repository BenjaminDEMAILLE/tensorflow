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
#define BOOSTEDTREES_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// BoostedTrees-related ops (Batch 12)
BOOSTEDTREES_OP(BoostedTreesEnsembleResourceHandle)
BOOSTEDTREES_OP(BoostedTreesMakeStatsSummary)
BOOSTEDTREES_OP(BoostedTreesCalculateBestGainsPerFeature)
BOOSTEDTREES_OP(BoostedTreesCreateQuantileStreamResource)
BOOSTEDTREES_OP(BoostedTreesFlushQuantileStreamResource)
BOOSTEDTREES_OP(BoostedTreesGetBucketBoundaries)
BOOSTEDTREES_OP(BoostedTreesQuantileStreamResourceAddSummaries)
BOOSTEDTREES_OP(BoostedTreesQuantileStreamResourceDeserialize)
BOOSTEDTREES_OP(BoostedTreesQuantileStreamResourceFlush)
BOOSTEDTREES_OP(BoostedTreesQuantileStreamResourceGetBucketBoundaries)
BOOSTEDTREES_OP(BoostedTreesQuantileStreamResourceHandle)
BOOSTEDTREES_OP(BoostedTreesQuantileStreamResourceIsInitialized)
BOOSTEDTREES_OP(BoostedTreesQuantileStreamResourceSerialize)
BOOSTEDTREES_OP(BoostedTreesSerializeEnsemble)
BOOSTEDTREES_OP(BoostedTreesTrainingPredict)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsemble)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV2)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV3)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV4)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV5)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV6)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV7)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV8)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV9)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV10)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV11)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV12)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV13)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV14)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV15)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV16)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV17)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV18)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV19)
BOOSTEDTREES_OP(BoostedTreesUpdateEnsembleV20)

#undef BOOSTEDTREES_OP
