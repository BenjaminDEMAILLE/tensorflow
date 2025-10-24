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
#define RAGGED_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Ragged-related ops (Batch 11)
RAGGED_OP(RaggedTensorToTensor)
RAGGED_OP(RaggedTensorToSparse)
RAGGED_OP(RaggedTensorToVariant)
RAGGED_OP(RaggedTensorToDense)
RAGGED_OP(RaggedTensorToList)
RAGGED_OP(RaggedTensorToArray)
RAGGED_OP(RaggedTensorToTuple)
RAGGED_OP(RaggedTensorToSet)
RAGGED_OP(RaggedTensorToMap)
RAGGED_OP(RaggedTensorToQueue)
RAGGED_OP(RaggedTensorToStack)
RAGGED_OP(RaggedTensorToHeap)
RAGGED_OP(RaggedTensorToDeque)
RAGGED_OP(RaggedTensorToPriorityQueue)
RAGGED_OP(RaggedTensorToBag)
RAGGED_OP(RaggedTensorToTree)
RAGGED_OP(RaggedTensorToGraph)
RAGGED_OP(RaggedTensorToTable)
RAGGED_OP(RaggedTensorToHashTable)
RAGGED_OP(RaggedTensorToBloomFilter)
RAGGED_OP(RaggedTensorToTrie)
RAGGED_OP(RaggedTensorToHeapQueue)
RAGGED_OP(RaggedTensorToSkipList)
RAGGED_OP(RaggedTensorToFlat)
RAGGED_OP(RaggedTensorToNested)
RAGGED_OP(RaggedTensorToPacked)
RAGGED_OP(RaggedTensorToUnpacked)
RAGGED_OP(RaggedTensorToIndexed)
RAGGED_OP(RaggedTensorToUnindexed)
RAGGED_OP(RaggedTensorToSorted)

#undef RAGGED_OP
