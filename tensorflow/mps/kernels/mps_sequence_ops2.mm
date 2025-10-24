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
#define SEQUENCE_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Sequence-related ops (Batch 11)
SEQUENCE_OP(SequenceMask)
SEQUENCE_OP(SequenceLength)
SEQUENCE_OP(SequencePad)
SEQUENCE_OP(SequenceUnpad)
SEQUENCE_OP(SequenceConcat)
SEQUENCE_OP(SequenceSplit)
SEQUENCE_OP(SequenceGather)
SEQUENCE_OP(SequenceScatter)
SEQUENCE_OP(SequenceToDense)
SEQUENCE_OP(SequenceToSparse)
SEQUENCE_OP(SequenceToRagged)
SEQUENCE_OP(SequenceToTensor)
SEQUENCE_OP(SequenceToList)
SEQUENCE_OP(SequenceToArray)
SEQUENCE_OP(SequenceToTuple)
SEQUENCE_OP(SequenceToSet)
SEQUENCE_OP(SequenceToMap)
SEQUENCE_OP(SequenceToQueue)
SEQUENCE_OP(SequenceToStack)
SEQUENCE_OP(SequenceToHeap)
SEQUENCE_OP(SequenceToDeque)
SEQUENCE_OP(SequenceToPriorityQueue)
SEQUENCE_OP(SequenceToBag)
SEQUENCE_OP(SequenceToTree)
SEQUENCE_OP(SequenceToGraph)
SEQUENCE_OP(SequenceToTable)
SEQUENCE_OP(SequenceToHashTable)
SEQUENCE_OP(SequenceToBloomFilter)
SEQUENCE_OP(SequenceToTrie)
SEQUENCE_OP(SequenceToHeapQueue)
SEQUENCE_OP(SequenceToSkipList)

#undef SEQUENCE_OP
