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
#define RNN_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// RNN-related ops (Batch 13)
RNN_OP(RNN)
RNN_OP(RNNV2)
RNN_OP(RNNV3)
RNN_OP(RNNV4)
RNN_OP(RNNV5)
RNN_OP(RNNV6)
RNN_OP(RNNV7)
RNN_OP(RNNV8)
RNN_OP(RNNV9)
RNN_OP(RNNV10)
RNN_OP(GRU)
RNN_OP(GRUV2)
RNN_OP(GRUV3)
RNN_OP(GRUV4)
RNN_OP(GRUV5)
RNN_OP(LSTM)
RNN_OP(LSTMV2)
RNN_OP(LSTMV3)
RNN_OP(LSTMV4)
RNN_OP(LSTMV5)
RNN_OP(LSTMBlockCell)
RNN_OP(LSTMBlockCellGrad)
RNN_OP(GRUBlockCell)
RNN_OP(GRUBlockCellGrad)
RNN_OP(BlockLSTM)
RNN_OP(BlockLSTMGrad)
RNN_OP(BlockGRU)
RNN_OP(BlockGRUGrad)
RNN_OP(CudnnRNN)
RNN_OP(CudnnRNNV2)
RNN_OP(CudnnRNNV3)
RNN_OP(CudnnRNNBackprop)
RNN_OP(CudnnRNNBackpropV2)
RNN_OP(CudnnRNNBackpropV3)

#undef RNN_OP
