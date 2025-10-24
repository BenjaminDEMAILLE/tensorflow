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
#define IO_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// IO-related ops (Batch 12)
IO_OP(IORead)
IO_OP(IOWrite)
IO_OP(IOOpen)
IO_OP(IOClose)
IO_OP(IOFlush)
IO_OP(IOSeek)
IO_OP(IOTell)
IO_OP(IOStat)
IO_OP(IODelete)
IO_OP(IOMkdir)
IO_OP(IORmdir)
IO_OP(IOListdir)
IO_OP(IOCopy)
IO_OP(IOMove)
IO_OP(IOExists)
IO_OP(IOChmod)
IO_OP(IOChown)
IO_OP(IOUtime)
IO_OP(IOFsync)
IO_OP(IOFtruncate)
IO_OP(IOFstat)
IO_OP(IOFchmod)
IO_OP(IOFchown)
IO_OP(IOFutime)
IO_OP(IOFlock)
IO_OP(IOFunlock)
IO_OP(IOFadvise)
IO_OP(IOFallocate)
IO_OP(IOFdatasync)
IO_OP(IOFsyncdata)
IO_OP(IOFsyncfile)

#undef IO_OP
