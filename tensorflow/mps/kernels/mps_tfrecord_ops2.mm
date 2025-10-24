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
#define TFRECORD_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// TFRecord-related ops (Batch 12)
TFRECORD_OP(TFRecordReader)
TFRECORD_OP(TFRecordWriter)
TFRECORD_OP(TFRecordOpen)
TFRECORD_OP(TFRecordClose)
TFRECORD_OP(TFRecordFlush)
TFRECORD_OP(TFRecordSeek)
TFRECORD_OP(TFRecordTell)
TFRECORD_OP(TFRecordStat)
TFRECORD_OP(TFRecordDelete)
TFRECORD_OP(TFRecordMkdir)
TFRECORD_OP(TFRecordRmdir)
TFRECORD_OP(TFRecordListdir)
TFRECORD_OP(TFRecordCopy)
TFRECORD_OP(TFRecordMove)
TFRECORD_OP(TFRecordExists)
TFRECORD_OP(TFRecordChmod)
TFRECORD_OP(TFRecordChown)
TFRECORD_OP(TFRecordUtime)
TFRECORD_OP(TFRecordFsync)
TFRECORD_OP(TFRecordFtruncate)
TFRECORD_OP(TFRecordFstat)
TFRECORD_OP(TFRecordFchmod)
TFRECORD_OP(TFRecordFchown)
TFRECORD_OP(TFRecordFutime)
TFRECORD_OP(TFRecordFlock)
TFRECORD_OP(TFRecordFunlock)
TFRECORD_OP(TFRecordFadvise)
TFRECORD_OP(TFRecordFallocate)
TFRECORD_OP(TFRecordFdatasync)
TFRECORD_OP(TFRecordFsyncdata)
TFRECORD_OP(TFRecordFsyncfile)

#undef TFRECORD_OP
