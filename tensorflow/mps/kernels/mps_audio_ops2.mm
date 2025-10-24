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
#define AUDIO_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Audio-related ops (Batch 11)
AUDIO_OP(AudioSpectrogram)
AUDIO_OP(AudioDecodeWav)
AUDIO_OP(AudioEncodeWav)
AUDIO_OP(AudioDecodeMp3)
AUDIO_OP(AudioEncodeMp3)
AUDIO_OP(AudioDecodeFlac)
AUDIO_OP(AudioEncodeFlac)
AUDIO_OP(AudioDecodeOgg)
AUDIO_OP(AudioEncodeOgg)
AUDIO_OP(AudioDecodeOpus)
AUDIO_OP(AudioEncodeOpus)
AUDIO_OP(AudioDecodeAac)
AUDIO_OP(AudioEncodeAac)
AUDIO_OP(AudioDecodeAlac)
AUDIO_OP(AudioEncodeAlac)
AUDIO_OP(AudioDecodeAiff)
AUDIO_OP(AudioEncodeAiff)
AUDIO_OP(AudioDecodeWma)
AUDIO_OP(AudioEncodeWma)
AUDIO_OP(AudioDecodeAmr)
AUDIO_OP(AudioEncodeAmr)
AUDIO_OP(AudioDecodeAu)
AUDIO_OP(AudioEncodeAu)
AUDIO_OP(AudioDecodeM4a)
AUDIO_OP(AudioEncodeM4a)
AUDIO_OP(AudioDecodeMka)
AUDIO_OP(AudioEncodeMka)
AUDIO_OP(AudioDecodeMpc)
AUDIO_OP(AudioEncodeMpc)
AUDIO_OP(AudioDecodeRa)
AUDIO_OP(AudioEncodeRa)

#undef AUDIO_OP
