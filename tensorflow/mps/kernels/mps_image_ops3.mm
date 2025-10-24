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
#define IMAGE_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Image-related ops (Batch 14)
IMAGE_OP(DecodeJpeg)
IMAGE_OP(DecodeJpegV2)
IMAGE_OP(DecodeJpegV3)
IMAGE_OP(EncodeJpeg)
IMAGE_OP(EncodeJpegV2)
IMAGE_OP(EncodeJpegV3)
IMAGE_OP(DecodePng)
IMAGE_OP(DecodePngV2)
IMAGE_OP(DecodePngV3)
IMAGE_OP(EncodePng)
IMAGE_OP(EncodePngV2)
IMAGE_OP(EncodePngV3)
IMAGE_OP(DecodeGif)
IMAGE_OP(DecodeGifV2)
IMAGE_OP(DecodeGifV3)
IMAGE_OP(DecodeBmp)
IMAGE_OP(DecodeBmpV2)
IMAGE_OP(DecodeBmpV3)
IMAGE_OP(DecodeWebp)
IMAGE_OP(DecodeWebpV2)
IMAGE_OP(DecodeWebpV3)
IMAGE_OP(EncodeWebp)
IMAGE_OP(EncodeWebpV2)
IMAGE_OP(EncodeWebpV3)
IMAGE_OP(DecodeImage)
IMAGE_OP(DecodeImageV2)
IMAGE_OP(DecodeImageV3)
IMAGE_OP(ExtractJpegShape)
IMAGE_OP(ExtractJpegShapeV2)
IMAGE_OP(ExtractImagePatches)
IMAGE_OP(ExtractImagePatchesV2)
IMAGE_OP(AdjustContrast)
IMAGE_OP(AdjustContrastV2)

#undef IMAGE_OP
