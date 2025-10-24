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
#define SIGNAL_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// Signal-related ops (Batch 14)
SIGNAL_OP(FFT)
SIGNAL_OP(FFTV2)
SIGNAL_OP(FFTV3)
SIGNAL_OP(FFT2D)
SIGNAL_OP(FFT2DV2)
SIGNAL_OP(FFT2DV3)
SIGNAL_OP(FFT3D)
SIGNAL_OP(FFT3DV2)
SIGNAL_OP(FFT3DV3)
SIGNAL_OP(IFFT)
SIGNAL_OP(IFFTV2)
SIGNAL_OP(IFFTV3)
SIGNAL_OP(IFFT2D)
SIGNAL_OP(IFFT2DV2)
SIGNAL_OP(IFFT2DV3)
SIGNAL_OP(IFFT3D)
SIGNAL_OP(IFFT3DV2)
SIGNAL_OP(IFFT3DV3)
SIGNAL_OP(RFFT)
SIGNAL_OP(RFFTV2)
SIGNAL_OP(RFFTV3)
SIGNAL_OP(RFFT2D)
SIGNAL_OP(RFFT2DV2)
SIGNAL_OP(RFFT2DV3)
SIGNAL_OP(RFFT3D)
SIGNAL_OP(RFFT3DV2)
SIGNAL_OP(RFFT3DV3)
SIGNAL_OP(IRFFT)
SIGNAL_OP(IRFFTV2)
SIGNAL_OP(IRFFTV3)
SIGNAL_OP(IRFFT2D)
SIGNAL_OP(IRFFT2DV2)
SIGNAL_OP(IRFFT2DV3)

#undef SIGNAL_OP
