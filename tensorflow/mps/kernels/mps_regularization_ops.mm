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

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// Macros for repetitive stub generation
#define REGULARIZER_OP(OpName) \
extern "C" void* MPS##OpName##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
extern "C" void MPS##OpName##_Delete(void* kernel) {} \
extern "C" void MPS##OpName##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
  TF_Status* status = TF_NewStatus(); \
  TF_SetStatus(status, TF_UNIMPLEMENTED, #OpName " not implemented for MPS"); \
  TF_OpKernelContext_Failure(ctx, status); \
  TF_DeleteStatus(status); \
}

// Regularization ops (Batch 20 - Final)
REGULARIZER_OP(L1Regularizer)
REGULARIZER_OP(L2Regularizer)
REGULARIZER_OP(L1L2Regularizer)
REGULARIZER_OP(ElasticNetRegularizer)
REGULARIZER_OP(Dropout)
REGULARIZER_OP(DropoutV2)
REGULARIZER_OP(SpatialDropout1D)
REGULARIZER_OP(SpatialDropout2D)
REGULARIZER_OP(SpatialDropout3D)
REGULARIZER_OP(AlphaDropout)
REGULARIZER_OP(GaussianDropout)
REGULARIZER_OP(GaussianNoise)
REGULARIZER_OP(ActivityRegularizer)
REGULARIZER_OP(OrthogonalRegularizer)
REGULARIZER_OP(MaxNormConstraint)
REGULARIZER_OP(MinMaxNormConstraint)
REGULARIZER_OP(NonNegConstraint)
REGULARIZER_OP(UnitNormConstraint)
REGULARIZER_OP(RadialConstraint)
REGULARIZER_OP(WeightDecay)
REGULARIZER_OP(GradientClipping)
REGULARIZER_OP(GradientClippingByValue)
REGULARIZER_OP(GradientClippingByNorm)
REGULARIZER_OP(GradientClippingByGlobalNorm)
REGULARIZER_OP(EarlyStopping)
REGULARIZER_OP(ModelCheckpoint)
REGULARIZER_OP(ReduceLROnPlateau)
REGULARIZER_OP(LearningRateScheduler)
REGULARIZER_OP(TensorBoard)
REGULARIZER_OP(CSVLogger)
REGULARIZER_OP(RemoteMonitor)
REGULARIZER_OP(LambdaCallback)
REGULARIZER_OP(TerminateOnNaN)
REGULARIZER_OP(ProgbarLogger)
REGULARIZER_OP(History)
REGULARIZER_OP(BaseLogger)

#undef REGULARIZER_OP
