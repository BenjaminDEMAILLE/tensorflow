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
#define TRAINING_OP(OpName) \
extern "C" void* MPS##OpName##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
extern "C" void MPS##OpName##_Delete(void* kernel) {} \
extern "C" void MPS##OpName##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
  TF_Status* status = TF_NewStatus(); \
  TF_SetStatus(status, TF_UNIMPLEMENTED, #OpName " not implemented for MPS"); \
  TF_OpKernelContext_Failure(ctx, status); \
  TF_DeleteStatus(status); \
}

// Training ops (Batch 18)
TRAINING_OP(ApplyGradientDescent)
TRAINING_OP(ApplyProximalGradientDescent)
TRAINING_OP(ApplyAdadelta)
TRAINING_OP(ApplyAdagrad)
TRAINING_OP(ApplyAdagradDA)
TRAINING_OP(ApplyAdagradV2)
TRAINING_OP(ApplyAdam)
TRAINING_OP(ApplyAdaMax)
TRAINING_OP(ApplyAdamWithAmsgrad)
TRAINING_OP(ApplyAddSign)
TRAINING_OP(ApplyCenteredRMSProp)
TRAINING_OP(ApplyFtrl)
TRAINING_OP(ApplyFtrlV2)
TRAINING_OP(ApplyMomentum)
TRAINING_OP(ApplyKerasMomentum)
TRAINING_OP(ApplyPowerSign)
TRAINING_OP(ApplyProximalAdagrad)
TRAINING_OP(ApplyRMSProp)
TRAINING_OP(ResourceApplyGradientDescent)
TRAINING_OP(ResourceApplyProximalGradientDescent)
TRAINING_OP(ResourceApplyAdadelta)
TRAINING_OP(ResourceApplyAdagrad)
TRAINING_OP(ResourceApplyAdagradDA)
TRAINING_OP(ResourceApplyAdagradV2)
TRAINING_OP(ResourceApplyAdam)
TRAINING_OP(ResourceApplyAdaMax)
TRAINING_OP(ResourceApplyAdamWithAmsgrad)
TRAINING_OP(ResourceApplyAddSign)
TRAINING_OP(ResourceApplyCenteredRMSProp)
TRAINING_OP(ResourceApplyFtrl)
TRAINING_OP(ResourceApplyFtrlV2)
TRAINING_OP(ResourceApplyMomentum)
TRAINING_OP(ResourceApplyKerasMomentum)
TRAINING_OP(ResourceApplyPowerSign)
TRAINING_OP(ResourceApplyProximalAdagrad)
TRAINING_OP(ResourceApplyRMSProp)
TRAINING_OP(SparseApplyAdadelta)
TRAINING_OP(SparseApplyAdagrad)
TRAINING_OP(SparseApplyAdagradDA)
TRAINING_OP(SparseApplyAdagradV2)
TRAINING_OP(SparseApplyCenteredRMSProp)
TRAINING_OP(SparseApplyFtrl)
TRAINING_OP(SparseApplyFtrlV2)
TRAINING_OP(SparseApplyMomentum)
TRAINING_OP(SparseApplyProximalAdagrad)
TRAINING_OP(SparseApplyProximalGradientDescent)
TRAINING_OP(SparseApplyRMSProp)
TRAINING_OP(ResourceSparseApplyAdadelta)
TRAINING_OP(ResourceSparseApplyAdagrad)
TRAINING_OP(ResourceSparseApplyAdagradDA)
TRAINING_OP(ResourceSparseApplyAdagradV2)
TRAINING_OP(ResourceSparseApplyCenteredRMSProp)
TRAINING_OP(ResourceSparseApplyFtrl)
TRAINING_OP(ResourceSparseApplyFtrlV2)
TRAINING_OP(ResourceSparseApplyKerasMomentum)
TRAINING_OP(ResourceSparseApplyMomentum)
TRAINING_OP(ResourceSparseApplyProximalAdagrad)
TRAINING_OP(ResourceSparseApplyProximalGradientDescent)
TRAINING_OP(ResourceSparseApplyRMSProp)

#undef TRAINING_OP
