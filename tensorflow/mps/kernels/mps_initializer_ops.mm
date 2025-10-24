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
#define INITIALIZER_OP(OpName) \
extern "C" void* MPS##OpName##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
extern "C" void MPS##OpName##_Delete(void* kernel) {} \
extern "C" void MPS##OpName##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
  TF_Status* status = TF_NewStatus(); \
  TF_SetStatus(status, TF_UNIMPLEMENTED, #OpName " not implemented for MPS"); \
  TF_OpKernelContext_Failure(ctx, status); \
  TF_DeleteStatus(status); \
}

// Initializer ops (Batch 19)
INITIALIZER_OP(Zeros)
INITIALIZER_OP(Ones)
INITIALIZER_OP(Constant)
INITIALIZER_OP(ConstantV2)
INITIALIZER_OP(Fill)
INITIALIZER_OP(RandomUniform)
INITIALIZER_OP(RandomNormal)
INITIALIZER_OP(TruncatedNormal)
INITIALIZER_OP(RandomGamma)
INITIALIZER_OP(RandomPoisson)
INITIALIZER_OP(RandomPoissonV2)
INITIALIZER_OP(Multinomial)
INITIALIZER_OP(RandomShuffle)
INITIALIZER_OP(ParameterizedTruncatedNormal)
INITIALIZER_OP(StatelessRandomUniform)
INITIALIZER_OP(StatelessRandomUniformV2)
INITIALIZER_OP(StatelessRandomUniformInt)
INITIALIZER_OP(StatelessRandomUniformIntV2)
INITIALIZER_OP(StatelessRandomNormal)
INITIALIZER_OP(StatelessRandomNormalV2)
INITIALIZER_OP(StatelessTruncatedNormal)
INITIALIZER_OP(StatelessTruncatedNormalV2)
INITIALIZER_OP(StatelessRandomGamma)
INITIALIZER_OP(StatelessRandomGammaV2)
INITIALIZER_OP(StatelessRandomGammaV3)
INITIALIZER_OP(StatelessMultinomial)
INITIALIZER_OP(StatelessRandomPoisson)
INITIALIZER_OP(GlorotUniform)
INITIALIZER_OP(GlorotNormal)
INITIALIZER_OP(HeUniform)
INITIALIZER_OP(HeNormal)
INITIALIZER_OP(LecunUniform)
INITIALIZER_OP(LecunNormal)
INITIALIZER_OP(VarianceScaling)
INITIALIZER_OP(Orthogonal)
INITIALIZER_OP(Identity)
INITIALIZER_OP(IdentityMatrix)
INITIALIZER_OP(Eye)
INITIALIZER_OP(Diag)
INITIALIZER_OP(DiagPart)
INITIALIZER_OP(TruncatedNormalInitializer)
INITIALIZER_OP(RandomNormalInitializer)
INITIALIZER_OP(RandomUniformInitializer)
INITIALIZER_OP(ConstantInitializer)
INITIALIZER_OP(ZerosInitializer)
INITIALIZER_OP(OnesInitializer)

#undef INITIALIZER_OP
