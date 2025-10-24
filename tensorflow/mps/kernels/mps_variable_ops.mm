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
#define VARIABLE_OP(OpName) \
extern "C" void* MPS##OpName##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
extern "C" void MPS##OpName##_Delete(void* kernel) {} \
extern "C" void MPS##OpName##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
  TF_Status* status = TF_NewStatus(); \
  TF_SetStatus(status, TF_UNIMPLEMENTED, #OpName " not implemented for MPS"); \
  TF_OpKernelContext_Failure(ctx, status); \
  TF_DeleteStatus(status); \
}

// Variable ops (Batch 18)
VARIABLE_OP(VarHandleOp)
VARIABLE_OP(ReadVariableOp)
VARIABLE_OP(AssignVariableOp)
VARIABLE_OP(AssignAddVariableOp)
VARIABLE_OP(AssignSubVariableOp)
VARIABLE_OP(ResourceGather)
VARIABLE_OP(ResourceGatherNd)
VARIABLE_OP(ResourceScatterAdd)
VARIABLE_OP(ResourceScatterSub)
VARIABLE_OP(ResourceScatterMul)
VARIABLE_OP(ResourceScatterDiv)
VARIABLE_OP(ResourceScatterMin)
VARIABLE_OP(ResourceScatterMax)
VARIABLE_OP(ResourceScatterUpdate)
VARIABLE_OP(ResourceScatterNdAdd)
VARIABLE_OP(ResourceScatterNdSub)
VARIABLE_OP(ResourceScatterNdUpdate)
VARIABLE_OP(VarIsInitializedOp)
VARIABLE_OP(VariableV2)
VARIABLE_OP(TemporaryVariable)
VARIABLE_OP(DestroyTemporaryVariable)
VARIABLE_OP(IsVariableInitialized)
VARIABLE_OP(Assign)
VARIABLE_OP(AssignAdd)
VARIABLE_OP(AssignSub)
VARIABLE_OP(ScatterAdd)
VARIABLE_OP(ScatterSub)
VARIABLE_OP(ScatterMul)
VARIABLE_OP(ScatterDiv)
VARIABLE_OP(ScatterMin)
VARIABLE_OP(ScatterMax)
VARIABLE_OP(ScatterUpdate)
VARIABLE_OP(ScatterNdAdd)
VARIABLE_OP(ScatterNdSub)
VARIABLE_OP(ScatterNdMul)
VARIABLE_OP(ScatterNdDiv)
VARIABLE_OP(ScatterNdMin)
VARIABLE_OP(ScatterNdMax)
VARIABLE_OP(ScatterNdUpdate)
VARIABLE_OP(TensorScatterAdd)
VARIABLE_OP(TensorScatterSub)
VARIABLE_OP(TensorScatterMul)
VARIABLE_OP(TensorScatterDiv)
VARIABLE_OP(TensorScatterMin)
VARIABLE_OP(TensorScatterMax)
VARIABLE_OP(TensorScatterUpdate)
VARIABLE_OP(CountUpTo)
VARIABLE_OP(DenseCountSparseOutput)
VARIABLE_OP(ResourceCountUpTo)

#undef VARIABLE_OP
