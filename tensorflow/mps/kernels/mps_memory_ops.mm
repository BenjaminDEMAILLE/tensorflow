/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Device memory operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

#define MEM_OP(name) \
  extern "C" void* MPS##name##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
  extern "C" void MPS##name##_Delete(void* kernel) {} \
  extern "C" void MPS##name##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
    TF_Status* status = TF_NewStatus(); \
    TF_SetStatus(status, TF_UNIMPLEMENTED, #name " requires memory management"); \
    TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status); \
  }

MEM_OP(HostConst)
MEM_OP(DeviceIndex)
MEM_OP(GetSessionHandle)
MEM_OP(GetSessionHandleV2)
MEM_OP(GetSessionTensor)
MEM_OP(DeleteSessionTensor)
MEM_OP(CopyHost)
MEM_OP(CopyToHost)
MEM_OP(Stage)
MEM_OP(Unstage)
MEM_OP(StagePeek)
MEM_OP(StageSize)
MEM_OP(StageClear)
MEM_OP(OrderedMapStage)
MEM_OP(OrderedMapUnstage)
MEM_OP(OrderedMapPeek)
MEM_OP(OrderedMapSize)
MEM_OP(OrderedMapClear)
MEM_OP(OrderedMapIncompleteSize)
MEM_OP(MapStage)
MEM_OP(MapUnstage)
MEM_OP(MapPeek)
MEM_OP(MapSize)
MEM_OP(MapClear)
MEM_OP(MapIncompleteSize)
MEM_OP(TensorArray)
MEM_OP(TensorArrayV2)
MEM_OP(TensorArrayV3)
MEM_OP(TensorArrayGrad)
MEM_OP(TensorArrayGradV2)
MEM_OP(TensorArrayGradV3)
MEM_OP(TensorArrayGradWithShape)
MEM_OP(TensorArrayWrite)
MEM_OP(TensorArrayWriteV2)
MEM_OP(TensorArrayWriteV3)
MEM_OP(TensorArrayRead)
MEM_OP(TensorArrayReadV2)
MEM_OP(TensorArrayReadV3)
MEM_OP(TensorArrayGather)
MEM_OP(TensorArrayGatherV2)
MEM_OP(TensorArrayGatherV3)
MEM_OP(TensorArrayScatter)
MEM_OP(TensorArrayScatterV2)
MEM_OP(TensorArrayScatterV3)
MEM_OP(TensorArrayConcat)
MEM_OP(TensorArrayConcatV2)
MEM_OP(TensorArrayConcatV3)
MEM_OP(TensorArraySplit)
MEM_OP(TensorArraySplitV2)
MEM_OP(TensorArraySplitV3)
MEM_OP(TensorArraySize)
MEM_OP(TensorArraySizeV2)
MEM_OP(TensorArraySizeV3)
MEM_OP(TensorArrayClose)
MEM_OP(TensorArrayCloseV2)
MEM_OP(TensorArrayCloseV3)
