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

// CTC (Connectionist Temporal Classification) operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

#define CTC_OP(name) \
  extern "C" void* MPS##name##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
  extern "C" void MPS##name##_Delete(void* kernel) {} \
  extern "C" void MPS##name##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
    TF_Status* status = TF_NewStatus(); \
    TF_SetStatus(status, TF_UNIMPLEMENTED, #name " requires CTC implementation"); \
    TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status); \
  }

CTC_OP(CTCLoss)
CTC_OP(CTCLossV2)
CTC_OP(CTCGreedyDecoder)
CTC_OP(CTCBeamSearchDecoder)
CTC_OP(CTCBeamSearchDecoderV2)
