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
#define CONTROLFLOW_OP(OpName) \
class OpName##Op : public OpKernel { \
 public: \
  explicit OpName##Op(OpKernelConstruction* ctx) : OpKernel(ctx) {} \
  void Compute(OpKernelContext* ctx) override { \
    ctx->CtxFailure(errors::Unimplemented("MPS backend: " #OpName " is not implemented yet.")); \
  } \
};

// ControlFlow-related ops (Batch 11)
CONTROLFLOW_OP(ControlTrigger)
CONTROLFLOW_OP(ControlSwitch)
CONTROLFLOW_OP(ControlMerge)
CONTROLFLOW_OP(ControlEnter)
CONTROLFLOW_OP(ControlExit)
CONTROLFLOW_OP(ControlNextIteration)
CONTROLFLOW_OP(ControlLoopCond)
CONTROLFLOW_OP(ControlSwitchN)
CONTROLFLOW_OP(ControlRefSwitch)
CONTROLFLOW_OP(ControlRefMerge)
CONTROLFLOW_OP(ControlRefEnter)
CONTROLFLOW_OP(ControlRefExit)
CONTROLFLOW_OP(ControlRefNextIteration)
CONTROLFLOW_OP(ControlRefLoopCond)
CONTROLFLOW_OP(ControlRefSwitchN)
CONTROLFLOW_OP(ControlRefIdentity)
CONTROLFLOW_OP(ControlRefIdentityN)
CONTROLFLOW_OP(ControlRefNoOp)
CONTROLFLOW_OP(ControlRefStopGradient)
CONTROLFLOW_OP(ControlRefPreventGradient)
CONTROLFLOW_OP(ControlRefAssert)
CONTROLFLOW_OP(ControlRefCheckNumerics)
CONTROLFLOW_OP(ControlRefPrint)
CONTROLFLOW_OP(ControlRefPrintV2)
CONTROLFLOW_OP(ControlRefPrintV3)
CONTROLFLOW_OP(ControlRefPrintV4)
CONTROLFLOW_OP(ControlRefPrintV5)
CONTROLFLOW_OP(ControlRefPrintV6)
CONTROLFLOW_OP(ControlRefPrintV7)
CONTROLFLOW_OP(ControlRefPrintV8)
CONTROLFLOW_OP(ControlRefPrintV9)

#undef CONTROLFLOW_OP
