/* Copyright 2025 The TensorFlow Authors.

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

#ifndef TENSORFLOW_MPS_OPS_MPS_OPS_REGISTRY_H_
#define TENSORFLOW_MPS_OPS_MPS_OPS_REGISTRY_H_

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/kernels.h"

namespace tensorflow {
namespace mps {

// Forward declarations for kernel registration functions
// Each ops category module implements its own registration function

// Elementwise operations (41 ops)
void RegisterElementwiseOps(const char* platform_name, TF_Status* status);

// Activation operations (9 ops)
void RegisterActivationOps(const char* platform_name, TF_Status* status);

// Comparison operations (6 ops)
void RegisterComparisonOps(const char* platform_name, TF_Status* status);

// Logical operations (3 ops)
void RegisterLogicalOps(const char* platform_name, TF_Status* status);

// Reduction operations (7 ops)
void RegisterReductionOps(const char* platform_name, TF_Status* status);

// Tensor operations (15 ops)
void RegisterTensorOps(const char* platform_name, TF_Status* status);

// Indexing operations (5 ops)
void RegisterIndexingOps(const char* platform_name, TF_Status* status);

// Neural network operations (7 ops)
void RegisterNNOps(const char* platform_name, TF_Status* status);

// Utility operations (3 ops)
void RegisterUtilityOps(const char* platform_name, TF_Status* status);

// ---- Macros for simplified kernel registration ----

// Register a unary op for float, half, and bfloat16
#define REGISTER_MPS_UNARY_OP_3DTYPE(OP_NAME, PLATFORM_NAME, STATUS) \
  do { \
    extern void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*); \
    extern void MPS##OP_NAME##_Delete(void*); \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    \
    TF_KernelBuilder* kb_f = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  &MPS##OP_NAME##_Create, \
                                                  &MPS##OP_NAME##_Compute, \
                                                  &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_f, "T", TF_FLOAT, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", kb_f, STATUS); \
    \
    TF_KernelBuilder* kb_h = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  &MPS##OP_NAME##_Create, \
                                                  &MPS##OP_NAME##_Compute, \
                                                  &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_h, "T", TF_HALF, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", kb_h, STATUS); \
    \
    TF_KernelBuilder* kb_bf = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                   &MPS##OP_NAME##_Create, \
                                                   &MPS##OP_NAME##_Compute, \
                                                   &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_bf, "T", TF_BFLOAT16, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", kb_bf, STATUS); \
  } while (0)

// Register a binary op for float, half, and bfloat16
#define REGISTER_MPS_BINARY_OP_3DTYPE(OP_NAME, PLATFORM_NAME, STATUS) \
  do { \
    extern void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*); \
    extern void MPS##OP_NAME##_Delete(void*); \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    \
    TF_KernelBuilder* kb_f = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  &MPS##OP_NAME##_Create, \
                                                  &MPS##OP_NAME##_Compute, \
                                                  &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_f, "T", TF_FLOAT, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", kb_f, STATUS); \
    \
    TF_KernelBuilder* kb_h = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  &MPS##OP_NAME##_Create, \
                                                  &MPS##OP_NAME##_Compute, \
                                                  &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_h, "T", TF_HALF, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", kb_h, STATUS); \
    \
    TF_KernelBuilder* kb_bf = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                   &MPS##OP_NAME##_Create, \
                                                   &MPS##OP_NAME##_Compute, \
                                                   &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_bf, "T", TF_BFLOAT16, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", kb_bf, STATUS); \
  } while (0)

// Register a stateless op (no create/delete) for 3 dtypes
#define REGISTER_MPS_STATELESS_OP_3DTYPE(OP_NAME, PLATFORM_NAME, STATUS) \
  do { \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    extern void MPS##OP_NAME##Half_Compute(void*, TF_OpKernelContext*); \
    extern void MPS##OP_NAME##BFloat16_Compute(void*, TF_OpKernelContext*); \
    \
    TF_KernelBuilder* kb_f = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  nullptr, \
                                                  &MPS##OP_NAME##_Compute, \
                                                  nullptr); \
    TF_KernelBuilder_TypeConstraint(kb_f, "T", TF_FLOAT, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", kb_f, STATUS); \
    \
    TF_KernelBuilder* kb_h = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  nullptr, \
                                                  &MPS##OP_NAME##Half_Compute, \
                                                  nullptr); \
    TF_KernelBuilder_TypeConstraint(kb_h, "T", TF_HALF, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", kb_h, STATUS); \
    \
    TF_KernelBuilder* kb_bf = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                   nullptr, \
                                                   &MPS##OP_NAME##BFloat16_Compute, \
                                                   nullptr); \
    TF_KernelBuilder_TypeConstraint(kb_bf, "T", TF_BFLOAT16, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", kb_bf, STATUS); \
  } while (0)

// Register comparison op (returns bool, inputs T)
#define REGISTER_MPS_COMPARISON_OP_3DTYPE(OP_NAME, PLATFORM_NAME, STATUS) \
  do { \
    extern void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*); \
    extern void MPS##OP_NAME##_Delete(void*); \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    \
    TF_KernelBuilder* kb_f = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  &MPS##OP_NAME##_Create, \
                                                  &MPS##OP_NAME##_Compute, \
                                                  &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_f, "T", TF_FLOAT, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", kb_f, STATUS); \
    \
    TF_KernelBuilder* kb_h = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                  &MPS##OP_NAME##_Create, \
                                                  &MPS##OP_NAME##_Compute, \
                                                  &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_h, "T", TF_HALF, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", kb_h, STATUS); \
    \
    TF_KernelBuilder* kb_bf = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                   &MPS##OP_NAME##_Create, \
                                                   &MPS##OP_NAME##_Compute, \
                                                   &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_bf, "T", TF_BFLOAT16, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", kb_bf, STATUS); \
    \
    TF_KernelBuilder* kb_i32 = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                    &MPS##OP_NAME##_Create, \
                                                    &MPS##OP_NAME##_Compute, \
                                                    &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_i32, "T", TF_INT32, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Int32", kb_i32, STATUS); \
    \
    TF_KernelBuilder* kb_i64 = TF_NewKernelBuilder(#OP_NAME, PLATFORM_NAME, \
                                                    &MPS##OP_NAME##_Create, \
                                                    &MPS##OP_NAME##_Compute, \
                                                    &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(kb_i64, "T", TF_INT64, STATUS); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Int64", kb_i64, STATUS); \
  } while (0)

}  // namespace mps
}  // namespace tensorflow

#endif  // TENSORFLOW_MPS_OPS_MPS_OPS_REGISTRY_H_
