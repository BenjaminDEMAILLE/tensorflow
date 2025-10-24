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

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/mps/ops/mps_ops_registry.h"

extern "C" {

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status);

}  // extern "C"
extern "C" {

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  if (!params || !status) return;

  static SP_Platform platform = {SP_PLATFORM_STRUCT_SIZE};
  platform.ext = nullptr;
  platform.name = kPlatformName;
  platform.type = kDeviceType;
  platform.supports_unified_memory = 0;
  platform.use_bfc_allocator = 1;
  platform.force_memory_growth = 0;

  static SP_PlatformFns platform_fns = {SP_PLATFORM_FNS_STRUCT_SIZE};
  platform_fns.get_device_count = &GetDeviceCount;
  platform_fns.create_device = &CreateDevice;
  platform_fns.destroy_device = &DestroyDevice;
  platform_fns.create_device_fns = &CreateDeviceFns;
  platform_fns.destroy_device_fns = &DestroyDeviceFns;
  platform_fns.create_stream_executor = &CreateStreamExecutor;
  platform_fns.destroy_stream_executor = &DestroyStreamExecutor;
  platform_fns.create_timer_fns = &CreateTimerFns;
  platform_fns.destroy_timer_fns = &DestroyTimerFns;

  params->platform = &platform;
  params->platform_fns = &platform_fns;
  params->destroy_platform = &DestroyPlatform;
  params->destroy_platform_fns = &DestroyPlatformFns;

  TfOk(status);
}

// Register a minimal Identity(T=float) kernel for MPS device to exercise device path.
namespace {
void* MPSIdentity_Create(TF_OpKernelConstruction* /*ctx*/) { return nullptr; }

void MPSIdentity_Delete(void* /*kernel*/) {}

void MPSIdentity_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();

  // Get input[0]
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, /*i=*/0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  // Build dims from input
  int ndims = TF_NumDims(input);
  int64_t dims_buf[8];
  std::unique_ptr<int64_t[]> dyn_dims;
  int64_t* dims = dims_buf;
  if (ndims > 8) { dyn_dims.reset(new int64_t[ndims]); dims = dyn_dims.get(); }
  for (int i = 0; i < ndims; ++i) dims[i] = TF_Dim(input, i);

  // Try to forward input to output 0.
  int forwarded = -1;
  int cand = 0;
  TF_Tensor* out = TF_ForwardInputOrAllocateOutput(ctx, &cand, /*num_candidate_input_indices=*/1,
                                                   /*output_index=*/0, dims, ndims, &forwarded, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  if (forwarded < 0) {
    // No forward possible; copy host->host as a fallback (runtime will place buffers correctly for MPS).
    void* dst = TF_TensorData(out);
    const void* src = TF_TensorData(input);
    size_t nbytes = TF_TensorByteSize(out);
    if (src && dst && nbytes) memcpy(dst, src, nbytes);
  }

  TF_DeleteStatus(status);
}
}  // namespace

void TF_InitKernel() {
  TF_Status* status = TF_NewStatus();

  // Register Identity for device "MPS" with T=float/half/bfloat16.
  TF_KernelBuilder* kb = TF_NewKernelBuilder("Identity", kPlatformName,
                                             &MPSIdentity_Create,
                                             &MPSIdentity_Compute,
                                             &MPSIdentity_Delete);
  TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSIdentityFloat", kb, status);

  TF_KernelBuilder* kb_h = TF_NewKernelBuilder("Identity", kPlatformName,
                                              &MPSIdentity_Create,
                                              &MPSIdentity_Compute,
                                              &MPSIdentity_Delete);
  TF_KernelBuilder_TypeConstraint(kb_h, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSIdentityHalf", kb_h, status);

  TF_KernelBuilder* kb_bf = TF_NewKernelBuilder("Identity", kPlatformName,
                                               &MPSIdentity_Create,
                                               &MPSIdentity_Compute,
                                               &MPSIdentity_Delete);
  TF_KernelBuilder_TypeConstraint(kb_bf, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSIdentityBFloat16", kb_bf, status);
  // Register Relu for float/half/bfloat16.
  extern void MPSRelu_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* relu_kb = TF_NewKernelBuilder("Relu", kPlatformName,
                                                 /*create*/ nullptr,
                                                 /*compute*/ &MPSRelu_Compute,
                                                 /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(relu_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSReluFloat", relu_kb, status);

  extern void MPSReluHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* relu_h_kb = TF_NewKernelBuilder("Relu", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSReluHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(relu_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSReluHalf", relu_h_kb, status);

  extern void MPSReluBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* relu_bf_kb = TF_NewKernelBuilder("Relu", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSReluBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(relu_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSReluBFloat16", relu_bf_kb, status);

  // Register AddV2(T=float) and Mul(T=float) for device "MPS".
  extern void MPSAdd_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* add_kb = TF_NewKernelBuilder("AddV2", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSAdd_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(add_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSAddV2Float", add_kb, status);

  extern void MPSAddHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* add_h_kb = TF_NewKernelBuilder("AddV2", kPlatformName,
                                                  /*create*/ nullptr,
                                                  /*compute*/ &MPSAddHalf_Compute,
                                                  /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(add_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSAddV2Half", add_h_kb, status);

  extern void MPSMul_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mul_kb = TF_NewKernelBuilder("Mul", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMul_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(mul_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMulFloat", mul_kb, status);

  extern void MPSMulHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mul_h_kb = TF_NewKernelBuilder("Mul", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSMulHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(mul_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMulHalf", mul_h_kb, status);

  extern void MPSMulBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mul_bf_kb = TF_NewKernelBuilder("Mul", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSMulBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(mul_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMulBFloat16", mul_bf_kb, status);

  extern void MPSAddV2BFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* add_bf_kb = TF_NewKernelBuilder("AddV2", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSAddV2BFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(add_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSAddV2BFloat16", add_bf_kb, status);

  // Register MatMul(T=float) for device "MPS".
  extern void* MPSMatMul_Create(TF_OpKernelConstruction*);
  extern void MPSMatMul_Delete(void*);
  extern void MPSMatMul_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mm_kb = TF_NewKernelBuilder("MatMul", kPlatformName,
                                               &MPSMatMul_Create,
                                               &MPSMatMul_Compute,
                                               &MPSMatMul_Delete);
  TF_KernelBuilder_TypeConstraint(mm_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMatMulFloat", mm_kb, status);

  // MatMul half
  TF_KernelBuilder* mm_h_kb = TF_NewKernelBuilder("MatMul", kPlatformName,
                                                 &MPSMatMul_Create,
                                                 &MPSMatMul_Compute,
                                                 &MPSMatMul_Delete);
  TF_KernelBuilder_TypeConstraint(mm_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMatMulHalf", mm_h_kb, status);

  // MatMul bfloat16
  TF_KernelBuilder* mm_bf_kb = TF_NewKernelBuilder("MatMul", kPlatformName,
                                                  &MPSMatMul_Create,
                                                  &MPSMatMul_Compute,
                                                  &MPSMatMul_Delete);
  TF_KernelBuilder_TypeConstraint(mm_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMatMulBFloat16", mm_bf_kb, status);

  // Register Maximum/Minimum (float)
  extern void MPSMaximum_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* max_kb = TF_NewKernelBuilder("Maximum", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMaximum_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(max_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMaximumFloat", max_kb, status);

  extern void MPSMaximumHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* max_h_kb = TF_NewKernelBuilder("Maximum", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSMaximumHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(max_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMaximumHalf", max_h_kb, status);

  extern void MPSMaximumBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* max_bf_kb = TF_NewKernelBuilder("Maximum", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSMaximumBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(max_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMaximumBFloat16", max_bf_kb, status);

  extern void MPSMinimum_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* min_kb = TF_NewKernelBuilder("Minimum", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMinimum_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(min_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMinimumFloat", min_kb, status);

  extern void MPSMinimumHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* min_h_kb = TF_NewKernelBuilder("Minimum", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSMinimumHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(min_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMinimumHalf", min_h_kb, status);

  extern void MPSMinimumBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* min_bf_kb = TF_NewKernelBuilder("Minimum", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSMinimumBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(min_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMinimumBFloat16", min_bf_kb, status);

  // Register Sigmoid/Tanh (float)
  extern void MPSSigmoid_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sig_kb = TF_NewKernelBuilder("Sigmoid", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSSigmoid_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(sig_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSigmoidFloat", sig_kb, status);

  extern void MPSSigmoidHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sig_h_kb = TF_NewKernelBuilder("Sigmoid", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSSigmoidHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(sig_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSSigmoidHalf", sig_h_kb, status);

  extern void MPSSigmoidBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sig_bf_kb = TF_NewKernelBuilder("Sigmoid", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSSigmoidBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(sig_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSSigmoidBFloat16", sig_bf_kb, status);

  extern void MPSTanh_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* tanh_kb = TF_NewKernelBuilder("Tanh", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSTanh_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(tanh_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSTanhFloat", tanh_kb, status);

  extern void MPSTanhHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* tanh_h_kb = TF_NewKernelBuilder("Tanh", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSTanhHalf_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(tanh_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSTanhHalf", tanh_h_kb, status);

  extern void MPSTanhBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* tanh_bf_kb = TF_NewKernelBuilder("Tanh", kPlatformName,
                                                     /*create*/ nullptr,
                                                     /*compute*/ &MPSTanhBFloat16_Compute,
                                                     /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(tanh_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSTanhBFloat16", tanh_bf_kb, status);

  // Softmax (float, half, bfloat16) via MPSGraph
  extern void* MPSSoftmax_Create(TF_OpKernelConstruction*);
  extern void MPSSoftmax_Delete(void*);
  extern void MPSSoftmax_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* softmax_kb = TF_NewKernelBuilder("Softmax", kPlatformName,
                                                     &MPSSoftmax_Create,
                                                     &MPSSoftmax_Compute,
                                                     &MPSSoftmax_Delete);
  TF_KernelBuilder_TypeConstraint(softmax_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSoftmaxFloat", softmax_kb, status);

  TF_KernelBuilder* softmax_h_kb = TF_NewKernelBuilder("Softmax", kPlatformName,
                                                       &MPSSoftmax_Create,
                                                       &MPSSoftmax_Compute,
                                                       &MPSSoftmax_Delete);
  TF_KernelBuilder_TypeConstraint(softmax_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSSoftmaxHalf", softmax_h_kb, status);

  TF_KernelBuilder* softmax_bf_kb = TF_NewKernelBuilder("Softmax", kPlatformName,
                                                        &MPSSoftmax_Create,
                                                        &MPSSoftmax_Compute,
                                                        &MPSSoftmax_Delete);
  TF_KernelBuilder_TypeConstraint(softmax_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSSoftmaxBFloat16", softmax_bf_kb, status);

  // Register FusedBatchNormV3 (float, half, bfloat16)
  extern void* MPSFusedBatchNormV3_Create(TF_OpKernelConstruction*);
  extern void MPSFusedBatchNormV3_Delete(void*);
  extern void MPSFusedBatchNormV3_Compute(void*, TF_OpKernelContext*);

  TF_KernelBuilder* bn_kb = TF_NewKernelBuilder("FusedBatchNormV3", kPlatformName,
                                                &MPSFusedBatchNormV3_Create,
                                                &MPSFusedBatchNormV3_Compute,
                                                &MPSFusedBatchNormV3_Delete);
  TF_KernelBuilder_TypeConstraint(bn_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSFusedBatchNormV3Float", bn_kb, status);

  TF_KernelBuilder* bn_h_kb = TF_NewKernelBuilder("FusedBatchNormV3", kPlatformName,
                                                  &MPSFusedBatchNormV3_Create,
                                                  &MPSFusedBatchNormV3_Compute,
                                                  &MPSFusedBatchNormV3_Delete);
  TF_KernelBuilder_TypeConstraint(bn_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSFusedBatchNormV3Half", bn_h_kb, status);

  TF_KernelBuilder* bn_bf_kb = TF_NewKernelBuilder("FusedBatchNormV3", kPlatformName,
                                                   &MPSFusedBatchNormV3_Create,
                                                   &MPSFusedBatchNormV3_Compute,
                                                   &MPSFusedBatchNormV3_Delete);
  TF_KernelBuilder_TypeConstraint(bn_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSFusedBatchNormV3BFloat16", bn_bf_kb, status);

  // Register Swish activation (float, half, bfloat16)
  extern void* MPSSwish_Create(TF_OpKernelConstruction*);
  extern void MPSSwish_Delete(void*);
  extern void MPSSwish_Compute(void*, TF_OpKernelContext*);

  TF_KernelBuilder* swish_kb = TF_NewKernelBuilder("Swish", kPlatformName,
                                                   &MPSSwish_Create,
                                                   &MPSSwish_Compute,
                                                   &MPSSwish_Delete);
  TF_KernelBuilder_TypeConstraint(swish_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSwishFloat", swish_kb, status);

  TF_KernelBuilder* swish_h_kb = TF_NewKernelBuilder("Swish", kPlatformName,
                                                     &MPSSwish_Create,
                                                     &MPSSwish_Compute,
                                                     &MPSSwish_Delete);
  TF_KernelBuilder_TypeConstraint(swish_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSSwishHalf", swish_h_kb, status);

  TF_KernelBuilder* swish_bf_kb = TF_NewKernelBuilder("Swish", kPlatformName,
                                                      &MPSSwish_Create,
                                                      &MPSSwish_Compute,
                                                      &MPSSwish_Delete);
  TF_KernelBuilder_TypeConstraint(swish_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSSwishBFloat16", swish_bf_kb, status);

  // Register Gelu activation (float, half, bfloat16)
  extern void* MPSGelu_Create(TF_OpKernelConstruction*);
  extern void MPSGelu_Delete(void*);
  extern void MPSGelu_Compute(void*, TF_OpKernelContext*);

  TF_KernelBuilder* gelu_kb = TF_NewKernelBuilder("Gelu", kPlatformName,
                                                  &MPSGelu_Create,
                                                  &MPSGelu_Compute,
                                                  &MPSGelu_Delete);
  TF_KernelBuilder_TypeConstraint(gelu_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSGeluFloat", gelu_kb, status);

  TF_KernelBuilder* gelu_h_kb = TF_NewKernelBuilder("Gelu", kPlatformName,
                                                    &MPSGelu_Create,
                                                    &MPSGelu_Compute,
                                                    &MPSGelu_Delete);
  TF_KernelBuilder_TypeConstraint(gelu_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSGeluHalf", gelu_h_kb, status);

  TF_KernelBuilder* gelu_bf_kb = TF_NewKernelBuilder("Gelu", kPlatformName,
                                                     &MPSGelu_Create,
                                                     &MPSGelu_Compute,
                                                     &MPSGelu_Delete);
  TF_KernelBuilder_TypeConstraint(gelu_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSGeluBFloat16", gelu_bf_kb, status);

  // ===== MASS REGISTRATION OF NEW OPERATIONS =====
  // Macro to register unary ops for 3 dtypes
  #define REGISTER_UNARY_OP_3DTYPE(OP_NAME) \
    extern void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*); \
    extern void MPS##OP_NAME##_Delete(void*); \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    TF_KernelBuilder* op_name##_f_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(op_name##_f_kb, "T", TF_FLOAT, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", op_name##_f_kb, status); \
    TF_KernelBuilder* op_name##_h_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(op_name##_h_kb, "T", TF_HALF, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", op_name##_h_kb, status); \
    TF_KernelBuilder* op_name##_bf_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(op_name##_bf_kb, "T", TF_BFLOAT16, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", op_name##_bf_kb, status);

  // Register 27 unary ops (81 kernels total)
  REGISTER_UNARY_OP_3DTYPE(Abs)
  REGISTER_UNARY_OP_3DTYPE(Neg)
  REGISTER_UNARY_OP_3DTYPE(Sqrt)
  REGISTER_UNARY_OP_3DTYPE(Rsqrt)
  REGISTER_UNARY_OP_3DTYPE(Exp)
  REGISTER_UNARY_OP_3DTYPE(Log)
  REGISTER_UNARY_OP_3DTYPE(Sin)
  REGISTER_UNARY_OP_3DTYPE(Cos)
  REGISTER_UNARY_OP_3DTYPE(Tan)
  REGISTER_UNARY_OP_3DTYPE(Asin)
  REGISTER_UNARY_OP_3DTYPE(Acos)
  REGISTER_UNARY_OP_3DTYPE(Atan)
  REGISTER_UNARY_OP_3DTYPE(Sinh)
  REGISTER_UNARY_OP_3DTYPE(Cosh)
  REGISTER_UNARY_OP_3DTYPE(Asinh)
  REGISTER_UNARY_OP_3DTYPE(Acosh)
  REGISTER_UNARY_OP_3DTYPE(Atanh)
  REGISTER_UNARY_OP_3DTYPE(Ceil)
  REGISTER_UNARY_OP_3DTYPE(Floor)
  REGISTER_UNARY_OP_3DTYPE(Round)
  REGISTER_UNARY_OP_3DTYPE(Erf)
  REGISTER_UNARY_OP_3DTYPE(Square)
  REGISTER_UNARY_OP_3DTYPE(Reciprocal)
  REGISTER_UNARY_OP_3DTYPE(Sign)
  REGISTER_UNARY_OP_3DTYPE(Expm1)
  REGISTER_UNARY_OP_3DTYPE(Log1p)
  REGISTER_UNARY_OP_3DTYPE(IsFinite)

  // Additional activation registrations
  REGISTER_UNARY_OP_3DTYPE(LeakyRelu)
  REGISTER_UNARY_OP_3DTYPE(Relu6)
  REGISTER_UNARY_OP_3DTYPE(Elu)
  REGISTER_UNARY_OP_3DTYPE(Selu)
  REGISTER_UNARY_OP_3DTYPE(Softplus)
  REGISTER_UNARY_OP_3DTYPE(Softsign)

  // Macro for binary ops
  #define REGISTER_BINARY_OP_3DTYPE(OP_NAME) \
    extern void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*); \
    extern void MPS##OP_NAME##_Delete(void*); \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    TF_KernelBuilder* bin##OP_NAME##_f_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(bin##OP_NAME##_f_kb, "T", TF_FLOAT, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", bin##OP_NAME##_f_kb, status); \
    TF_KernelBuilder* bin##OP_NAME##_h_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(bin##OP_NAME##_h_kb, "T", TF_HALF, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", bin##OP_NAME##_h_kb, status); \
    TF_KernelBuilder* bin##OP_NAME##_bf_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(bin##OP_NAME##_bf_kb, "T", TF_BFLOAT16, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", bin##OP_NAME##_bf_kb, status);

  // Register 14 binary ops (42 kernels total)
  REGISTER_BINARY_OP_3DTYPE(Div)
  REGISTER_BINARY_OP_3DTYPE(RealDiv)
  REGISTER_BINARY_OP_3DTYPE(Sub)
  REGISTER_BINARY_OP_3DTYPE(Pow)
  REGISTER_BINARY_OP_3DTYPE(FloorDiv)
  REGISTER_BINARY_OP_3DTYPE(FloorMod)
  REGISTER_BINARY_OP_3DTYPE(Atan2)
  REGISTER_BINARY_OP_3DTYPE(SquaredDifference)
  REGISTER_BINARY_OP_3DTYPE(Equal)
  REGISTER_BINARY_OP_3DTYPE(NotEqual)
  REGISTER_BINARY_OP_3DTYPE(Less)
  REGISTER_BINARY_OP_3DTYPE(LessEqual)
  REGISTER_BINARY_OP_3DTYPE(Greater)
  REGISTER_BINARY_OP_3DTYPE(GreaterEqual)

  // Tensor op registrations (float/half/bfloat16)
  REGISTER_UNARY_OP_3DTYPE(Slice)
  REGISTER_UNARY_OP_3DTYPE(StridedSlice)
  REGISTER_UNARY_OP_3DTYPE(Fill)
  REGISTER_UNARY_OP_3DTYPE(ZerosLike)
  REGISTER_UNARY_OP_3DTYPE(OnesLike)
  REGISTER_UNARY_OP_3DTYPE(Pad)
  REGISTER_UNARY_OP_3DTYPE(MirrorPad)
  REGISTER_UNARY_OP_3DTYPE(Tile)
  REGISTER_UNARY_OP_3DTYPE(Select)
  REGISTER_UNARY_OP_3DTYPE(ClipByValue)

  // Split (equal splits): axis Tidx=int32/int64, T in float/half/bfloat16
  extern void* MPSplit_Create(TF_OpKernelConstruction*);
  extern void MPSplit_Delete(void*);
  extern void MPSplit_Compute(void*, TF_OpKernelContext*);
  // float
  TF_KernelBuilder* split_f_i32 = TF_NewKernelBuilder("Split", kPlatformName, &MPSplit_Create, &MPSplit_Compute, &MPSplit_Delete);
  TF_KernelBuilder_TypeConstraint(split_f_i32, "T", TF_FLOAT, status);
  TF_KernelBuilder_TypeConstraint(split_f_i32, "Tidx", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSSplitFloatInt32", split_f_i32, status);
  TF_KernelBuilder* split_f_i64 = TF_NewKernelBuilder("Split", kPlatformName, &MPSplit_Create, &MPSplit_Compute, &MPSplit_Delete);
  TF_KernelBuilder_TypeConstraint(split_f_i64, "T", TF_FLOAT, status);
  TF_KernelBuilder_TypeConstraint(split_f_i64, "Tidx", TF_INT64, status);
  TF_RegisterKernelBuilder("MPSSplitFloatInt64", split_f_i64, status);
  // half
  TF_KernelBuilder* split_h_i32 = TF_NewKernelBuilder("Split", kPlatformName, &MPSplit_Create, &MPSplit_Compute, &MPSplit_Delete);
  TF_KernelBuilder_TypeConstraint(split_h_i32, "T", TF_HALF, status);
  TF_KernelBuilder_TypeConstraint(split_h_i32, "Tidx", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSSplitHalfInt32", split_h_i32, status);
  TF_KernelBuilder* split_h_i64 = TF_NewKernelBuilder("Split", kPlatformName, &MPSplit_Create, &MPSplit_Compute, &MPSplit_Delete);
  TF_KernelBuilder_TypeConstraint(split_h_i64, "T", TF_HALF, status);
  TF_KernelBuilder_TypeConstraint(split_h_i64, "Tidx", TF_INT64, status);
  TF_RegisterKernelBuilder("MPSSplitHalfInt64", split_h_i64, status);
  // bfloat16
  TF_KernelBuilder* split_bf_i32 = TF_NewKernelBuilder("Split", kPlatformName, &MPSplit_Create, &MPSplit_Compute, &MPSplit_Delete);
  TF_KernelBuilder_TypeConstraint(split_bf_i32, "T", TF_BFLOAT16, status);
  TF_KernelBuilder_TypeConstraint(split_bf_i32, "Tidx", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSSplitBFloat16Int32", split_bf_i32, status);
  TF_KernelBuilder* split_bf_i64 = TF_NewKernelBuilder("Split", kPlatformName, &MPSplit_Create, &MPSplit_Compute, &MPSplit_Delete);
  TF_KernelBuilder_TypeConstraint(split_bf_i64, "T", TF_BFLOAT16, status);
  TF_KernelBuilder_TypeConstraint(split_bf_i64, "Tidx", TF_INT64, status);
  TF_RegisterKernelBuilder("MPSSplitBFloat16Int64", split_bf_i64, status);

  // OneHot (indices int32) for float/half/bfloat16
  extern void* MPSOneHot_Create(TF_OpKernelConstruction*);
  extern void MPSOneHot_Delete(void*);
  extern void MPSOneHot_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* oh_f = TF_NewKernelBuilder("OneHot", kPlatformName, &MPSOneHot_Create, &MPSOneHot_Compute, &MPSOneHot_Delete);
  TF_KernelBuilder_TypeConstraint(oh_f, "T", TF_FLOAT, status);
  TF_KernelBuilder_TypeConstraint(oh_f, "TI", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSOneHotFloat", oh_f, status);
  TF_KernelBuilder* oh_h = TF_NewKernelBuilder("OneHot", kPlatformName, &MPSOneHot_Create, &MPSOneHot_Compute, &MPSOneHot_Delete);
  TF_KernelBuilder_TypeConstraint(oh_h, "T", TF_HALF, status);
  TF_KernelBuilder_TypeConstraint(oh_h, "TI", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSOneHotHalf", oh_h, status);
  TF_KernelBuilder* oh_bf = TF_NewKernelBuilder("OneHot", kPlatformName, &MPSOneHot_Create, &MPSOneHot_Compute, &MPSOneHot_Delete);
  TF_KernelBuilder_TypeConstraint(oh_bf, "T", TF_BFLOAT16, status);
  TF_KernelBuilder_TypeConstraint(oh_bf, "TI", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSOneHotBFloat16", oh_bf, status);

  // Range
  extern void* MPSRange_Create(TF_OpKernelConstruction*);
  extern void MPSRange_Delete(void*);
  extern void MPSRange_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* range_f = TF_NewKernelBuilder("Range", kPlatformName, &MPSRange_Create, &MPSRange_Compute, &MPSRange_Delete);
  TF_KernelBuilder_TypeConstraint(range_f, "Tidx", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSRangeFloat", range_f, status);
  TF_KernelBuilder* range_i = TF_NewKernelBuilder("Range", kPlatformName, &MPSRange_Create, &MPSRange_Compute, &MPSRange_Delete);
  TF_KernelBuilder_TypeConstraint(range_i, "Tidx", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSRangeInt32", range_i, status);

  // GatherV2 registrations: T=float/half/bfloat16, Tindices=int32/int64 (axis provided as tensor input)
  extern void* MPSGatherV2_Create(TF_OpKernelConstruction*);
  extern void MPSGatherV2_Delete(void*);
  extern void MPSGatherV2_Compute(void*, TF_OpKernelContext*);
  auto reg_gather = [&](TF_DataType t, const char* tname) {
    TF_KernelBuilder* g_i32 = TF_NewKernelBuilder("GatherV2", kPlatformName, &MPSGatherV2_Create, &MPSGatherV2_Compute, &MPSGatherV2_Delete);
    TF_KernelBuilder_TypeConstraint(g_i32, "Tparams", t, status);
    TF_KernelBuilder_TypeConstraint(g_i32, "Tindices", TF_INT32, status);
    TF_RegisterKernelBuilder((std::string("MPSGatherV2") + tname + "Int32").c_str(), g_i32, status);
    TF_KernelBuilder* g_i64 = TF_NewKernelBuilder("GatherV2", kPlatformName, &MPSGatherV2_Create, &MPSGatherV2_Compute, &MPSGatherV2_Delete);
    TF_KernelBuilder_TypeConstraint(g_i64, "Tparams", t, status);
    TF_KernelBuilder_TypeConstraint(g_i64, "Tindices", TF_INT64, status);
    TF_RegisterKernelBuilder((std::string("MPSGatherV2") + tname + "Int64").c_str(), g_i64, status);
  };
  reg_gather(TF_FLOAT, "Float");
  reg_gather(TF_HALF, "Half");
  reg_gather(TF_BFLOAT16, "BFloat16");

  // GatherND registrations: T=float/half/bfloat16, Tindices=int32/int64
  extern void* MPSGatherND_Create(TF_OpKernelConstruction*);
  extern void MPSGatherND_Delete(void*);
  extern void MPSGatherND_Compute(void*, TF_OpKernelContext*);
  auto reg_gathernd = [&](TF_DataType t, const char* tname) {
    TF_KernelBuilder* gi32 = TF_NewKernelBuilder("GatherNd", kPlatformName, &MPSGatherND_Create, &MPSGatherND_Compute, &MPSGatherND_Delete);
    TF_KernelBuilder_TypeConstraint(gi32, "Tparams", t, status);
    TF_KernelBuilder_TypeConstraint(gi32, "Tindices", TF_INT32, status);
    TF_RegisterKernelBuilder((std::string("MPSGatherNd") + tname + "Int32").c_str(), gi32, status);
    TF_KernelBuilder* gi64 = TF_NewKernelBuilder("GatherNd", kPlatformName, &MPSGatherND_Create, &MPSGatherND_Compute, &MPSGatherND_Delete);
    TF_KernelBuilder_TypeConstraint(gi64, "Tparams", t, status);
    TF_KernelBuilder_TypeConstraint(gi64, "Tindices", TF_INT64, status);
    TF_RegisterKernelBuilder((std::string("MPSGatherNd") + tname + "Int64").c_str(), gi64, status);
  };
  reg_gathernd(TF_FLOAT, "Float");
  reg_gathernd(TF_HALF, "Half");
  reg_gathernd(TF_BFLOAT16, "BFloat16");

  // TensorScatterUpdate and TensorScatterAdd registrations
  auto reg_scatter = [&](const char* op, void* (*create)(TF_OpKernelConstruction*),
                          void (*compute)(void*, TF_OpKernelContext*), void (*del)(void*)) {
    auto reg_one = [&](TF_DataType t, const char* tname, TF_DataType tidx, const char* iname) {
      TF_KernelBuilder* kb = TF_NewKernelBuilder(op, kPlatformName, create, compute, del);
      TF_KernelBuilder_TypeConstraint(kb, "T", t, status);
      TF_KernelBuilder_TypeConstraint(kb, "Tindices", tidx, status);
      std::string name = std::string("MPS") + op + tname + iname;
      TF_RegisterKernelBuilder(name.c_str(), kb, status);
    };
    for (auto t : {TF_FLOAT, TF_HALF, TF_BFLOAT16}) {
      const char* tname = (t == TF_FLOAT) ? "Float" : ((t == TF_HALF) ? "Half" : "BFloat16");
      reg_one(t, tname, TF_INT32, "Int32");
      reg_one(t, tname, TF_INT64, "Int64");
    }
  };
  extern void* MPSTensorScatterUpdate_Create(TF_OpKernelConstruction*);
  extern void MPSTensorScatterUpdate_Delete(void*);
  extern void MPSTensorScatterUpdate_Compute(void*, TF_OpKernelContext*);
  reg_scatter("TensorScatterUpdate", &MPSTensorScatterUpdate_Create, &MPSTensorScatterUpdate_Compute, &MPSTensorScatterUpdate_Delete);
  
  extern void* MPSTensorScatterAdd_Create(TF_OpKernelConstruction*);
  extern void MPSTensorScatterAdd_Delete(void*);
  extern void MPSTensorScatterAdd_Compute(void*, TF_OpKernelContext*);
  reg_scatter("TensorScatterAdd", &MPSTensorScatterAdd_Create, &MPSTensorScatterAdd_Compute, &MPSTensorScatterAdd_Delete);

  // Logical ops (bool)
  extern void* MPSLogicalAnd_Create(TF_OpKernelConstruction*);
  extern void MPSLogicalAnd_Delete(void*);
  extern void MPSLogicalAnd_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* land_kb = TF_NewKernelBuilder("LogicalAnd", kPlatformName,
                                                  &MPSLogicalAnd_Create,
                                                  &MPSLogicalAnd_Compute,
                                                  &MPSLogicalAnd_Delete);
  TF_KernelBuilder_TypeConstraint(land_kb, "T", TF_BOOL, status);
  TF_RegisterKernelBuilder("MPSLogicalAnd", land_kb, status);

  extern void* MPSLogicalOr_Create(TF_OpKernelConstruction*);
  extern void MPSLogicalOr_Delete(void*);
  extern void MPSLogicalOr_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* lor_kb = TF_NewKernelBuilder("LogicalOr", kPlatformName,
                                                &MPSLogicalOr_Create,
                                                &MPSLogicalOr_Compute,
                                                &MPSLogicalOr_Delete);
  TF_KernelBuilder_TypeConstraint(lor_kb, "T", TF_BOOL, status);
  TF_RegisterKernelBuilder("MPSLogicalOr", lor_kb, status);

  extern void* MPSLogicalNot_Create(TF_OpKernelConstruction*);
  extern void MPSLogicalNot_Delete(void*);
  extern void MPSLogicalNot_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* lnot_kb = TF_NewKernelBuilder("LogicalNot", kPlatformName,
                                                 &MPSLogicalNot_Create,
                                                 &MPSLogicalNot_Compute,
                                                 &MPSLogicalNot_Delete);
  TF_KernelBuilder_TypeConstraint(lnot_kb, "T", TF_BOOL, status);
  TF_RegisterKernelBuilder("MPSLogicalNot", lnot_kb, status);

  // Comparison ops (float/half/bfloat16/int32/int64 inputs, bool output)
  auto register_comparison = [&](const char* op_name, void* (*create)(TF_OpKernelConstruction*),
                                   void (*compute)(void*, TF_OpKernelContext*), void (*del)(void*)) {
    std::vector<TF_DataType> dtypes = {TF_FLOAT, TF_HALF, TF_BFLOAT16, TF_INT32, TF_INT64};
    std::vector<const char*> dnames = {"Float", "Half", "BFloat16", "Int32", "Int64"};
    for (size_t i = 0; i < dtypes.size(); ++i) {
      TF_KernelBuilder* kb = TF_NewKernelBuilder(op_name, kPlatformName, create, compute, del);
      TF_KernelBuilder_TypeConstraint(kb, "T", dtypes[i], status);
      std::string name = std::string("MPS") + op_name + dnames[i];
      TF_RegisterKernelBuilder(name.c_str(), kb, status);
    }
  };
  extern void* MPSEqual_Create(TF_OpKernelConstruction*);
  extern void MPSEqual_Delete(void*);
  extern void MPSEqual_Compute(void*, TF_OpKernelContext*);
  register_comparison("Equal", &MPSEqual_Create, &MPSEqual_Compute, &MPSEqual_Delete);
  
  extern void* MPSNotEqual_Create(TF_OpKernelConstruction*);
  extern void MPSNotEqual_Delete(void*);
  extern void MPSNotEqual_Compute(void*, TF_OpKernelContext*);
  register_comparison("NotEqual", &MPSNotEqual_Create, &MPSNotEqual_Compute, &MPSNotEqual_Delete);
  
  extern void* MPSLess_Create(TF_OpKernelConstruction*);
  extern void MPSLess_Delete(void*);
  extern void MPSLess_Compute(void*, TF_OpKernelContext*);
  register_comparison("Less", &MPSLess_Create, &MPSLess_Compute, &MPSLess_Delete);
  
  extern void* MPSLessEqual_Create(TF_OpKernelConstruction*);
  extern void MPSLessEqual_Delete(void*);
  extern void MPSLessEqual_Compute(void*, TF_OpKernelContext*);
  register_comparison("LessEqual", &MPSLessEqual_Create, &MPSLessEqual_Compute, &MPSLessEqual_Delete);
  
  extern void* MPSGreater_Create(TF_OpKernelConstruction*);
  extern void MPSGreater_Delete(void*);
  extern void MPSGreater_Compute(void*, TF_OpKernelContext*);
  register_comparison("Greater", &MPSGreater_Create, &MPSGreater_Compute, &MPSGreater_Delete);
  
  extern void* MPSGreaterEqual_Create(TF_OpKernelConstruction*);
  extern void MPSGreaterEqual_Delete(void*);
  extern void MPSGreaterEqual_Compute(void*, TF_OpKernelContext*);
  register_comparison("GreaterEqual", &MPSGreaterEqual_Create, &MPSGreaterEqual_Compute, &MPSGreaterEqual_Delete);

  // Boolean reduction ops
  extern void* MPSAll_Create(TF_OpKernelConstruction*);
  extern void MPSAll_Delete(void*);
  extern void MPSAll_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* all_kb = TF_NewKernelBuilder("All", kPlatformName, &MPSAll_Create, &MPSAll_Compute, &MPSAll_Delete);
  TF_KernelBuilder_TypeConstraint(all_kb, "Tidx", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSAll", all_kb, status);
  
  extern void* MPSAny_Create(TF_OpKernelConstruction*);
  extern void MPSAny_Delete(void*);
  extern void MPSAny_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* any_kb = TF_NewKernelBuilder("Any", kPlatformName, &MPSAny_Create, &MPSAny_Compute, &MPSAny_Delete);
  TF_KernelBuilder_TypeConstraint(any_kb, "Tidx", TF_INT32, status);
  TF_RegisterKernelBuilder("MPSAny", any_kb, status);

  // Register Conv2D (T=float) for device "MPS" (NHWC only)
  extern void* MPSConv2D_Create(TF_OpKernelConstruction*);
  extern void MPSConv2D_Delete(void*);
  extern void MPSConv2D_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* conv_kb = TF_NewKernelBuilder("Conv2D", kPlatformName,
                                                  &MPSConv2D_Create,
                                                  &MPSConv2D_Compute,
                                                  &MPSConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(conv_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSConv2DFloat", conv_kb, status);

  // Conv2D half
  TF_KernelBuilder* conv_h_kb = TF_NewKernelBuilder("Conv2D", kPlatformName,
                                                    &MPSConv2D_Create,
                                                    &MPSConv2D_Compute,
                                                    &MPSConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(conv_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSConv2DHalf", conv_h_kb, status);

  // Conv2D bfloat16
  TF_KernelBuilder* conv_bf_kb = TF_NewKernelBuilder("Conv2D", kPlatformName,
                                                     &MPSConv2D_Create,
                                                     &MPSConv2D_Compute,
                                                     &MPSConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(conv_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSConv2DBFloat16", conv_bf_kb, status);

  // DepthwiseConv2dNative (float and half) for device "MPS" (NHWC only)
  extern void* MPSDepthwiseConv2D_Create(TF_OpKernelConstruction*);
  extern void MPSDepthwiseConv2D_Delete(void*);
  extern void MPSDepthwiseConv2D_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* dwconv_kb = TF_NewKernelBuilder("DepthwiseConv2dNative", kPlatformName,
                                                    &MPSDepthwiseConv2D_Create,
                                                    &MPSDepthwiseConv2D_Compute,
                                                    &MPSDepthwiseConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(dwconv_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSDepthwiseConv2DFloat", dwconv_kb, status);

  TF_KernelBuilder* dwconv_h_kb = TF_NewKernelBuilder("DepthwiseConv2dNative", kPlatformName,
                                                      &MPSDepthwiseConv2D_Create,
                                                      &MPSDepthwiseConv2D_Compute,
                                                      &MPSDepthwiseConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(dwconv_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSDepthwiseConv2DHalf", dwconv_h_kb, status);

  TF_KernelBuilder* dwconv_bf_kb = TF_NewKernelBuilder("DepthwiseConv2dNative", kPlatformName,
                                                       &MPSDepthwiseConv2D_Create,
                                                       &MPSDepthwiseConv2D_Compute,
                                                       &MPSDepthwiseConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(dwconv_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSDepthwiseConv2DBFloat16", dwconv_bf_kb, status);

  // MaxPool (float and half) for device "MPS" (NHWC only)
  extern void* MPSMaxPool_Create(TF_OpKernelConstruction*);
  extern void MPSMaxPool_Delete(void*);
  extern void MPSMaxPool_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* maxpool_kb = TF_NewKernelBuilder("MaxPool", kPlatformName,
                                                     &MPSMaxPool_Create,
                                                     &MPSMaxPool_Compute,
                                                     &MPSMaxPool_Delete);
  TF_KernelBuilder_TypeConstraint(maxpool_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMaxPoolFloat", maxpool_kb, status);

  TF_KernelBuilder* maxpool_h_kb = TF_NewKernelBuilder("MaxPool", kPlatformName,
                                                       &MPSMaxPool_Create,
                                                       &MPSMaxPool_Compute,
                                                       &MPSMaxPool_Delete);
  TF_KernelBuilder_TypeConstraint(maxpool_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMaxPoolHalf", maxpool_h_kb, status);

  TF_KernelBuilder* maxpool_bf_kb = TF_NewKernelBuilder("MaxPool", kPlatformName,
                                                        &MPSMaxPool_Create,
                                                        &MPSMaxPool_Compute,
                                                        &MPSMaxPool_Delete);
  TF_KernelBuilder_TypeConstraint(maxpool_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMaxPoolBFloat16", maxpool_bf_kb, status);

  // AvgPool (float and half) for device "MPS" (NHWC only)
  extern void* MPSAvgPool_Create(TF_OpKernelConstruction*);
  extern void MPSAvgPool_Delete(void*);
  extern void MPSAvgPool_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* avgpool_kb = TF_NewKernelBuilder("AvgPool", kPlatformName,
                                                     &MPSAvgPool_Create,
                                                     &MPSAvgPool_Compute,
                                                     &MPSAvgPool_Delete);
  TF_KernelBuilder_TypeConstraint(avgpool_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSAvgPoolFloat", avgpool_kb, status);

  TF_KernelBuilder* avgpool_h_kb = TF_NewKernelBuilder("AvgPool", kPlatformName,
                                                       &MPSAvgPool_Create,
                                                       &MPSAvgPool_Compute,
                                                       &MPSAvgPool_Delete);
  TF_KernelBuilder_TypeConstraint(avgpool_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSAvgPoolHalf", avgpool_h_kb, status);

  TF_KernelBuilder* avgpool_bf_kb = TF_NewKernelBuilder("AvgPool", kPlatformName,
                                                        &MPSAvgPool_Create,
                                                        &MPSAvgPool_Compute,
                                                        &MPSAvgPool_Delete);
  TF_KernelBuilder_TypeConstraint(avgpool_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSAvgPoolBFloat16", avgpool_bf_kb, status);

  // ===== NEW OPERATION CATEGORIES (Batch 1) =====
  
  // Padding operations (Pad, MirrorPad, SpaceToBatchND, BatchToSpaceND)
  extern void* MPSPad_Create(TF_OpKernelConstruction*);
  extern void MPSPad_Delete(void*);
  extern void MPSPad_Compute(void*, TF_OpKernelContext*);
  auto reg_pad = [&](TF_DataType t, const char* tname) {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Pad", kPlatformName, &MPSPad_Create, &MPSPad_Compute, &MPSPad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", t, status);
    TF_RegisterKernelBuilder((std::string("MPSPad") + tname).c_str(), kb, status);
  };
  reg_pad(TF_FLOAT, "Float");
  reg_pad(TF_HALF, "Half");
  reg_pad(TF_BFLOAT16, "BFloat16");

  extern void* MPSMirrorPad_Create(TF_OpKernelConstruction*);
  extern void MPSMirrorPad_Delete(void*);
  extern void MPSMirrorPad_Compute(void*, TF_OpKernelContext*);
  auto reg_mirrorpad = [&](TF_DataType t, const char* tname) {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MirrorPad", kPlatformName, &MPSMirrorPad_Create, &MPSMirrorPad_Compute, &MPSMirrorPad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", t, status);
    TF_RegisterKernelBuilder((std::string("MPSMirrorPad") + tname).c_str(), kb, status);
  };
  reg_mirrorpad(TF_FLOAT, "Float");
  reg_mirrorpad(TF_HALF, "Half");
  reg_mirrorpad(TF_BFLOAT16, "BFloat16");

  extern void* MPSSpaceToBatchND_Create(TF_OpKernelConstruction*);
  extern void MPSSpaceToBatchND_Delete(void*);
  extern void MPSSpaceToBatchND_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* s2b_kb = TF_NewKernelBuilder("SpaceToBatchND", kPlatformName, &MPSSpaceToBatchND_Create, &MPSSpaceToBatchND_Compute, &MPSSpaceToBatchND_Delete);
  TF_KernelBuilder_TypeConstraint(s2b_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSpaceToBatchND", s2b_kb, status);

  extern void* MPSBatchToSpaceND_Create(TF_OpKernelConstruction*);
  extern void MPSBatchToSpaceND_Delete(void*);
  extern void MPSBatchToSpaceND_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* b2s_kb = TF_NewKernelBuilder("BatchToSpaceND", kPlatformName, &MPSBatchToSpaceND_Create, &MPSBatchToSpaceND_Compute, &MPSBatchToSpaceND_Delete);
  TF_KernelBuilder_TypeConstraint(b2s_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSBatchToSpaceND", b2s_kb, status);

  // Array operations extended (Tile, Reverse, Unique, OneHot, TopKV2)
  extern void* MPSTile_Create(TF_OpKernelConstruction*);
  extern void MPSTile_Delete(void*);
  extern void MPSTile_Compute(void*, TF_OpKernelContext*);
  auto reg_tile = [&](TF_DataType t, const char* tname) {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Tile", kPlatformName, &MPSTile_Create, &MPSTile_Compute, &MPSTile_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", t, status);
    TF_RegisterKernelBuilder((std::string("MPSTile") + tname).c_str(), kb, status);
  };
  reg_tile(TF_FLOAT, "Float");
  reg_tile(TF_HALF, "Half");
  reg_tile(TF_BFLOAT16, "BFloat16");

  extern void* MPSReverse_Create(TF_OpKernelConstruction*);
  extern void MPSReverse_Delete(void*);
  extern void MPSReverse_Compute(void*, TF_OpKernelContext*);
  auto reg_reverse = [&](TF_DataType t, const char* tname) {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ReverseV2", kPlatformName, &MPSReverse_Create, &MPSReverse_Compute, &MPSReverse_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", t, status);
    TF_RegisterKernelBuilder((std::string("MPSReverseV2") + tname).c_str(), kb, status);
  };
  reg_reverse(TF_FLOAT, "Float");
  reg_reverse(TF_HALF, "Half");
  reg_reverse(TF_BFLOAT16, "BFloat16");

  extern void* MPSUnique_Create(TF_OpKernelConstruction*);
  extern void MPSUnique_Delete(void*);
  extern void MPSUnique_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* unique_kb = TF_NewKernelBuilder("Unique", kPlatformName, &MPSUnique_Create, &MPSUnique_Compute, &MPSUnique_Delete);
  TF_KernelBuilder_TypeConstraint(unique_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSUnique", unique_kb, status);

  // OneHot already registered above via macro

  extern void* MPSTopKV2_Create(TF_OpKernelConstruction*);
  extern void MPSTopKV2_Delete(void*);
  extern void MPSTopKV2_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* topk_kb = TF_NewKernelBuilder("TopKV2", kPlatformName, &MPSTopKV2_Create, &MPSTopKV2_Compute, &MPSTopKV2_Delete);
  TF_KernelBuilder_TypeConstraint(topk_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSTopKV2", topk_kb, status);

  // Optimizer operations (ApplyGradientDescent, ApplyMomentum, ApplyAdam, ApplyRMSprop, ApplyAdagrad)
  extern void* MPSApplyGradientDescent_Create(TF_OpKernelConstruction*);
  extern void MPSApplyGradientDescent_Delete(void*);
  extern void MPSApplyGradientDescent_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sgd_kb = TF_NewKernelBuilder("ApplyGradientDescent", kPlatformName, &MPSApplyGradientDescent_Create, &MPSApplyGradientDescent_Compute, &MPSApplyGradientDescent_Delete);
  TF_KernelBuilder_TypeConstraint(sgd_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSApplyGradientDescent", sgd_kb, status);

  extern void* MPSApplyMomentum_Create(TF_OpKernelConstruction*);
  extern void MPSApplyMomentum_Delete(void*);
  extern void MPSApplyMomentum_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mom_kb = TF_NewKernelBuilder("ApplyMomentum", kPlatformName, &MPSApplyMomentum_Create, &MPSApplyMomentum_Compute, &MPSApplyMomentum_Delete);
  TF_KernelBuilder_TypeConstraint(mom_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSApplyMomentum", mom_kb, status);

  extern void* MPSApplyAdam_Create(TF_OpKernelConstruction*);
  extern void MPSApplyAdam_Delete(void*);
  extern void MPSApplyAdam_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* adam_kb = TF_NewKernelBuilder("ApplyAdam", kPlatformName, &MPSApplyAdam_Create, &MPSApplyAdam_Compute, &MPSApplyAdam_Delete);
  TF_KernelBuilder_TypeConstraint(adam_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSApplyAdam", adam_kb, status);

  extern void* MPSApplyRMSprop_Create(TF_OpKernelConstruction*);
  extern void MPSApplyRMSprop_Delete(void*);
  extern void MPSApplyRMSprop_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* rms_kb = TF_NewKernelBuilder("ApplyRMSProp", kPlatformName, &MPSApplyRMSprop_Create, &MPSApplyRMSprop_Compute, &MPSApplyRMSprop_Delete);
  TF_KernelBuilder_TypeConstraint(rms_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSApplyRMSProp", rms_kb, status);

  extern void* MPSApplyAdagrad_Create(TF_OpKernelConstruction*);
  extern void MPSApplyAdagrad_Delete(void*);
  extern void MPSApplyAdagrad_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* adagrad_kb = TF_NewKernelBuilder("ApplyAdagrad", kPlatformName, &MPSApplyAdagrad_Create, &MPSApplyAdagrad_Compute, &MPSApplyAdagrad_Delete);
  TF_KernelBuilder_TypeConstraint(adagrad_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSApplyAdagrad", adagrad_kb, status);

  // Miscellaneous operations (Cast, Identity, CheckNumerics, ClipByValue, Fill) - some already registered
  extern void* MPSCast_Create(TF_OpKernelConstruction*);
  extern void MPSCast_Delete(void*);
  extern void MPSCast_Compute(void*, TF_OpKernelContext*);
  // Cast requires SrcT and DstT constraints - register common pairs
  auto reg_cast = [&](TF_DataType src, const char* sname, TF_DataType dst, const char* dname) {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Cast", kPlatformName, &MPSCast_Create, &MPSCast_Compute, &MPSCast_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "SrcT", src, status);
    TF_KernelBuilder_TypeConstraint(kb, "DstT", dst, status);
    std::string name = std::string("MPSCast") + sname + "To" + dname;
    TF_RegisterKernelBuilder(name.c_str(), kb, status);
  };
  reg_cast(TF_FLOAT, "Float", TF_HALF, "Half");
  reg_cast(TF_HALF, "Half", TF_FLOAT, "Float");
  reg_cast(TF_FLOAT, "Float", TF_BFLOAT16, "BFloat16");
  reg_cast(TF_BFLOAT16, "BFloat16", TF_FLOAT, "Float");
  reg_cast(TF_HALF, "Half", TF_BFLOAT16, "BFloat16");
  reg_cast(TF_BFLOAT16, "BFloat16", TF_HALF, "Half");

  // Identity, Fill, ClipByValue already registered via macros above

  extern void* MPSCheckNumerics_Create(TF_OpKernelConstruction*);
  extern void MPSCheckNumerics_Delete(void*);
  extern void MPSCheckNumerics_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* checknum_kb = TF_NewKernelBuilder("CheckNumerics", kPlatformName, &MPSCheckNumerics_Create, &MPSCheckNumerics_Compute, &MPSCheckNumerics_Delete);
  TF_KernelBuilder_TypeConstraint(checknum_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSCheckNumerics", checknum_kb, status);

  // Dataset operations (TensorSliceDataset, BatchDataset) - CPU fallback, no registration needed

  // If registration fails, status is dropped intentionally (plugin load should continue).
  TF_DeleteStatus(status);
}

}

// ===== MPS Relu kernel implementation (float/half) =====
namespace {
// Static pipeline cache for the Relu compute shader.
static id<MTLComputePipelineState> g_relu_pipeline = nil;
static id<MTLComputePipelineState> g_relu_h_pipeline = nil;
static id<MTLLibrary> g_relu_lib = nil;
static id<MTLLibrary> g_relu_h_lib = nil;
static dispatch_once_t g_relu_once;
static dispatch_once_t g_relu_h_once;

static void EnsureReluPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_relu_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void relu_k(const device float* in_ [[buffer(0)]],\n"
                       @"                 device float* out_ [[buffer(1)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out_[gid] = max(in_[gid], 0.0f);\n"
                       @"}";
    NSError* err = nil;
    g_relu_lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!g_relu_lib) {
      NSLog(@"MPS Relu: failed to compile MSL: %@", err);
      return;
    }
    id<MTLFunction> fn = [g_relu_lib newFunctionWithName:@"relu_k"];
    g_relu_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_relu_pipeline) {
      NSLog(@"MPS Relu: pipeline error: %@", err);
    }
  });
}

static void EnsureReluHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_relu_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void relu_h_k(const device half* in_ [[buffer(0)]],\n"
                       @"                   device half* out_ [[buffer(1)]],\n"
                       @"                   uint gid [[thread_position_in_grid]]) {\n"
                       @"  half zero = (half)0.0h;\n"
                       @"  out_[gid] = max(in_[gid], zero);\n"
                       @"}";
    NSError* err = nil;
    g_relu_h_lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!g_relu_h_lib) {
      NSLog(@"MPS Relu half: failed to compile MSL: %@", err);
      return;
    }
    id<MTLFunction> fn = [g_relu_h_lib newFunctionWithName:@"relu_h_k"];
    g_relu_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_relu_h_pipeline) {
      NSLog(@"MPS Relu half: pipeline error: %@", err);
    }
  });
}
}  // namespace

extern "C" void MPSRelu_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // Input 0
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Relu[MPS float] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims_stack[8];
  int64_t* dims = dims_stack;
  std::unique_ptr<int64_t[]> dyn;
  if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(input, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(float);

  // Output allocation
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Get MPS stream
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    // Host fallback
    const float* in = static_cast<const float*>(TF_TensorData(input));
    float* out = static_cast<float*>(TF_TensorData(output));
    for (int64_t i = 0; i < nelems; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    TF_DeleteStatus(s);
    return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  EnsureReluPipeline(dev);
  if (!g_relu_pipeline) {
    // Host fallback
    const float* in = static_cast<const float*>(TF_TensorData(input));
    float* out = static_cast<float*>(TF_TensorData(output));
    for (int64_t i = 0; i < nelems; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    TF_DeleteStatus(s);
    return;
  }

  // Stage, compute, stage back
  const void* in_host = TF_TensorData(input);
  void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);

  id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_relu_pipeline];
  [enc setBuffer:inb offset:0 atIndex:0];
  [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads = 256;
  NSUInteger grid = (NSUInteger)nelems;
  NSUInteger groups = (grid + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  memcpy(out_host, outb.contents, bytes);
  TF_DeleteStatus(s);
}

extern "C" void MPSReluHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_HALF) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Relu[MPS half] expects half");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd = TF_NumDims(input);
  int64_t nelems = 1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems * sizeof(uint16_t);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd, bytes, s);
  if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }

  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    // Host fallback with conversion
    const uint16_t* in = (const uint16_t*)TF_TensorData(input);
    uint16_t* out = (uint16_t*)TF_TensorData(output);
    for (int64_t i=0;i<nelems;++i){ float v = HalfToFloat(in[i]); if (v < 0) v = 0.0f; out[i] = FloatToHalf(v); }
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device; EnsureReluHalfPipeline(dev);
  if (!g_relu_h_pipeline) {
    const uint16_t* in = (const uint16_t*)TF_TensorData(input);
    uint16_t* out = (uint16_t*)TF_TensorData(output);
    for (int64_t i=0;i<nelems;++i){ float v = HalfToFloat(in[i]); if (v < 0) v = 0.0f; out[i] = FloatToHalf(v); }
    TF_DeleteStatus(s); return;
  }
  const void* in_host = TF_TensorData(input); void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_relu_h_pipeline];
  [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(out_host, outb.contents, bytes); TF_DeleteStatus(s);
}

extern "C" void MPSReluBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Relu[MPS bfloat16] expects bfloat16");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int nd = TF_NumDims(input); int64_t nelems=1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems*sizeof(uint16_t);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, bytes, s);
  if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  const uint16_t* in = (const uint16_t*)TF_TensorData(input);
  uint16_t* out = (uint16_t*)TF_TensorData(output);
  for (int64_t i=0;i<nelems;++i){ float v = BFloat16ToFloat(in[i]); if (v<0) v=0.0f; out[i]=FloatToBFloat16(v); }
  TF_DeleteStatus(s);
}

// ===== MPS Add and Mul kernels (float) =====
namespace {
static id<MTLComputePipelineState> g_add_pipeline = nil;
static id<MTLComputePipelineState> g_mul_pipeline = nil;
static dispatch_once_t g_add_once;
static dispatch_once_t g_mul_once;

static void EnsureAddPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_add_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void add_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] + b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Add: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"add_k"];
    g_add_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_add_pipeline) { NSLog(@"MPS Add: pipeline error: %@", err); }
  });
}

static void EnsureMulPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_mul_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void mul_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] * b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Mul: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_k"];
    g_mul_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_mul_pipeline) { NSLog(@"MPS Mul: pipeline error: %@", err); }
  });
}
}  // namespace

extern "C" void MPSAdd_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Same-shape fast path; allow scalar broadcasting on host fallback for now.
  int nd_a = TF_NumDims(a);
  int nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }

  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) {
      if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; }
    }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      // Host fallback
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + pb[i];
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureAddPipeline(dev);
    if (!g_add_pipeline) {
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + pb[i];
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a);
    const void* hb = TF_TensorData(b);
    void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes);
    memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_add_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256; NSUInteger grid = (NSUInteger)nelems;
    NSUInteger groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s);
    return;
  }

  // Scalar broadcasting on host
  bool a_scalar = (nelems_a == 1);
  bool b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2 shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  // Output dims = other tensor dims
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const float* pa = static_cast<const float*>(TF_TensorData(a));
  const float* pb = static_cast<const float*>(TF_TensorData(b));
  float* po = static_cast<float*>(TF_TensorData(out));
  if (a_scalar) {
    float av = pa[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = av + pb[i];
  } else {
    float bv = pb[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + bv;
  }
  TF_DeleteStatus(s); return;
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  EnsureAddPipeline(dev);
  if (!g_add_pipeline) {
    const float* pa = static_cast<const float*>(TF_TensorData(a));
    const float* pb = static_cast<const float*>(TF_TensorData(b));
    float* po = static_cast<float*>(TF_TensorData(out));
    for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + pb[i];
    TF_DeleteStatus(s); return;
  }
  const void* ha = TF_TensorData(a);
  const void* hb = TF_TensorData(b);
  void* ho = TF_TensorData(out);
  id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(ba.contents, ha, bytes);
  memcpy(bb.contents, hb, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_add_pipeline];
  [enc setBuffer:ba offset:0 atIndex:0];
  [enc setBuffer:bb offset:0 atIndex:1];
  [enc setBuffer:bo offset:0 atIndex:2];
  NSUInteger threads = 256; NSUInteger grid = (NSUInteger)nelems;
  NSUInteger groups = (grid + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(ho, bo.contents, bytes);
  TF_DeleteStatus(s);
}

extern "C" void MPSMul_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Same-shape fast path; allow scalar broadcasting on host fallback for now.
  int nd_a = TF_NumDims(a);
  int nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }

  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) {
      if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; }
    }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] * pb[i];
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureMulPipeline(dev);
    if (!g_mul_pipeline) {
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] * pb[i];
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a);
    const void* hb = TF_TensorData(b);
    void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes);
    memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_mul_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256; NSUInteger grid = (NSUInteger)nelems;
    NSUInteger groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s);
    return;
  }

  // Scalar broadcasting on host
  bool a_scalar = (nelems_a == 1);
  bool b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const float* pa = static_cast<const float*>(TF_TensorData(a));
  const float* pb = static_cast<const float*>(TF_TensorData(b));
  float* po = static_cast<float*>(TF_TensorData(out));
  if (a_scalar) {
    float av = pa[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = av * pb[i];
  } else {
    float bv = pb[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] * bv;
  }
  TF_DeleteStatus(s); return;
}

// ===== MPS Add/Mul kernels (half) =====
namespace {
static id<MTLComputePipelineState> g_add_h_pipeline = nil;
static id<MTLComputePipelineState> g_mul_h_pipeline = nil;
static dispatch_once_t g_add_h_once;
static dispatch_once_t g_mul_h_once;

static void EnsureAddHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_add_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void add_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2)]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] + b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS AddHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"add_h_k"];
    g_add_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_add_h_pipeline) { NSLog(@"MPS AddHalf: pipeline error: %@", err); }
  });
}

static void EnsureMulHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_mul_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void mul_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2)]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] * b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS MulHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_h_k"];
    g_mul_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_mul_h_pipeline) { NSLog(@"MPS MulHalf: pipeline error: %@", err); }
  });
}
}  // namespace

extern "C" void MPSAddHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_HALF || TF_TensorType(b) != TF_HALF) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2[MPS half] expects half");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) { if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; } }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(uint16_t);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) + HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureAddHalfPipeline(dev);
    if (!g_add_h_pipeline) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) + HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a), *hb = TF_TensorData(b); void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes); memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_add_h_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256, grid = (NSUInteger)nelems, groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s); return;
  }
  // Scalar broadcast on host
  bool a_scalar = (nelems_a == 1), b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2 shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(uint16_t);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  if (a_scalar) {
    float av = HalfToFloat(pa[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(av + HalfToFloat(pb[i]));
  } else {
    float bv = HalfToFloat(pb[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) + bv);
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMulHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_HALF || TF_TensorType(b) != TF_HALF) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul[MPS half] expects half");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) { if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; } }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(uint16_t);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) * HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureMulHalfPipeline(dev);
    if (!g_mul_h_pipeline) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) * HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a), *hb = TF_TensorData(b); void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes); memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_mul_h_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256, grid = (NSUInteger)nelems, groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s); return;
  }
  // Scalar broadcast on host
  bool a_scalar = (nelems_a == 1), b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(uint16_t);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  if (a_scalar) {
    float av = HalfToFloat(pa[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(av * HalfToFloat(pb[i]));
  } else {
    float bv = HalfToFloat(pb[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) * bv);
  }
  TF_DeleteStatus(s);
}

// ===== MPS Add/Mul/Maximum/Minimum/Sigmoid/Tanh kernels (bfloat16, host-based) =====
extern "C" void MPSAddV2BFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(fa + fb);
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMulBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(fa * fb);
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMaximumBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(std::max(fa, fb));
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMinimumBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(std::min(fa, fb));
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSSigmoidBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(input, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* in = (const uint16_t*)TF_TensorData(input);
  uint16_t* out = (uint16_t*)TF_TensorData(output);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float x = BFloat16ToFloat(in[i]);
    out[i] = FloatToBFloat16(1.0f / (1.0f + expf(-x)));
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSTanhBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(input, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* in = (const uint16_t*)TF_TensorData(input);
  uint16_t* out = (uint16_t*)TF_TensorData(output);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float x = BFloat16ToFloat(in[i]);
    out[i] = FloatToBFloat16(tanhf(x));
  }
  TF_DeleteStatus(s);
}

// ===== MPS MatMul kernel (float) =====
namespace {
struct MPSMatMulAttrs { bool ta; bool tb; };
}

extern "C" void* MPSMatMul_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Bool ta = 0, tb = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_a", &ta, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return nullptr; }
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_b", &tb, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return nullptr; }
  auto* attrs = new MPSMatMulAttrs{ta != 0, tb != 0};
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSMatMul_Delete(void* kernel) {
  auto* attrs = reinterpret_cast<MPSMatMulAttrs*>(kernel);
  delete attrs;
}

extern "C" void MPSMatMul_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = reinterpret_cast<MPSMatMulAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_DataType dtype_a = TF_TensorType(a), dtype_b = TF_TensorType(b);
  if (dtype_a != dtype_b) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul[MPS] inputs must have same dtype");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (dtype_a != TF_FLOAT && dtype_a != TF_HALF && dtype_a != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul[MPS] supports float/half/bfloat16");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_NumDims(a) != 2 || TF_NumDims(b) != 2) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul[MPS] requires rank-2 tensors");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int64_t a_rows = TF_Dim(a, 0), a_cols = TF_Dim(a, 1);
  int64_t b_rows = TF_Dim(b, 0), b_cols = TF_Dim(b, 1);
  int64_t M = attrs && attrs->ta ? a_cols : a_rows;
  int64_t K_a = attrs && attrs->ta ? a_rows : a_cols;
  int64_t K_b = attrs && attrs->tb ? b_cols : b_rows;
  int64_t N = attrs && attrs->tb ? b_rows : b_cols;
  if (K_a != K_b) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul inner dims mismatch");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int64_t K = K_a;

  bool is_bf16 = (dtype_a == TF_BFLOAT16);
  bool is_half = (dtype_a == TF_HALF);
  bool is_float = (dtype_a == TF_FLOAT);
  size_t elem_size = is_float ? sizeof(float) : sizeof(uint16_t);

  int64_t out_dims[2] = {M, N};
  size_t bytes = (size_t)M * (size_t)N * elem_size;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype_a, out_dims, 2, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Try GPU first; if stream is unavailable, fall back to host.
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    // Host matmul (works for all dtypes via conversion)
    if (is_float) {
      const float* A = static_cast<const float*>(TF_TensorData(a));
      const float* B = static_cast<const float*>(TF_TensorData(b));
      float* C = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            float va = attrs && attrs->ta ? A[k * a_cols + i] : A[i * a_cols + k];
            float vb = attrs && attrs->tb ? B[j * b_cols + k] : B[k * b_cols + j];
            sum += va * vb;
          }
          C[i * N + j] = sum;
        }
      }
    } else if (is_half) {
      const uint16_t* A = static_cast<const uint16_t*>(TF_TensorData(a));
      const uint16_t* B = static_cast<const uint16_t*>(TF_TensorData(b));
      uint16_t* C = static_cast<uint16_t*>(TF_TensorData(out));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            uint16_t ua = attrs && attrs->ta ? A[k * a_cols + i] : A[i * a_cols + k];
            uint16_t ub = attrs && attrs->tb ? B[j * b_cols + k] : B[k * b_cols + j];
            sum += HalfToFloat(ua) * HalfToFloat(ub);
          }
          C[i * N + j] = FloatToHalf(sum);
        }
      }
    } else {  // bfloat16
      const uint16_t* A = static_cast<const uint16_t*>(TF_TensorData(a));
      const uint16_t* B = static_cast<const uint16_t*>(TF_TensorData(b));
      uint16_t* C = static_cast<uint16_t*>(TF_TensorData(out));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            uint16_t ua = attrs && attrs->ta ? A[k * a_cols + i] : A[i * a_cols + k];
            uint16_t ub = attrs && attrs->tb ? B[j * b_cols + k] : B[k * b_cols + j];
            sum += BFloat16ToFloat(ua) * BFloat16ToFloat(ub);
          }
          C[i * N + j] = FloatToBFloat16(sum);
        }
      }
    }
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;

  // For bfloat16, use MPSGraph which supports MPSDataTypeBFloat16 natively
  if (is_bf16) {
    @autoreleasepool {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* tA = [graph placeholderWithShape:@[@(a_rows), @(a_cols)]
                                               dataType:MPSDataTypeBFloat16
                                                   name:@"A"];
      MPSGraphTensor* tB = [graph placeholderWithShape:@[@(b_rows), @(b_cols)]
                                               dataType:MPSDataTypeBFloat16
                                                   name:@"B"];
      if (attrs && attrs->ta) tA = [graph transposeTensor:tA dimension:0 withDimension:1 name:@"A_T"];
      if (attrs && attrs->tb) tB = [graph transposeTensor:tB dimension:0 withDimension:1 name:@"B_T"];
      MPSGraphTensor* tC = [graph matrixMultiplicationWithPrimaryTensor:tA
                                                        secondaryTensor:tB
                                                                   name:@"C"];
      size_t bytesA = (size_t)a_rows * (size_t)a_cols * sizeof(uint16_t);
      size_t bytesB = (size_t)b_rows * (size_t)b_cols * sizeof(uint16_t);
      id<MTLBuffer> bufA = [dev newBufferWithBytes:TF_TensorData(a) length:bytesA
                                           options:MTLResourceStorageModeShared];
      id<MTLBuffer> bufB = [dev newBufferWithBytes:TF_TensorData(b) length:bytesB
                                           options:MTLResourceStorageModeShared];
      id<MTLBuffer> bufC = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      MPSGraphTensorData* dA = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA
                                                                        shape:@[@(a_rows), @(a_cols)]
                                                                     dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData* dB = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB
                                                                        shape:@[@(b_rows), @(b_cols)]
                                                                     dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData* dC = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufC
                                                                        shape:@[@(M), @(N)]
                                                                     dataType:MPSDataTypeBFloat16];
      id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
      [graph runWithMTLCommandBuffer:cb
                                feeds:@{tA: dA, tB: dB}
                      targetTensors:@[tC]
                   targetOperations:nil
                   executionDescriptor:nil];
      [cb commit]; [cb waitUntilCompleted];
      memcpy(TF_TensorData(out), bufC.contents, bytes);
    }
    TF_DeleteStatus(s); return;
  }

  // For float and half, use MPSMatrixMultiplication
  // Stage to shared buffers (float or half)
  size_t bytesA = (size_t)a_rows * (size_t)a_cols * elem_size;
  size_t bytesB = (size_t)b_rows * (size_t)b_cols * elem_size;
  id<MTLBuffer> bufA = [dev newBufferWithLength:bytesA options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [dev newBufferWithLength:bytesB options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(bufA.contents, TF_TensorData(a), bytesA);
  memcpy(bufB.contents, TF_TensorData(b), bytesB);

  MPSDataType mps_dtype = is_float ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
  MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)a_rows
                                                                   columns:(NSUInteger)a_cols
                                                                  rowBytes:(NSUInteger)(a_cols * elem_size)
                                                                  dataType:mps_dtype];
  MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)b_rows
                                                                   columns:(NSUInteger)b_cols
                                                                  rowBytes:(NSUInteger)(b_cols * elem_size)
                                                                  dataType:mps_dtype];
  MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                                                   columns:(NSUInteger)N
                                                                  rowBytes:(NSUInteger)(N * elem_size)
                                                                  dataType:mps_dtype];
  MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA offset:0 descriptor:dA];
  MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufB offset:0 descriptor:dB];
  MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC offset:0 descriptor:dC];
  MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                 transposeLeft:(attrs && attrs->ta)
                                                                transposeRight:(attrs && attrs->tb)
                                                                   resultRows:(NSUInteger)M
                                                                resultColumns:(NSUInteger)N
                                                              interiorColumns:(NSUInteger)K
                                                                          alpha:1.0
                                                                           beta:0.0];
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
  [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(TF_TensorData(out), bufC.contents, bytes);
  TF_DeleteStatus(s);
}

// ===== StreamExecutor memset32 implementation (uint32_t pattern) =====
namespace {
static id<MTLComputePipelineState> g_memset32_pipeline = nil;
static dispatch_once_t g_memset32_once;
static void EnsureMemset32Pipeline(id<MTLDevice> dev) {
  dispatch_once(&g_memset32_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void memset32_k(device uint* out [[buffer(0)]],\n"
                       @"                      constant uint& pat [[buffer(1)]],\n"
                       @"                      uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = pat;\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS memset32: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"memset32_k"];
    g_memset32_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_memset32_pipeline) { NSLog(@"MPS memset32: pipeline error: %@", err); }
  });
}
}  // namespace

// (implementation moved earlier to avoid duplicate definitions)

// ===== MPS Maximum and Minimum kernels (float) =====
namespace {
static id<MTLComputePipelineState> g_max_pipeline = nil;
static id<MTLComputePipelineState> g_min_pipeline = nil;
static dispatch_once_t g_max_once;
static dispatch_once_t g_min_once;

static void EnsureMaxPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_max_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void max_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = fmax(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Maximum: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"max_k"];
    g_max_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_max_pipeline) { NSLog(@"MPS Maximum: pipeline error: %@", err); }
  });
}
static void EnsureMinPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_min_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void min_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = fmin(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Minimum: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"min_k"];
    g_min_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_min_pipeline) { NSLog(@"MPS Minimum: pipeline error: %@", err); }
  });
}
}

extern "C" void MPSMaximum_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Maximum[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  if (same_shape) for (int i = 0; i < nd_a; ++i) if (TF_Dim(a,i)!=TF_Dim(b,i)) { same_shape=false; break; }

  int nd = nd_a;
  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1; for (int i=0;i<nd;++i){ int64_t d=TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
      for (int64_t i=0;i<nelems;++i) po[i]= pa[i] > pb[i] ? pa[i] : pb[i];
      TF_DeleteStatus(s); return; }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureMaxPipeline(dev);
    if (!g_max_pipeline) { const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out); for (int64_t i=0;i<nelems;++i) po[i]= pa[i] > pb[i] ? pa[i] : pb[i]; TF_DeleteStatus(s); return; }
    const void* ha=TF_TensorData(a); const void* hb=TF_TensorData(b); void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes); memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_max_pipeline]; [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes); TF_DeleteStatus(s); return;
  }
  // Scalar broadcast
  bool a_scalar = (nelems_a==1), b_scalar=(nelems_b==1);
  if (!(a_scalar||b_scalar)) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Maximum shapes must match or one be scalar"); TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  int nd_out = a_scalar ? nd_b : nd_a; if (nd_out>8){ dyn.reset(new int64_t[nd_out]); dims=dyn.get(); }
  int64_t nelems=1; for (int i=0;i<nd_out;++i){ int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s); if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
  if (a_scalar){ float av=pa[0]; for (int64_t i=0;i<nelems;++i) po[i] = av > pb[i] ? av : pb[i]; }
  else { float bv=pb[0]; for (int64_t i=0;i<nelems;++i) po[i] = pa[i] > bv ? pa[i] : bv; }
  TF_DeleteStatus(s);
}

extern "C" void MPSMinimum_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Minimum[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  if (same_shape) for (int i = 0; i < nd_a; ++i) if (TF_Dim(a,i)!=TF_Dim(b,i)) { same_shape=false; break; }

  int nd = nd_a;
  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1; for (int i=0;i<nd;++i){ int64_t d=TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
      for (int64_t i=0;i<nelems;++i) po[i]= pa[i] < pb[i] ? pa[i] : pb[i];
      TF_DeleteStatus(s); return; }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureMinPipeline(dev);
    if (!g_min_pipeline) { const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out); for (int64_t i=0;i<nelems;++i) po[i]= pa[i] < pb[i] ? pa[i] : pb[i]; TF_DeleteStatus(s); return; }
    const void* ha=TF_TensorData(a); const void* hb=TF_TensorData(b); void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes); memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_min_pipeline]; [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes); TF_DeleteStatus(s); return;
  }
  // Scalar broadcast
  bool a_scalar = (nelems_a==1), b_scalar=(nelems_b==1);
  if (!(a_scalar||b_scalar)) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Minimum shapes must match or one be scalar"); TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  int nd_out = a_scalar ? nd_b : nd_a; if (nd_out>8){ dyn.reset(new int64_t[nd_out]); dims=dyn.get(); }
  int64_t nelems=1; for (int i=0;i<nd_out;++i){ int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s); if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
  if (a_scalar){ float av=pa[0]; for (int64_t i=0;i<nelems;++i) po[i] = av < pb[i] ? av : pb[i]; }
  else { float bv=pb[0]; for (int64_t i=0;i<nelems;++i) po[i] = pa[i] < bv ? pa[i] : bv; }
  TF_DeleteStatus(s);
}

// ===== MPS Sigmoid and Tanh kernels (float) =====
namespace {
static id<MTLComputePipelineState> g_sigmoid_pipeline = nil;
static id<MTLComputePipelineState> g_tanh_pipeline = nil;
static dispatch_once_t g_sigmoid_once;
static dispatch_once_t g_tanh_once;

static void EnsureSigmoidPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_sigmoid_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void sigmoid_k(const device float* in [[buffer(0)]],\n"
                       @"                     device float* out [[buffer(1)]],\n"
                       @"                     uint gid [[thread_position_in_grid]]) {\n"
                       @"  float x = in[gid];\n"
                       @"  out[gid] = 1.0f / (1.0f + exp(-x));\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Sigmoid: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"sigmoid_k"];
    g_sigmoid_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_sigmoid_pipeline) { NSLog(@"MPS Sigmoid: pipeline error: %@", err); }
  });
}
static void EnsureTanhPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_tanh_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void tanh_k(const device float* in [[buffer(0)]],\n"
                       @"                  device float* out [[buffer(1)]],\n"
                       @"                  uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = tanh(in[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Tanh: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"tanh_k"];
    g_tanh_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_tanh_pipeline) { NSLog(@"MPS Tanh: pipeline error: %@", err); }
  });
}
}

extern "C" void MPSSigmoid_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Sigmoid[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int nd = TF_NumDims(input); int64_t nelems = 1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems * sizeof(float);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) { float x = in[i]; out[i] = 1.0f / (1.0f + expf(-x)); }
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureSigmoidPipeline(dev);
  if (!g_sigmoid_pipeline) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) { float x = in[i]; out[i] = 1.0f / (1.0f + expf(-x)); }
    TF_DeleteStatus(s); return;
  }
  const void* in_host = TF_TensorData(input); void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_sigmoid_pipeline]; [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted]; memcpy(out_host, outb.contents, bytes); TF_DeleteStatus(s);
}

