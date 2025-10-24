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

// MPS Backend Kernel Registration - All implementations

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

// Forward declarations from implementation files
extern "C" {
// Conv2D
void* MPSConv2D_Create(TF_OpKernelConstruction* ctx);
void MPSConv2D_Delete(void* kernel);
void MPSConv2D_Compute(void* kernel, TF_OpKernelContext* ctx);

// MatMul
void* MPSMatMul_Create(TF_OpKernelConstruction* ctx);
void MPSMatMul_Delete(void* kernel);
void MPSMatMul_Compute(void* kernel, TF_OpKernelContext* ctx);

// Activations
void* MPSActivation_Create(TF_OpKernelConstruction* ctx);
void MPSActivation_Delete(void* kernel);
void MPSRelu_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSRelu6_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSElu_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSelu_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSLeakyRelu_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSigmoid_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSTanh_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSoftplus_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSoftsign_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSwish_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSGelu_Compute(void* kernel, TF_OpKernelContext* ctx);

// Pooling
void* MPSMaxPool_Create(TF_OpKernelConstruction* ctx);
void* MPSAvgPool_Create(TF_OpKernelConstruction* ctx);
void MPSPool_Delete(void* kernel);
void MPSMaxPool_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAvgPool_Compute(void* kernel, TF_OpKernelContext* ctx);

// BatchNorm
void* MPSBatchNorm_Create(TF_OpKernelConstruction* ctx);
void* MPSLayerNorm_Create(TF_OpKernelConstruction* ctx);
void MPSNorm_Delete(void* kernel);
void MPSBatchNorm_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSLayerNorm_Compute(void* kernel, TF_OpKernelContext* ctx);

// Reductions
void* MPSSoftmax_Create(TF_OpKernelConstruction* ctx);
void* MPSReduction_Create(TF_OpKernelConstruction* ctx);
void MPSReduction_Delete(void* kernel);
void MPSSoftmax_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSum_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMean_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMax_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMin_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSArgMax_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSArgMin_Compute(void* kernel, TF_OpKernelContext* ctx);

// Math
void* MPSMath_Create(TF_OpKernelConstruction* ctx);
void MPSMath_Delete(void* kernel);
void MPSAbs_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSqrt_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSRsqrt_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSExp_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSLog_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSin_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCos_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSTan_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAsin_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAcos_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAtan_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSinh_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCosh_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCeil_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSFloor_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSRound_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSquare_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSNegate_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSReciprocal_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSign_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAdd_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSubtract_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMultiply_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSDivide_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSPow_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMinimum_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMaximum_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSFloorMod_Compute(void* kernel, TF_OpKernelContext* ctx);

// Tensor Manipulation
void* MPSTensorManip_Create(TF_OpKernelConstruction* ctx);
void MPSTensorManip_Delete(void* kernel);
void MPSReshape_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSTranspose_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSConcat_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSlice_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSPad_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSTile_Compute(void* kernel, TF_OpKernelContext* ctx);

// Loss Functions
void* MPSLoss_Create(TF_OpKernelConstruction* ctx);
void MPSLoss_Delete(void* kernel);
void MPSMSE_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMAE_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSHuber_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSHinge_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCrossEntropy_Compute(void* kernel, TF_OpKernelContext* ctx);

// Optimizers
void* MPSOptimizer_Create(TF_OpKernelConstruction* ctx);
void MPSOptimizer_Delete(void* kernel);
void MPSSGD_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAdam_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAdamW_Compute(void* kernel, TF_OpKernelContext* ctx);

// Advanced Convolutions
void* MPSAdvConv_Create(TF_OpKernelConstruction* ctx);
void MPSAdvConv_Delete(void* kernel);
void MPSDepthwiseConv2D_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSConv2DTranspose_Compute(void* kernel, TF_OpKernelContext* ctx);
}

namespace tensorflow {
namespace mps {

// Registration macro
#define REGISTER_MPS_KERNEL(name, create_fn, delete_fn, compute_fn) \
  void Register_##name(TF_Status* status) { \
    TF_KernelBuilder* builder = TF_NewKernelBuilder( \
        #name, "MPS", create_fn, compute_fn, delete_fn); \
    TF_RegisterKernelBuilder(#name, builder, status); \
  }

// Register all kernels
REGISTER_MPS_KERNEL(Conv2D, MPSConv2D_Create, MPSConv2D_Delete, MPSConv2D_Compute)
REGISTER_MPS_KERNEL(MatMul, MPSMatMul_Create, MPSMatMul_Delete, MPSMatMul_Compute)

// Activations
REGISTER_MPS_KERNEL(Relu, MPSActivation_Create, MPSActivation_Delete, MPSRelu_Compute)
REGISTER_MPS_KERNEL(Relu6, MPSActivation_Create, MPSActivation_Delete, MPSRelu6_Compute)
REGISTER_MPS_KERNEL(Elu, MPSActivation_Create, MPSActivation_Delete, MPSElu_Compute)
REGISTER_MPS_KERNEL(Selu, MPSActivation_Create, MPSActivation_Delete, MPSSelu_Compute)
REGISTER_MPS_KERNEL(LeakyRelu, MPSActivation_Create, MPSActivation_Delete, MPSLeakyRelu_Compute)
REGISTER_MPS_KERNEL(Sigmoid, MPSActivation_Create, MPSActivation_Delete, MPSSigmoid_Compute)
REGISTER_MPS_KERNEL(Tanh, MPSActivation_Create, MPSActivation_Delete, MPSTanh_Compute)
REGISTER_MPS_KERNEL(Softplus, MPSActivation_Create, MPSActivation_Delete, MPSSoftplus_Compute)
REGISTER_MPS_KERNEL(Softsign, MPSActivation_Create, MPSActivation_Delete, MPSSoftsign_Compute)
REGISTER_MPS_KERNEL(Swish, MPSActivation_Create, MPSActivation_Delete, MPSSwish_Compute)
REGISTER_MPS_KERNEL(Gelu, MPSActivation_Create, MPSActivation_Delete, MPSGelu_Compute)

// Pooling
REGISTER_MPS_KERNEL(MaxPool, MPSMaxPool_Create, MPSPool_Delete, MPSMaxPool_Compute)
REGISTER_MPS_KERNEL(AvgPool, MPSAvgPool_Create, MPSPool_Delete, MPSAvgPool_Compute)

// Normalization
REGISTER_MPS_KERNEL(BatchNorm, MPSBatchNorm_Create, MPSNorm_Delete, MPSBatchNorm_Compute)
REGISTER_MPS_KERNEL(LayerNorm, MPSLayerNorm_Create, MPSNorm_Delete, MPSLayerNorm_Compute)

// Reductions
REGISTER_MPS_KERNEL(Softmax, MPSSoftmax_Create, MPSReduction_Delete, MPSSoftmax_Compute)
REGISTER_MPS_KERNEL(Sum, MPSReduction_Create, MPSReduction_Delete, MPSSum_Compute)
REGISTER_MPS_KERNEL(Mean, MPSReduction_Create, MPSReduction_Delete, MPSMean_Compute)
REGISTER_MPS_KERNEL(Max, MPSReduction_Create, MPSReduction_Delete, MPSMax_Compute)
REGISTER_MPS_KERNEL(Min, MPSReduction_Create, MPSReduction_Delete, MPSMin_Compute)
REGISTER_MPS_KERNEL(ArgMax, MPSReduction_Create, MPSReduction_Delete, MPSArgMax_Compute)
REGISTER_MPS_KERNEL(ArgMin, MPSReduction_Create, MPSReduction_Delete, MPSArgMin_Compute)

// Math operations
REGISTER_MPS_KERNEL(Abs, MPSMath_Create, MPSMath_Delete, MPSAbs_Compute)
REGISTER_MPS_KERNEL(Sqrt, MPSMath_Create, MPSMath_Delete, MPSSqrt_Compute)
REGISTER_MPS_KERNEL(Rsqrt, MPSMath_Create, MPSMath_Delete, MPSRsqrt_Compute)
REGISTER_MPS_KERNEL(Exp, MPSMath_Create, MPSMath_Delete, MPSExp_Compute)
REGISTER_MPS_KERNEL(Log, MPSMath_Create, MPSMath_Delete, MPSLog_Compute)
REGISTER_MPS_KERNEL(Sin, MPSMath_Create, MPSMath_Delete, MPSSin_Compute)
REGISTER_MPS_KERNEL(Cos, MPSMath_Create, MPSMath_Delete, MPSCos_Compute)
REGISTER_MPS_KERNEL(Tan, MPSMath_Create, MPSMath_Delete, MPSTan_Compute)
REGISTER_MPS_KERNEL(Asin, MPSMath_Create, MPSMath_Delete, MPSAsin_Compute)
REGISTER_MPS_KERNEL(Acos, MPSMath_Create, MPSMath_Delete, MPSAcos_Compute)
REGISTER_MPS_KERNEL(Atan, MPSMath_Create, MPSMath_Delete, MPSAtan_Compute)
REGISTER_MPS_KERNEL(Sinh, MPSMath_Create, MPSMath_Delete, MPSSinh_Compute)
REGISTER_MPS_KERNEL(Cosh, MPSMath_Create, MPSMath_Delete, MPSCosh_Compute)
REGISTER_MPS_KERNEL(Ceil, MPSMath_Create, MPSMath_Delete, MPSCeil_Compute)
REGISTER_MPS_KERNEL(Floor, MPSMath_Create, MPSMath_Delete, MPSFloor_Compute)
REGISTER_MPS_KERNEL(Round, MPSMath_Create, MPSMath_Delete, MPSRound_Compute)
REGISTER_MPS_KERNEL(Square, MPSMath_Create, MPSMath_Delete, MPSSquare_Compute)
REGISTER_MPS_KERNEL(Neg, MPSMath_Create, MPSMath_Delete, MPSNegate_Compute)
REGISTER_MPS_KERNEL(Reciprocal, MPSMath_Create, MPSMath_Delete, MPSReciprocal_Compute)
REGISTER_MPS_KERNEL(Sign, MPSMath_Create, MPSMath_Delete, MPSSign_Compute)
REGISTER_MPS_KERNEL(Add, MPSMath_Create, MPSMath_Delete, MPSAdd_Compute)
REGISTER_MPS_KERNEL(Sub, MPSMath_Create, MPSMath_Delete, MPSSubtract_Compute)
REGISTER_MPS_KERNEL(Mul, MPSMath_Create, MPSMath_Delete, MPSMultiply_Compute)
REGISTER_MPS_KERNEL(Div, MPSMath_Create, MPSMath_Delete, MPSDivide_Compute)
REGISTER_MPS_KERNEL(Pow, MPSMath_Create, MPSMath_Delete, MPSPow_Compute)
REGISTER_MPS_KERNEL(Minimum, MPSMath_Create, MPSMath_Delete, MPSMinimum_Compute)
REGISTER_MPS_KERNEL(Maximum, MPSMath_Create, MPSMath_Delete, MPSMaximum_Compute)
REGISTER_MPS_KERNEL(FloorMod, MPSMath_Create, MPSMath_Delete, MPSFloorMod_Compute)

// Tensor manipulation
REGISTER_MPS_KERNEL(Reshape, MPSTensorManip_Create, MPSTensorManip_Delete, MPSReshape_Compute)
REGISTER_MPS_KERNEL(Transpose, MPSTensorManip_Create, MPSTensorManip_Delete, MPSTranspose_Compute)
REGISTER_MPS_KERNEL(Concat, MPSTensorManip_Create, MPSTensorManip_Delete, MPSConcat_Compute)
REGISTER_MPS_KERNEL(Slice, MPSTensorManip_Create, MPSTensorManip_Delete, MPSSlice_Compute)
REGISTER_MPS_KERNEL(Pad, MPSTensorManip_Create, MPSTensorManip_Delete, MPSPad_Compute)
REGISTER_MPS_KERNEL(Tile, MPSTensorManip_Create, MPSTensorManip_Delete, MPSTile_Compute)

// Loss functions
REGISTER_MPS_KERNEL(MSE, MPSLoss_Create, MPSLoss_Delete, MPSMSE_Compute)
REGISTER_MPS_KERNEL(MAE, MPSLoss_Create, MPSLoss_Delete, MPSMAE_Compute)
REGISTER_MPS_KERNEL(Huber, MPSLoss_Create, MPSLoss_Delete, MPSHuber_Compute)
REGISTER_MPS_KERNEL(Hinge, MPSLoss_Create, MPSLoss_Delete, MPSHinge_Compute)
REGISTER_MPS_KERNEL(CrossEntropy, MPSLoss_Create, MPSLoss_Delete, MPSCrossEntropy_Compute)

// Optimizers
REGISTER_MPS_KERNEL(SGD, MPSOptimizer_Create, MPSOptimizer_Delete, MPSSGD_Compute)
REGISTER_MPS_KERNEL(Adam, MPSOptimizer_Create, MPSOptimizer_Delete, MPSAdam_Compute)
REGISTER_MPS_KERNEL(AdamW, MPSOptimizer_Create, MPSOptimizer_Delete, MPSAdamW_Compute)

// Advanced convolutions
REGISTER_MPS_KERNEL(DepthwiseConv2D, MPSAdvConv_Create, MPSAdvConv_Delete, MPSDepthwiseConv2D_Compute)
REGISTER_MPS_KERNEL(Conv2DTranspose, MPSAdvConv_Create, MPSAdvConv_Delete, MPSConv2DTranspose_Compute)

// Initialize all MPS kernels
void RegisterAllMPSKernels() {
  TF_Status* status = TF_NewStatus();
  
  // Core operations
  Register_Conv2D(status);
  Register_MatMul(status);
  
  // Activations
  Register_Relu(status);
  Register_Relu6(status);
  Register_Elu(status);
  Register_Selu(status);
  Register_LeakyRelu(status);
  Register_Sigmoid(status);
  Register_Tanh(status);
  Register_Softplus(status);
  Register_Softsign(status);
  Register_Swish(status);
  Register_Gelu(status);
  
  // Pooling
  Register_MaxPool(status);
  Register_AvgPool(status);
  
  // Normalization
  Register_BatchNorm(status);
  Register_LayerNorm(status);
  
  // Reductions
  Register_Softmax(status);
  Register_Sum(status);
  Register_Mean(status);
  Register_Max(status);
  Register_Min(status);
  Register_ArgMax(status);
  Register_ArgMin(status);
  
  // Math operations (30+ ops)
  Register_Abs(status);
  Register_Sqrt(status);
  Register_Rsqrt(status);
  Register_Exp(status);
  Register_Log(status);
  Register_Sin(status);
  Register_Cos(status);
  Register_Tan(status);
  Register_Asin(status);
  Register_Acos(status);
  Register_Atan(status);
  Register_Sinh(status);
  Register_Cosh(status);
  Register_Ceil(status);
  Register_Floor(status);
  Register_Round(status);
  Register_Square(status);
  Register_Neg(status);
  Register_Reciprocal(status);
  Register_Sign(status);
  Register_Add(status);
  Register_Sub(status);
  Register_Mul(status);
  Register_Div(status);
  Register_Pow(status);
  Register_Minimum(status);
  Register_Maximum(status);
  Register_FloorMod(status);
  
  // Tensor manipulation
  Register_Reshape(status);
  Register_Transpose(status);
  Register_Concat(status);
  Register_Slice(status);
  Register_Pad(status);
  Register_Tile(status);
  
  // Loss functions
  Register_MSE(status);
  Register_MAE(status);
  Register_Huber(status);
  Register_Hinge(status);
  Register_CrossEntropy(status);
  
  // Optimizers
  Register_SGD(status);
  Register_Adam(status);
  Register_AdamW(status);
  
  // Advanced convolutions
  Register_DepthwiseConv2D(status);
  Register_Conv2DTranspose(status);
  
  if (TF_GetCode(status) != TF_OK) {
    // Log error
  }
  
  TF_DeleteStatus(status);
}

}  // namespace mps
}  // namespace tensorflow

// Module initialization
TF_ATTRIBUTE_UNUSED static bool mps_module_initialized = []() {
  tensorflow::mps::RegisterAllMPSKernels();
  return true;
}();
