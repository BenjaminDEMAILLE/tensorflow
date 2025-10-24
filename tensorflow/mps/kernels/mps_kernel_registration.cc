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

// RNN/LSTM/GRU
void* MPSLSTM_Create(TF_OpKernelConstruction* ctx);
void* MPSGRU_Create(TF_OpKernelConstruction* ctx);
void MPSRNN_Delete(void* kernel);
void MPSLSTM_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSGRU_Compute(void* kernel, TF_OpKernelContext* ctx);

// Attention
void* MPSScaledDotProductAttention_Create(TF_OpKernelConstruction* ctx);
void* MPSMultiHeadAttention_Create(TF_OpKernelConstruction* ctx);
void* MPSAdditiveAttention_Create(TF_OpKernelConstruction* ctx);
void* MPSSelfAttention_Create(TF_OpKernelConstruction* ctx);
void* MPSCrossAttention_Create(TF_OpKernelConstruction* ctx);
void MPSAttention_Delete(void* kernel);
void MPSScaledDotProductAttention_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMultiHeadAttention_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAdditiveAttention_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSelfAttention_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCrossAttention_Compute(void* kernel, TF_OpKernelContext* ctx);

// Image operations
void* MPSResizeBilinear_Create(TF_OpKernelConstruction* ctx);
void* MPSResizeNearestNeighbor_Create(TF_OpKernelConstruction* ctx);
void* MPSCropAndResize_Create(TF_OpKernelConstruction* ctx);
void* MPSImageGradients_Create(TF_OpKernelConstruction* ctx);
void* MPSRGBToGrayscale_Create(TF_OpKernelConstruction* ctx);
void* MPSHSVToRGB_Create(TF_OpKernelConstruction* ctx);
void* MPSAdjustBrightness_Create(TF_OpKernelConstruction* ctx);
void* MPSAdjustContrast_Create(TF_OpKernelConstruction* ctx);
void MPSImage_Delete(void* kernel);
void MPSResizeBilinear_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSResizeNearestNeighbor_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCropAndResize_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSImageGradients_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSRGBToGrayscale_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSHSVToRGB_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAdjustBrightness_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSAdjustContrast_Compute(void* kernel, TF_OpKernelContext* ctx);

// Sparse operations
void* MPSSparseToDense_Create(TF_OpKernelConstruction* ctx);
void* MPSSparseMatMul_Create(TF_OpKernelConstruction* ctx);
void* MPSSparseSoftmax_Create(TF_OpKernelConstruction* ctx);
void* MPSSparseAdd_Create(TF_OpKernelConstruction* ctx);
void* MPSSparseReorder_Create(TF_OpKernelConstruction* ctx);
void* MPSSparseSlice_Create(TF_OpKernelConstruction* ctx);
void MPSSparse_Delete(void* kernel);
void MPSSparseToDense_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSparseMatMul_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSparseSoftmax_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSparseAdd_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSparseReorder_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSSparseSlice_Compute(void* kernel, TF_OpKernelContext* ctx);

// Embedding operations
void* MPSEmbeddingLookup_Create(TF_OpKernelConstruction* ctx);
void* MPSGatherNd_Create(TF_OpKernelConstruction* ctx);
void* MPSScatterNd_Create(TF_OpKernelConstruction* ctx);
void* MPSGather_Create(TF_OpKernelConstruction* ctx);
void MPSEmbedding_Delete(void* kernel);
void MPSEmbeddingLookup_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSGatherNd_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSScatterNd_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSGather_Compute(void* kernel, TF_OpKernelContext* ctx);

// Conv3D and FFT
void* MPSConv3D_Create(TF_OpKernelConstruction* ctx);
void* MPSFFT_Create(TF_OpKernelConstruction* ctx);
void* MPSIFFT_Create(TF_OpKernelConstruction* ctx);
void* MPSRFFT_Create(TF_OpKernelConstruction* ctx);
void* MPSFFT2D_Create(TF_OpKernelConstruction* ctx);
void* MPSMaxPool3D_Create(TF_OpKernelConstruction* ctx);
void MPSConv3DFFT_Delete(void* kernel);
void MPSConv3D_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSFFT_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSIFFT_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSRFFT_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSFFT2D_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSMaxPool3D_Compute(void* kernel, TF_OpKernelContext* ctx);

// Control flow
void* MPSSelect_Create(TF_OpKernelConstruction* ctx);
void* MPSTopK_Create(TF_OpKernelConstruction* ctx);
void* MPSUnique_Create(TF_OpKernelConstruction* ctx);
void* MPSCumsum_Create(TF_OpKernelConstruction* ctx);
void* MPSRange_Create(TF_OpKernelConstruction* ctx);
void* MPSCast_Create(TF_OpKernelConstruction* ctx);
void MPSControlFlow_Delete(void* kernel);
void MPSSelect_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSTopK_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSUnique_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCumsum_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSRange_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSCast_Compute(void* kernel, TF_OpKernelContext* ctx);

// Random and Quantization
void* MPSRandomUniform_Create(TF_OpKernelConstruction* ctx);
void* MPSRandomNormal_Create(TF_OpKernelConstruction* ctx);
void* MPSDropout_Create(TF_OpKernelConstruction* ctx);
void* MPSQuantizeV2_Create(TF_OpKernelConstruction* ctx);
void* MPSDequantize_Create(TF_OpKernelConstruction* ctx);
void* MPSFakeQuant_Create(TF_OpKernelConstruction* ctx);
void* MPSClipByValue_Create(TF_OpKernelConstruction* ctx);
void MPSRandomQuant_Delete(void* kernel);
void MPSRandomUniform_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSRandomNormal_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSDropout_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSQuantizeV2_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSDequantize_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSFakeQuant_Compute(void* kernel, TF_OpKernelContext* ctx);
void MPSClipByValue_Compute(void* kernel, TF_OpKernelContext* ctx);
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

// RNN/LSTM/GRU
REGISTER_MPS_KERNEL(LSTM, MPSLSTM_Create, MPSRNN_Delete, MPSLSTM_Compute)
REGISTER_MPS_KERNEL(GRU, MPSGRU_Create, MPSRNN_Delete, MPSGRU_Compute)

// Attention mechanisms
REGISTER_MPS_KERNEL(ScaledDotProductAttention, MPSScaledDotProductAttention_Create, MPSAttention_Delete, MPSScaledDotProductAttention_Compute)
REGISTER_MPS_KERNEL(MultiHeadAttention, MPSMultiHeadAttention_Create, MPSAttention_Delete, MPSMultiHeadAttention_Compute)
REGISTER_MPS_KERNEL(AdditiveAttention, MPSAdditiveAttention_Create, MPSAttention_Delete, MPSAdditiveAttention_Compute)
REGISTER_MPS_KERNEL(SelfAttention, MPSSelfAttention_Create, MPSAttention_Delete, MPSSelfAttention_Compute)
REGISTER_MPS_KERNEL(CrossAttention, MPSCrossAttention_Create, MPSAttention_Delete, MPSCrossAttention_Compute)

// Image operations
REGISTER_MPS_KERNEL(ResizeBilinear, MPSResizeBilinear_Create, MPSImage_Delete, MPSResizeBilinear_Compute)
REGISTER_MPS_KERNEL(ResizeNearestNeighbor, MPSResizeNearestNeighbor_Create, MPSImage_Delete, MPSResizeNearestNeighbor_Compute)
REGISTER_MPS_KERNEL(CropAndResize, MPSCropAndResize_Create, MPSImage_Delete, MPSCropAndResize_Compute)
REGISTER_MPS_KERNEL(ImageGradients, MPSImageGradients_Create, MPSImage_Delete, MPSImageGradients_Compute)
REGISTER_MPS_KERNEL(RGBToGrayscale, MPSRGBToGrayscale_Create, MPSImage_Delete, MPSRGBToGrayscale_Compute)
REGISTER_MPS_KERNEL(HSVToRGB, MPSHSVToRGB_Create, MPSImage_Delete, MPSHSVToRGB_Compute)
REGISTER_MPS_KERNEL(AdjustBrightness, MPSAdjustBrightness_Create, MPSImage_Delete, MPSAdjustBrightness_Compute)
REGISTER_MPS_KERNEL(AdjustContrast, MPSAdjustContrast_Create, MPSImage_Delete, MPSAdjustContrast_Compute)

// Sparse operations
REGISTER_MPS_KERNEL(SparseToDense, MPSSparseToDense_Create, MPSSparse_Delete, MPSSparseToDense_Compute)
REGISTER_MPS_KERNEL(SparseMatMul, MPSSparseMatMul_Create, MPSSparse_Delete, MPSSparseMatMul_Compute)
REGISTER_MPS_KERNEL(SparseSoftmax, MPSSparseSoftmax_Create, MPSSparse_Delete, MPSSparseSoftmax_Compute)
REGISTER_MPS_KERNEL(SparseAdd, MPSSparseAdd_Create, MPSSparse_Delete, MPSSparseAdd_Compute)
REGISTER_MPS_KERNEL(SparseReorder, MPSSparseReorder_Create, MPSSparse_Delete, MPSSparseReorder_Compute)
REGISTER_MPS_KERNEL(SparseSlice, MPSSparseSlice_Create, MPSSparse_Delete, MPSSparseSlice_Compute)

// Embedding operations
REGISTER_MPS_KERNEL(EmbeddingLookup, MPSEmbeddingLookup_Create, MPSEmbedding_Delete, MPSEmbeddingLookup_Compute)
REGISTER_MPS_KERNEL(GatherNd, MPSGatherNd_Create, MPSEmbedding_Delete, MPSGatherNd_Compute)
REGISTER_MPS_KERNEL(ScatterNd, MPSScatterNd_Create, MPSEmbedding_Delete, MPSScatterNd_Compute)
REGISTER_MPS_KERNEL(Gather, MPSGather_Create, MPSEmbedding_Delete, MPSGather_Compute)

// Conv3D and FFT
REGISTER_MPS_KERNEL(Conv3D, MPSConv3D_Create, MPSConv3DFFT_Delete, MPSConv3D_Compute)
REGISTER_MPS_KERNEL(FFT, MPSFFT_Create, MPSConv3DFFT_Delete, MPSFFT_Compute)
REGISTER_MPS_KERNEL(IFFT, MPSIFFT_Create, MPSConv3DFFT_Delete, MPSIFFT_Compute)
REGISTER_MPS_KERNEL(RFFT, MPSRFFT_Create, MPSConv3DFFT_Delete, MPSRFFT_Compute)
REGISTER_MPS_KERNEL(FFT2D, MPSFFT2D_Create, MPSConv3DFFT_Delete, MPSFFT2D_Compute)
REGISTER_MPS_KERNEL(MaxPool3D, MPSMaxPool3D_Create, MPSConv3DFFT_Delete, MPSMaxPool3D_Compute)

// Control flow
REGISTER_MPS_KERNEL(Select, MPSSelect_Create, MPSControlFlow_Delete, MPSSelect_Compute)
REGISTER_MPS_KERNEL(TopK, MPSTopK_Create, MPSControlFlow_Delete, MPSTopK_Compute)
REGISTER_MPS_KERNEL(Unique, MPSUnique_Create, MPSControlFlow_Delete, MPSUnique_Compute)
REGISTER_MPS_KERNEL(Cumsum, MPSCumsum_Create, MPSControlFlow_Delete, MPSCumsum_Compute)
REGISTER_MPS_KERNEL(Range, MPSRange_Create, MPSControlFlow_Delete, MPSRange_Compute)
REGISTER_MPS_KERNEL(Cast, MPSCast_Create, MPSControlFlow_Delete, MPSCast_Compute)

// Random and Quantization
REGISTER_MPS_KERNEL(RandomUniform, MPSRandomUniform_Create, MPSRandomQuant_Delete, MPSRandomUniform_Compute)
REGISTER_MPS_KERNEL(RandomNormal, MPSRandomNormal_Create, MPSRandomQuant_Delete, MPSRandomNormal_Compute)
REGISTER_MPS_KERNEL(Dropout, MPSDropout_Create, MPSRandomQuant_Delete, MPSDropout_Compute)
REGISTER_MPS_KERNEL(QuantizeV2, MPSQuantizeV2_Create, MPSRandomQuant_Delete, MPSQuantizeV2_Compute)
REGISTER_MPS_KERNEL(Dequantize, MPSDequantize_Create, MPSRandomQuant_Delete, MPSDequantize_Compute)
REGISTER_MPS_KERNEL(FakeQuantWithMinMaxVars, MPSFakeQuant_Create, MPSRandomQuant_Delete, MPSFakeQuant_Compute)
REGISTER_MPS_KERNEL(ClipByValue, MPSClipByValue_Create, MPSRandomQuant_Delete, MPSClipByValue_Compute)

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
  
  // RNN/LSTM/GRU
  Register_LSTM(status);
  Register_GRU(status);
  
  // Attention mechanisms
  Register_ScaledDotProductAttention(status);
  Register_MultiHeadAttention(status);
  Register_AdditiveAttention(status);
  Register_SelfAttention(status);
  Register_CrossAttention(status);
  
  // Image operations
  Register_ResizeBilinear(status);
  Register_ResizeNearestNeighbor(status);
  Register_CropAndResize(status);
  Register_ImageGradients(status);
  Register_RGBToGrayscale(status);
  Register_HSVToRGB(status);
  Register_AdjustBrightness(status);
  Register_AdjustContrast(status);
  
  // Sparse operations
  Register_SparseToDense(status);
  Register_SparseMatMul(status);
  Register_SparseSoftmax(status);
  Register_SparseAdd(status);
  Register_SparseReorder(status);
  Register_SparseSlice(status);
  
  // Embedding operations
  Register_EmbeddingLookup(status);
  Register_GatherNd(status);
  Register_ScatterNd(status);
  Register_Gather(status);
  
  // Conv3D and FFT
  Register_Conv3D(status);
  Register_FFT(status);
  Register_IFFT(status);
  Register_RFFT(status);
  Register_FFT2D(status);
  Register_MaxPool3D(status);
  
  // Control flow
  Register_Select(status);
  Register_TopK(status);
  Register_Unique(status);
  Register_Cumsum(status);
  Register_Range(status);
  Register_Cast(status);
  
  // Random and Quantization
  Register_RandomUniform(status);
  Register_RandomNormal(status);
  Register_Dropout(status);
  Register_QuantizeV2(status);
  Register_Dequantize(status);
  Register_FakeQuantWithMinMaxVars(status);
  Register_ClipByValue(status);
  
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
