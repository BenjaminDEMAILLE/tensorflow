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
#define METRIC_OP(OpName) \
extern "C" void* MPS##OpName##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
extern "C" void MPS##OpName##_Delete(void* kernel) {} \
extern "C" void MPS##OpName##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
  TF_Status* status = TF_NewStatus(); \
  TF_SetStatus(status, TF_UNIMPLEMENTED, #OpName " not implemented for MPS"); \
  TF_OpKernelContext_Failure(ctx, status); \
  TF_DeleteStatus(status); \
}

// Metrics ops (Batch 19)
METRIC_OP(Accuracy)
METRIC_OP(BinaryAccuracy)
METRIC_OP(CategoricalAccuracy)
METRIC_OP(SparseCategoricalAccuracy)
METRIC_OP(TopKCategoricalAccuracy)
METRIC_OP(SparseTopKCategoricalAccuracy)
METRIC_OP(Precision)
METRIC_OP(Recall)
METRIC_OP(AUC)
METRIC_OP(TruePositives)
METRIC_OP(TrueNegatives)
METRIC_OP(FalsePositives)
METRIC_OP(FalseNegatives)
METRIC_OP(PrecisionAtRecall)
METRIC_OP(RecallAtPrecision)
METRIC_OP(SensitivityAtSpecificity)
METRIC_OP(SpecificityAtSensitivity)
METRIC_OP(MeanIoU)
METRIC_OP(BinaryCrossentropy)
METRIC_OP(CategoricalCrossentropy)
METRIC_OP(SparseCategoricalCrossentropy)
METRIC_OP(KLDivergence)
METRIC_OP(Poisson)
METRIC_OP(MeanSquaredError)
METRIC_OP(RootMeanSquaredError)
METRIC_OP(MeanAbsoluteError)
METRIC_OP(MeanAbsolutePercentageError)
METRIC_OP(MeanSquaredLogarithmicError)
METRIC_OP(CosineSimilarity)
METRIC_OP(LogCoshError)
METRIC_OP(Hinge)
METRIC_OP(SquaredHinge)
METRIC_OP(CategoricalHinge)
METRIC_OP(MeanTensor)
METRIC_OP(SumOverBatchSize)
METRIC_OP(MeanMetricWrapper)
METRIC_OP(F1Score)
METRIC_OP(FBetaScore)
METRIC_OP(MatthewsCorrelationCoefficient)
METRIC_OP(CohenKappa)
METRIC_OP(Hamming Distance)
METRIC_OP(JaccardIndex)
METRIC_OP(DiceCoefficient)
METRIC_OP(ConfusionMatrix)
METRIC_OP(R2Score)
METRIC_OP(MeanDirectionalAccuracy)
METRIC_OP(ExplainedVariance)

#undef METRIC_OP
