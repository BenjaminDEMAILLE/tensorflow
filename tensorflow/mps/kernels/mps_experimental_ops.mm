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

// Experimental operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

#define EXP_OP(name) \
  extern "C" void* MPS##name##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
  extern "C" void MPS##name##_Delete(void* kernel) {} \
  extern "C" void MPS##name##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
    TF_Status* status = TF_NewStatus(); \
    TF_SetStatus(status, TF_UNIMPLEMENTED, #name " is experimental"); \
    TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status); \
  }

EXP_OP(ExperimentalThreadPoolHandle)
EXP_OP(ExperimentalThreadPoolDataset)
EXP_OP(ExperimentalMapAndBatchDataset)
EXP_OP(ExperimentalParallelInterleaveDataset)
EXP_OP(ExperimentalScanDataset)
EXP_OP(ExperimentalGroupByWindowDataset)
EXP_OP(ExperimentalDenseToSparseBatchDataset)
EXP_OP(ExperimentalUnbatchDataset)
EXP_OP(ExperimentalUniqueDataset)
EXP_OP(ExperimentalSqlDataset)
EXP_OP(ExperimentalStatsAggregatorHandle)
EXP_OP(ExperimentalStatsAggregatorSummary)
EXP_OP(ExperimentalSetStatsAggregatorDataset)
EXP_OP(ExperimentalBytesProducedStatsDataset)
EXP_OP(ExperimentalLatencyStatsDataset)
EXP_OP(ExperimentalSleepDataset)
EXP_OP(ExperimentalAssertNextDataset)
EXP_OP(ExperimentalIgnoreErrorsDataset)
EXP_OP(ExperimentalPrivateThreadPoolDataset)
EXP_OP(ExperimentalMaxIntraOpParallelismDataset)
EXP_OP(ExperimentalNonSerializableDataset)
EXP_OP(ExperimentalParseExampleDataset)
EXP_OP(ExperimentalCSVDataset)
EXP_OP(ExperimentalAutoShardDataset)
EXP_OP(ExperimentalMatchingFilesDataset)
EXP_OP(ExperimentalChooseFastestDataset)
EXP_OP(ExperimentalRebatchDataset)
EXP_OP(ExperimentalDatasetCardinality)
EXP_OP(ExperimentalDirectedInterleaveDataset)
EXP_OP(ExperimentalIteratorGetDevice)
