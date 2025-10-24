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

// Deprecated/Legacy operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

#define LEGACY_OP(name) \
  extern "C" void* MPS##name##_Create(TF_OpKernelConstruction* ctx) { return nullptr; } \
  extern "C" void MPS##name##_Delete(void* kernel) {} \
  extern "C" void MPS##name##_Compute(void* kernel, TF_OpKernelContext* ctx) { \
    TF_Status* status = TF_NewStatus(); \
    TF_SetStatus(status, TF_UNIMPLEMENTED, #name " is deprecated/legacy"); \
    TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status); \
  }

LEGACY_OP(AdjustContrastv2)
LEGACY_OP(AdjustSaturation)
LEGACY_OP(AdjustHue)
LEGACY_OP(DrawBoundingBoxes)
LEGACY_OP(DrawBoundingBoxesV2)
LEGACY_OP(SampleDistortedBoundingBox)
LEGACY_OP(SampleDistortedBoundingBoxV2)
LEGACY_OP(NonMaxSuppression)
LEGACY_OP(NonMaxSuppressionV2)
LEGACY_OP(NonMaxSuppressionV3)
LEGACY_OP(NonMaxSuppressionV4)
LEGACY_OP(NonMaxSuppressionV5)
LEGACY_OP(NonMaxSuppressionWithOverlaps)
LEGACY_OP(CombinedNonMaxSuppression)
LEGACY_OP(ExtractGlimpse)
LEGACY_OP(CropAndResize)
LEGACY_OP(CropAndResizeGradImage)
LEGACY_OP(CropAndResizeGradBoxes)
LEGACY_OP(ResizeArea)
LEGACY_OP(ResizeBicubic)
LEGACY_OP(ResizeBicubicGrad)
LEGACY_OP(ResizeBilinear)
LEGACY_OP(ResizeBilinearGrad)
LEGACY_OP(ResizeNearestNeighbor)
LEGACY_OP(ResizeNearestNeighborGrad)
LEGACY_OP(ScaleAndTranslate)
LEGACY_OP(ScaleAndTranslateGrad)
LEGACY_OP(GenerateBoundingBoxProposals)
LEGACY_OP(IOU)
