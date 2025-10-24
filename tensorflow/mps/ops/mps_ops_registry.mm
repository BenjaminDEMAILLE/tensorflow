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

#include "tensorflow/mps/ops/mps_ops_registry.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// Main kernel registration function
// Called by SE_InitPlugin to register all MPS kernels
void RegisterMPSKernels(const char* platform_name, TF_Status* status) {
  // Register all operation categories
  RegisterElementwiseOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterActivationOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterComparisonOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterLogicalOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterReductionOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterTensorOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterIndexingOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterNNOps(platform_name, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  RegisterUtilityOps(platform_name, status);
}

}  // namespace mps
}  // namespace tensorflow
