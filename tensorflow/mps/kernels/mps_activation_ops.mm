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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/mps/ops/mps_ops_registry.h"
#include "tensorflow/c/kernels.h"

// ============================================================================
// Activation Operations (9 ops)
// ============================================================================
// Relu, Relu6, Elu, Selu, LeakyRelu, Gelu, Swish, Softplus, Softsign

namespace tensorflow {
namespace mps {

// TODO: Extract from mps_pluggable_device_plugin.mm

void* MPSRelu_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSRelu_Compute(void* kernel, TF_OpKernelContext* ctx) { /* TODO */ }
void MPSRelu_Delete(void* kernel) {}

// ... (8 more activation ops)

void RegisterActivationOps(const char* platform_name, TF_Status* status) {
  // REGISTER_MPS_UNARY_OP_3DTYPE(Relu, platform_name, status);
  // REGISTER_MPS_UNARY_OP_3DTYPE(Gelu, platform_name, status);
  // ... etc
}

}  // namespace mps
}  // namespace tensorflow
