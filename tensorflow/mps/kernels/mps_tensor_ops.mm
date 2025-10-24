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
#include "tensorflow/c/kernels.h"

// ============================================================================
// Tensor Operations (15 ops)
// ============================================================================
// Cast, Reshape, Transpose, Concat, Slice, StridedSlice, Fill, ZerosLike, 
// OnesLike, Pad, MirrorPad, Tile, Select, ClipByValue

namespace tensorflow {
namespace mps {

// TODO: Extract from mps_pluggable_device_plugin.mm

void RegisterTensorOps(const char* platform_name, TF_Status* status) {
  // TODO: Registration
}

}  // namespace mps
}  // namespace tensorflow
