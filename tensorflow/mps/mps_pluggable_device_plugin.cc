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

#include <cstring>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"

// Minimal internal PluggableDevice plugin skeleton for macOS Apple Silicon.
// NOTE: This is a stub. It registers an MPS platform with zero devices so that
// TensorFlow can discover the platform without relying on an external
// tensorflow-metal wheel. No kernels or runtime functionality are provided.
// Extending this to a full MPS backend requires implementing StreamExecutor
// callbacks and kernel registrations backed by Metal/MPS.

namespace {

constexpr char kPlatformName[] = "MPS";  // Subtype name shown by TF
constexpr char kDeviceType[] = "GPU";    // Device type category

void GetDeviceCount(const SP_Platform* /*platform*/, int* device_count,
                    TF_Status* /*status*/) {
  // Stub plugin: report 0 devices to avoid interfering with runtime.
  *device_count = 0;
}

void CreateDevice(const SP_Platform* /*platform*/,
                  SE_CreateDeviceParams* /*params*/, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "TensorFlow MPS plugin stub: device creation unimplemented");
}

void DestroyDevice(const SP_Platform* /*platform*/, SP_Device* /*device*/) {}

void CreateDeviceFns(const SP_Platform* /*platform*/,
                     SE_CreateDeviceFnsParams* /*params*/, TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "TensorFlow MPS plugin stub: device fns unimplemented");
}

void DestroyDeviceFns(const SP_Platform* /*platform*/,
                      SP_DeviceFns* /*device_fns*/) {}

void CreateStreamExecutor(const SP_Platform* /*platform*/,
                          SE_CreateStreamExecutorParams* /*params*/,
                          TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "TensorFlow MPS plugin stub: stream executor unimplemented");
}

void DestroyStreamExecutor(const SP_Platform* /*platform*/,
                           SP_StreamExecutor* /*se*/) {}

void CreateTimerFns(const SP_Platform* /*platform*/, SP_TimerFns* /*timer_fns*/,
                    TF_Status* status) {
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "TensorFlow MPS plugin stub: timer fns unimplemented");
}

void DestroyTimerFns(const SP_Platform* /*platform*/,
                     SP_TimerFns* /*timer_fns*/) {}

void DestroyPlatform(SP_Platform* platform) {
  // Only fields allocated here must be freed here. We only used static strings
  // for name/type, so nothing to free in this stub.
  (void)platform;  // unused
}

void DestroyPlatformFns(SP_PlatformFns* /*platform_fns*/) {}

}  // namespace

extern "C" {

// Entry point called by TensorFlow to initialize a StreamExecutor-based
// PluggableDevice plugin.
void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  if (!params || !status) return;

  // Version compatibility is managed by TF; we fill platform and fns.
  static SP_Platform platform = {SP_PLATFORM_STRUCT_SIZE};
  platform.ext = nullptr;
  platform.name = kPlatformName;
  platform.type = kDeviceType;
  platform.supports_unified_memory = /*TF_False*/ 0;
  platform.use_bfc_allocator = /*TF_True*/ 1;
  platform.force_memory_growth = /*TF_False*/ 0;

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

  // Success.
  TF_SetStatus(status, TF_OK, "");
}

// Optional kernel registration entry point; empty in stub.
void TF_InitKernel() {}

}  // extern "C"
