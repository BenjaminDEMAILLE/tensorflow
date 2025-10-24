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

#import "tensorflow/mps/device/mps_device.h"
#import "tensorflow/mps/utils/mps_utils.h"

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/kernels.h"

using namespace tensorflow::mps;

namespace {

inline void TfOk(TF_Status* s) { TF_SetStatus(s, TF_OK, ""); }
inline void TfUnimpl(TF_Status* s, const char* msg) {
  TF_SetStatus(s, TF_UNIMPLEMENTED, msg);
}

double DevGetGflops(const SP_Device* /*device*/) { return -1.0; }

void CreateDeviceFns(const SP_Platform* /*platform*/,
                     SE_CreateDeviceFnsParams* params, TF_Status* status) {
  params->device_fns->struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
  params->device_fns->ext = nullptr;
  params->device_fns->get_numa_node = &tensorflow::mps::DevGetNumaNode;
  params->device_fns->get_memory_bandwidth = &tensorflow::mps::DevGetBandwidth;
  params->device_fns->get_gflops = &DevGetGflops;
  TfOk(status);
}

void DestroyDeviceFns(const SP_Platform* /*platform*/, SP_DeviceFns* /*fns*/) {}

TF_Bool SE_GetAllocStats(const SP_Device* /*device*/, SP_AllocatorStats* /*s*/) {
  return 0;  // not available
}

TF_Bool SE_DeviceMemUsage(const SP_Device* /*device*/, int64_t* /*free*/, int64_t* /*total*/) {
  return 0;  // Metal doesn't expose
}

SE_EventStatus SE_GetEventStatus(const SP_Device* /*device*/, SP_Event /*event*/) {
  return SE_EVENT_PENDING;
}

// Timer implementation
struct MPSTimerStruct {
  uint64_t start_ns;
  uint64_t end_ns;
};

static inline uint64_t NowNanos() {
  static mach_timebase_info_data_t timebase = {0, 0};
  if (timebase.denom == 0) mach_timebase_info(&timebase);
  uint64_t t = mach_absolute_time();
  return (t * timebase.numer) / timebase.denom;
}

void SE_CreateTimer(const SP_Device* /*device*/, SP_Timer* timer, TF_Status* status) {
  auto* t = new MPSTimerStruct{0, 0};
  *timer = reinterpret_cast<SP_Timer>(t);
  TfOk(status);
}

void SE_DestroyTimer(const SP_Device* /*device*/, SP_Timer timer) {
  if (!timer) return;
  delete reinterpret_cast<MPSTimerStruct*>(timer);
}

void SE_StartTimer(const SP_Device* /*device*/, SP_Stream /*stream*/, SP_Timer timer, TF_Status* status) {
  auto* t = reinterpret_cast<MPSTimerStruct*>(timer);
  t->start_ns = NowNanos();
  TfOk(status);
}

void SE_StopTimer(const SP_Device* /*device*/, SP_Stream /*stream*/, SP_Timer timer, TF_Status* status) {
  auto* t = reinterpret_cast<MPSTimerStruct*>(timer);
  t->end_ns = NowNanos();
  TfOk(status);
}

void SE_BlockHostForEvent(const SP_Device* /*device*/, SP_Event event, TF_Status* status) {
  auto* ev = reinterpret_cast<tensorflow::mps::MPSEventStruct*>(event);
  dispatch_semaphore_wait(ev->sema, DISPATCH_TIME_FOREVER);
  TfOk(status);
}

void SE_SynchronizeAll(const SP_Device* device, TF_Status* status) {
  auto* w = static_cast<tensorflow::mps::MPSDevice*>(device->device_handle);
  id<MTLCommandQueue> q = [w->device newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];
  TfOk(status);
}

// ---- Platform Registration ----

void CreateStreamExecutor(const SP_Platform* /*platform*/,
                          SE_CreateStreamExecutorParams* params, TF_Status* status) {
  params->stream_executor->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  params->stream_executor->ext = nullptr;
  
  // Memory operations
  params->stream_executor->allocate = &tensorflow::mps::MemAllocate;
  params->stream_executor->deallocate = &tensorflow::mps::MemDeallocate;
  params->stream_executor->host_memory_allocate = &tensorflow::mps::HostMemAllocate;
  params->stream_executor->host_memory_deallocate = &tensorflow::mps::HostMemDeallocate;
  params->stream_executor->get_allocator_stats = &SE_GetAllocStats;
  params->stream_executor->device_memory_usage = &SE_DeviceMemUsage;
  
  // Stream operations
  params->stream_executor->create_stream = &tensorflow::mps::CreateStream;
  params->stream_executor->destroy_stream = &tensorflow::mps::DestroyStream;
  params->stream_executor->create_stream_dependency = &tensorflow::mps::CreateStreamDependency;
  params->stream_executor->get_stream_status = &tensorflow::mps::GetStreamStatus;
  
  // Event operations
  params->stream_executor->create_event = &tensorflow::mps::CreateEvent;
  params->stream_executor->destroy_event = &tensorflow::mps::DestroyEvent;
  params->stream_executor->get_event_status = &SE_GetEventStatus;
  params->stream_executor->record_event = &tensorflow::mps::RecordEvent;
  params->stream_executor->wait_for_event = &tensorflow::mps::WaitForEvent;
  
  // Timer operations
  params->stream_executor->create_timer = &SE_CreateTimer;
  params->stream_executor->destroy_timer = &SE_DestroyTimer;
  params->stream_executor->start_timer = &SE_StartTimer;
  params->stream_executor->stop_timer = &SE_StopTimer;
  
  // Memory copy operations
  params->stream_executor->memcpy_dtoh = &tensorflow::mps::MemcpyDeviceToHost;
  params->stream_executor->memcpy_htod = &tensorflow::mps::MemcpyHostToDevice;
  params->stream_executor->memcpy_dtod = &tensorflow::mps::MemcpyDeviceToDevice;
  params->stream_executor->sync_memcpy_dtoh = &tensorflow::mps::SyncMemcpyDeviceToHost;
  params->stream_executor->sync_memcpy_htod = &tensorflow::mps::SyncMemcpyHostToDevice;
  params->stream_executor->sync_memcpy_dtod = &tensorflow::mps::SyncMemcpyDeviceToDevice;
  
  // Synchronization
  params->stream_executor->block_host_until_done = &tensorflow::mps::BlockHostUntilDone;
  params->stream_executor->block_host_for_event = &SE_BlockHostForEvent;
  params->stream_executor->synchronize_all_activity = &SE_SynchronizeAll;
  
  // Memory fill operations
  params->stream_executor->mem_zero = &tensorflow::mps::MemZero;
  params->stream_executor->memset = &tensorflow::mps::Memset;
  params->stream_executor->memset32 = &tensorflow::mps::Memset32;
  
  TfOk(status);
}

void DestroyStreamExecutor(const SP_Platform* /*platform*/, SP_StreamExecutor* /*se*/) {}

void CreatePlatformFns(const SP_Platform* /*platform*/,
                       SP_PlatformFns* const platform_fns, TF_Status* const status) {
  platform_fns->struct_size = SP_PLATFORM_FNS_STRUCT_SIZE;
  platform_fns->ext = nullptr;
  TfOk(status);
}

void DestroyPlatformFns(const SP_Platform* /*platform*/, SP_PlatformFns* /*fns*/) {}

void CreatePlatform(SP_Platform* const platform, TF_Status* const status) {
  platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  platform->ext = nullptr;
  platform->name = tensorflow::mps::kPlatformName;
  platform->type = tensorflow::mps::kDeviceType;
  TfOk(status);
}

void DestroyPlatform(SP_Platform* platform) {
  platform->struct_size = 0;
}

}  // namespace

// ---- Plugin Registration Entry Point ----
extern "C" {

// Forward declarations for kernel registration (defined in ops module)
void RegisterMPSKernels(const char* platform_name, TF_Status* status);

void SE_InitPlugin(SE_PlatformRegistrationParams* const params, TF_Status* const status) {
  // Register platform
  params->platform->struct_size = SP_PLATFORM_STRUCT_SIZE;
  params->platform->ext = nullptr;
  params->platform->name = tensorflow::mps::kPlatformName;
  params->platform->type = tensorflow::mps::kDeviceType;
  
  params->platform_fns->get_device_count = &tensorflow::mps::GetDeviceCount;
  params->platform_fns->create_device = &tensorflow::mps::CreateDevice;
  params->platform_fns->destroy_device = &tensorflow::mps::DestroyDevice;
  params->platform_fns->create_device_fns = &CreateDeviceFns;
  params->platform_fns->destroy_device_fns = &DestroyDeviceFns;
  params->platform_fns->create_stream_executor = &CreateStreamExecutor;
  params->platform_fns->destroy_stream_executor = &DestroyStreamExecutor;
  params->platform_fns->create_platform_fns = &CreatePlatformFns;
  params->platform_fns->destroy_platform_fns = &DestroyPlatformFns;
  params->platform_fns->create_platform = &CreatePlatform;
  params->platform_fns->destroy_platform = &DestroyPlatform;
  
  TfOk(status);
  
  // Register all MPS kernels
  RegisterMPSKernels(tensorflow::mps::kPlatformName, status);
}

}  // extern "C"
