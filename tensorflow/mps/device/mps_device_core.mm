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
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <mach/mach_time.h>

#include <atomic>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cmath>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/kernels.h"

// Minimal functional MPS (Metal) StreamExecutor plugin for macOS.
// Scope: Enumerates Metal devices, creates command queues per stream, supports
// basic memcpy (HTOD/DTOH/DTOD) via blit commands, memzero (fill), events and
// host blocking. Some callbacks are stubbed or return unimplemented.
// No TensorFlow kernels are registered yet.

namespace {

constexpr char kPlatformName[] = "MPS";   // TF Platform (subtype)
constexpr char kDeviceType[] = "MPS";     // TF Device type (distinct to avoid GPU kernel collisions)

struct MPSDevice {
  id<MTLDevice> device;
  explicit MPSDevice(id<MTLDevice> d) : device(d) {}
};

struct MPSStreamStruct {  // Opaque SP_Stream
  MPSDevice* dev;
  id<MTLCommandQueue> queue;
  id<MTLCommandBuffer> last_cb;  // last enqueued CB (optional)
  explicit MPSStreamStruct(MPSDevice* d)
      : dev(d), queue([d->device newCommandQueue]), last_cb(nil) {}
  ~MPSStreamStruct() { queue = nil; last_cb = nil; }
};

struct MPSEventStruct {  // Opaque SP_Event
  dispatch_semaphore_t sema;
  MPSEventStruct() : sema(dispatch_semaphore_create(0)) {}
  ~MPSEventStruct() { sema = nullptr; }
};

inline void TfOk(TF_Status* s) { TF_SetStatus(s, TF_OK, ""); }
inline void TfUnimpl(TF_Status* s, const char* msg) {
  TF_SetStatus(s, TF_UNIMPLEMENTED, msg);
}

// ---- Dtype conversion helpers (bf16/half) ----
static inline float BFloat16ToFloat(uint16_t bf) {
  uint32_t bits = ((uint32_t)bf) << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}
static inline uint16_t FloatToBFloat16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even on the cut
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t round_bias = 0x7FFF + lsb;
  bits += round_bias;
  return (uint16_t)(bits >> 16);
}

// float16 conversion (IEEE half precision)
static inline uint16_t FloatToHalf(float f) {
  // Based on OpenEXR half conversion algorithm (simplified)
  union { uint32_t u; float f; } v = {0}; v.f = f;
  uint32_t x = v.u;
  uint32_t sign = (x >> 16) & 0x8000;
  uint32_t mant = x & 0x007FFFFF;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;  // underflow to zero
    mant = (mant | 0x00800000) >> (1 - exp);
    return (uint16_t)(sign | (mant + 0x00001000) >> 13);
  } else if (exp >= 31) {
    return (uint16_t)(sign | 0x7C00);  // Inf
  }
  return (uint16_t)(sign | (exp << 10) | ((mant + 0x00001000) >> 13));
}
static inline float HalfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x03FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) { bits = sign; }
    else {
      // subnormal
      exp = 1; while ((mant & 0x0400) == 0) { mant <<= 1; --exp; }
      mant &= 0x03FF; exp += (127 - 15);
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000 | (mant << 13);  // Inf/NaN
  } else {
    exp = exp + (127 - 15);
    bits = sign | (exp << 23) | (mant << 13);
  }
  float f; memcpy(&f, &bits, sizeof(f)); return f;
}

// Retain an Objective-C object and pass as opaque pointer.
template <typename T>
inline void* RetainToOpaque(T obj) {
  return (__bridge_retained void*)obj;
}

template <typename T>
inline T TransferFromOpaque(void* p) {
  return (__bridge_transfer T)p;
}

template <typename T>
inline T BridgeNoTransfer(void* p) {  // Do not change refcount
  return (__bridge T)p;
}

// ---- Platform callbacks ----

void GetDeviceCount(const SP_Platform* /*platform*/, int* device_count,
                    TF_Status* status) {
  @autoreleasepool {
    // Prefer MTLCopyAllDevices (macOS); fallback to default device.
    NSArray<id<MTLDevice>>* devices = nil;
    if (@available(macOS 10.11, *)) {
      if ([MTLCopyAllDevices respondsToSelector:@selector(new)]) {
        devices = MTLCopyAllDevices();
      } else {
        devices = MTLCopyAllDevices();
      }
    }
    if (!devices) {
      id<MTLDevice> d = MTLCreateSystemDefaultDevice();
      *device_count = d ? 1 : 0;
      TfOk(status);
      return;
    }
    *device_count = (int)devices.count;
    TfOk(status);
  }
}

void CreateDevice(const SP_Platform* /*platform*/,
                  SE_CreateDeviceParams* params, TF_Status* status) {
  @autoreleasepool {
    int ordinal = params->ordinal;
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    id<MTLDevice> dev = nil;
    if (devices && ordinal >= 0 && ordinal < (int)devices.count) {
      dev = devices[ordinal];
    } else {
      dev = MTLCreateSystemDefaultDevice();
    }
    if (!dev) {
      TF_SetStatus(status, TF_NOT_FOUND, "No Metal device found");
      return;
    }
    params->device->struct_size = SP_DEVICE_STRUCT_SIZE;
    params->device->ext = nullptr;
    params->device->ordinal = ordinal;
    // Keep a retained wrapper for lifetime management.
    MPSDevice* wrapper = new MPSDevice(dev);
    params->device->device_handle = static_cast<void*>(wrapper);
    params->device->hardware_name = dev.name.UTF8String;
    params->device->device_vendor = "Apple";
    params->device->pci_bus_id = nullptr;  // Not applicable
    TfOk(status);
  }
}

void DestroyDevice(const SP_Platform* /*platform*/, SP_Device* device) {
  if (!device || !device->device_handle) return;
  auto* w = static_cast<MPSDevice*>(device->device_handle);
  delete w;
  device->device_handle = nullptr;
}

int32_t DevGetNumaNode(const SP_Device* /*device*/) { return -1; }
int64_t DevGetBandwidth(const SP_Device* /*device*/) { return -1; }
double DevGetGflops(const SP_Device* /*device*/) { return -1.0; }

void CreateDeviceFns(const SP_Platform* /*platform*/,
                     SE_CreateDeviceFnsParams* params, TF_Status* status) {
  params->device_fns->struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
  params->device_fns->ext = nullptr;
  params->device_fns->get_numa_node = &DevGetNumaNode;
  params->device_fns->get_memory_bandwidth = &DevGetBandwidth;
  params->device_fns->get_gflops = &DevGetGflops;
  TfOk(status);
}

void DestroyDeviceFns(const SP_Platform* /*platform*/, SP_DeviceFns* /*fns*/) {}

// ---- StreamExecutor callbacks ----

// Allocation helpers: use private buffers for device memory; CPU staging for host.
void SE_Allocate(const SP_Device* device, uint64_t size, int64_t /*space*/,
                 SP_DeviceMemoryBase* mem) {
  @autoreleasepool {
    auto* w = static_cast<MPSDevice*>(device->device_handle);
    if (size == 0) {
      mem->opaque = nullptr; mem->size = 0; mem->payload = 0; return;
    }
    id<MTLBuffer> buf = [w->device newBufferWithLength:size options:MTLResourceStorageModePrivate];
    if (!buf) { mem->opaque = nullptr; mem->size = 0; mem->payload = 0; return; }
    mem->opaque = RetainToOpaque<id<MTLBuffer>>(buf);
    mem->size = size;
    mem->payload = 0;
  }
}

void SE_Deallocate(const SP_Device* /*device*/, SP_DeviceMemoryBase* memory) {
  if (!memory || !memory->opaque) return;
  (void)TransferFromOpaque<id<MTLBuffer>>(memory->opaque);  // release
  memory->opaque = nullptr; memory->size = 0; memory->payload = 0;
}

void* SE_HostAlloc(const SP_Device* /*device*/, uint64_t size) {
  if (size == 0) return nullptr;
  void* p = malloc(size);
  return p;
}
void SE_HostFree(const SP_Device* /*device*/, void* mem) { free(mem); }

TF_Bool SE_GetAllocStats(const SP_Device* /*device*/, SP_AllocatorStats* /*s*/) {
  return 0;  // not available
}

TF_Bool SE_DeviceMemUsage(const SP_Device* /*device*/, int64_t* /*free*/, int64_t* /*total*/) {
  return 0; // Metal doesn't expose
}

void SE_CreateStream(const SP_Device* device, SP_Stream* stream, TF_Status* status) {
  auto* w = static_cast<MPSDevice*>(device->device_handle);
  auto* s = new MPSStreamStruct(w);
  *stream = reinterpret_cast<SP_Stream>(s);
  TfOk(status);
}

void SE_DestroyStream(const SP_Device* /*device*/, SP_Stream stream) {
  if (!stream) return;
  auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
  delete s;
}

void SE_CreateStreamDependency(const SP_Device* /*device*/, SP_Stream dependent,
                               SP_Stream other, TF_Status* status) {
  // For simplicity, submit an empty command buffer on dependent after other completes.
  auto* s_dep = reinterpret_cast<MPSStreamStruct*>(dependent);
  auto* s_other = reinterpret_cast<MPSStreamStruct*>(other);
  id<MTLCommandBuffer> cb = [s_other->queue commandBuffer];
  [cb addCompletedHandler:^(__unused id<MTLCommandBuffer>) {
    id<MTLCommandBuffer> cb2 = [s_dep->queue commandBuffer];
    [cb2 commit];
  }];
  [cb commit];
  TfOk(status);
}

void SE_GetStreamStatus(const SP_Device* /*device*/, SP_Stream stream, TF_Status* status) {
  auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
  id<MTLCommandBuffer> last = s->last_cb;
  if (!last) { TfOk(status); return; }
  if (last.status == MTLCommandBufferStatusError) {
    TF_SetStatus(status, TF_INTERNAL, "Metal command buffer error");
  } else if (last.status == MTLCommandBufferStatusCompleted) {
    TfOk(status);
  } else {
    // Pending; report OK (non-blocking query)
    TfOk(status);
  }
}

void SE_CreateEvent(const SP_Device* /*device*/, SP_Event* event, TF_Status* status) {
  *event = reinterpret_cast<SP_Event>(new MPSEventStruct());
  TfOk(status);
}
void SE_DestroyEvent(const SP_Device* /*device*/, SP_Event event) {
  delete reinterpret_cast<MPSEventStruct*>(event);
}
SE_EventStatus SE_GetEventStatus(const SP_Device* /*device*/, SP_Event event) {
  // We can't poll dispatch_semaphore without affecting the count. Treat as pending.
  // A more complete impl would use MTLSharedEvent.
  (void)event; return SE_EVENT_PENDING;
}
void SE_RecordEvent(const SP_Device* /*device*/, SP_Stream stream, SP_Event event, TF_Status* status) {
  auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
  auto* ev = reinterpret_cast<MPSEventStruct*>(event);
  id<MTLCommandBuffer> cb = [s->queue commandBuffer];
  [cb addCompletedHandler:^(__unused id<MTLCommandBuffer>) {
    dispatch_semaphore_signal(ev->sema);
  }];
  [cb commit];
  s->last_cb = cb;
  TfOk(status);
}
void SE_WaitForEvent(const SP_Device* /*device*/, SP_Stream /*stream*/, SP_Event event, TF_Status* status) {
  auto* ev = reinterpret_cast<MPSEventStruct*>(event);
  dispatch_semaphore_wait(ev->sema, DISPATCH_TIME_FOREVER);
  TfOk(status);
}

struct MPSTimerStruct { uint64_t start_ns; uint64_t end_ns; };

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

// Memcpy helpers using blit encoders.
void SE_MemcpyDtoH(const SP_Device* /*device*/, SP_Stream stream, void* host_dst,
                   const SP_DeviceMemoryBase* device_src, uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> src = BridgeNoTransfer<id<MTLBuffer>>(device_src->opaque);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBuffer> staging = [s->dev->device newBufferWithLength:size options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:staging destinationOffset:0 size:size];
    [blit endEncoding];
    [cb addCompletedHandler:^(__unused id<MTLCommandBuffer>) {
      memcpy(host_dst, staging.contents, (size_t)size);
    }];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
  }
}

void SE_MemcpyHtoD(const SP_Device* /*device*/, SP_Stream stream, SP_DeviceMemoryBase* device_dst,
                   const void* host_src, uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(device_dst->opaque);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBuffer> staging = [s->dev->device newBufferWithLength:size options:MTLResourceStorageModeShared];
    memcpy(staging.contents, host_src, (size_t)size);
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
  }
}

void SE_MemcpyDtoD(const SP_Device* /*device*/, SP_Stream stream, SP_DeviceMemoryBase* device_dst,
                   const SP_DeviceMemoryBase* device_src, uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(device_dst->opaque);
    id<MTLBuffer> src = BridgeNoTransfer<id<MTLBuffer>>(device_src->opaque);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
  }
}

void SE_SyncDtoH(const SP_Device* /*device*/, void* host_dst, const SP_DeviceMemoryBase* device_src,
                 uint64_t size, TF_Status* status) {
  @autoreleasepool {
    // Use blit then wait for completion.
    // Create a temporary stream to run blocking copy.
    id<MTLDevice> dev = BridgeNoTransfer<id<MTLBuffer>>(device_src->opaque).device;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBuffer> src = BridgeNoTransfer<id<MTLBuffer>>(device_src->opaque);
    id<MTLBuffer> staging = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:staging destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(host_dst, staging.contents, (size_t)size);
    TfOk(status);
  }
}

void SE_SyncHtoD(const SP_Device* /*device*/, SP_DeviceMemoryBase* device_dst, const void* host_src,
                 uint64_t size, TF_Status* status) {
  @autoreleasepool {
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(device_dst->opaque);
    id<MTLCommandQueue> q = [dst.device newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBuffer> staging = [dst.device newBufferWithLength:size options:MTLResourceStorageModeShared];
    memcpy(staging.contents, host_src, (size_t)size);
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    TfOk(status);
  }
}

void SE_SyncDtoD(const SP_Device* /*device*/, SP_DeviceMemoryBase* device_dst,
                 const SP_DeviceMemoryBase* device_src, uint64_t size, TF_Status* status) {
  @autoreleasepool {
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(device_dst->opaque);
    id<MTLBuffer> src = BridgeNoTransfer<id<MTLBuffer>>(device_src->opaque);
    id<MTLCommandQueue> q = [dst.device newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    TfOk(status);
  }
}

void SE_BlockHostForEvent(const SP_Device* /*device*/, SP_Event event, TF_Status* status) {
  auto* ev = reinterpret_cast<MPSEventStruct*>(event);
  dispatch_semaphore_wait(ev->sema, DISPATCH_TIME_FOREVER);
  TfOk(status);
}

void SE_BlockHostUntilDone(const SP_Device* /*device*/, SP_Stream stream, TF_Status* status) {
  auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
  id<MTLCommandBuffer> last = s->last_cb;
  if (last) { [last waitUntilCompleted]; }
  TfOk(status);
}

void SE_SynchronizeAll(const SP_Device* device, TF_Status* status) {
  // Submit and wait on a no-op command buffer on a fresh queue.
  auto* w = static_cast<MPSDevice*>(device->device_handle);
  id<MTLCommandQueue> q = [w->device newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];
  TfOk(status);
}

void SE_MemZero(const SP_Device* /*device*/, SP_Stream stream, SP_DeviceMemoryBase* location,
                uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(location->opaque);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst range:NSMakeRange(0, (NSUInteger)size) value:0];
    [blit endEncoding];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
  }
}

void SE_Memset(const SP_Device* /*device*/, SP_Stream stream, SP_DeviceMemoryBase* location,
               uint8_t pattern, uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(location->opaque);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:dst range:NSMakeRange(0, (NSUInteger)size) value:pattern];
    [blit endEncoding];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
  }
}

// Forward declaration for memset32 pipeline helper
static void EnsureMemset32Pipeline(id<MTLDevice> dev);

void SE_Memset32(const SP_Device* /*device*/, SP_Stream stream, SP_DeviceMemoryBase* location,
                 uint32_t pattern, uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLDevice> dev = s->dev->device;
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(location->opaque);
    uint64_t words = size / 4;
    uint64_t tail = size - words * 4;
    if (words == 0) {
      // Just do byte fill for tiny sizes
      id<MTLCommandBuffer> cb = [s->queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit fillBuffer:dst range:NSMakeRange(0, (NSUInteger)size) value:(uint8_t)(pattern & 0xFF)];
      [blit endEncoding];
      [cb commit]; s->last_cb = cb; TfOk(status); return;
    }
    EnsureMemset32Pipeline(dev);
    id<MTLBuffer> patbuf = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    *(uint32_t*)patbuf.contents = pattern;
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_memset32_pipeline];
    [enc setBuffer:dst offset:0 atIndex:0];
    [enc setBuffer:patbuf offset:0 atIndex:1];
    NSUInteger threads = 256; NSUInteger grid = (NSUInteger)words;
    NSUInteger groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding];
    if (tail) {
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit fillBuffer:dst range:NSMakeRange((NSUInteger)(words*4), (NSUInteger)tail) value:(uint8_t)(pattern & 0xFF)];
      [blit endEncoding];
    }
    [cb commit]; s->last_cb = cb; TfOk(status);
  }
}

TF_Bool SE_HostCallback(const SP_Device* /*device*/, SP_Stream /*stream*/,
                        SE_StatusCallbackFn callback_fn, void* callback_arg) {
  // Execute callback immediately on host.
  tensorflow::TF_StatusPtr s(TF_NewStatus());
  callback_fn(callback_arg, s.get());
  return 1;
}

void CreateStreamExecutor(const SP_Platform* /*platform*/,
                          SE_CreateStreamExecutorParams* params,
                          TF_Status* status) {
  SP_StreamExecutor* se = params->stream_executor;
  se->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  se->ext = nullptr;

  se->allocate = &SE_Allocate;
  se->deallocate = &SE_Deallocate;
  se->host_memory_allocate = &SE_HostAlloc;
  se->host_memory_deallocate = &SE_HostFree;
  se->unified_memory_allocate = nullptr;
  se->unified_memory_deallocate = nullptr;
  se->get_allocator_stats = &SE_GetAllocStats;
  se->device_memory_usage = &SE_DeviceMemUsage;

  se->create_stream = &SE_CreateStream;
  se->destroy_stream = &SE_DestroyStream;
  se->create_stream_dependency = &SE_CreateStreamDependency;
  se->get_stream_status = &SE_GetStreamStatus;

  se->create_event = &SE_CreateEvent;
  se->destroy_event = &SE_DestroyEvent;
  se->get_event_status = &SE_GetEventStatus;
  se->record_event = &SE_RecordEvent;
  se->wait_for_event = &SE_WaitForEvent;

  se->create_timer = &SE_CreateTimer;
  se->destroy_timer = &SE_DestroyTimer;
  se->start_timer = &SE_StartTimer;
  se->stop_timer = &SE_StopTimer;

  se->memcpy_dtoh = &SE_MemcpyDtoH;
  se->memcpy_htod = &SE_MemcpyHtoD;
  se->memcpy_dtod = &SE_MemcpyDtoD;
  se->sync_memcpy_dtoh = &SE_SyncDtoH;
  se->sync_memcpy_htod = &SE_SyncHtoD;
  se->sync_memcpy_dtod = &SE_SyncDtoD;
  se->block_host_for_event = &SE_BlockHostForEvent;
  se->block_host_until_done = &SE_BlockHostUntilDone;
  se->synchronize_all_activity = &SE_SynchronizeAll;
  se->mem_zero = &SE_MemZero;
  se->memset = &SE_Memset;
  se->memset32 = &SE_Memset32;
  se->host_callback = &SE_HostCallback;

  TfOk(status);
}

void DestroyStreamExecutor(const SP_Platform* /*platform*/, SP_StreamExecutor* /*se*/) {}

static uint64_t MPSTimerNanoseconds(SP_Timer timer) {
  auto* t = reinterpret_cast<MPSTimerStruct*>(timer);
  if (!t) return 0;
  if (t->end_ns < t->start_ns) return 0;
  return t->end_ns - t->start_ns;
}

void CreateTimerFns(const SP_Platform* /*platform*/, SP_TimerFns* timer_fns,
                    TF_Status* status) {
  timer_fns->struct_size = SP_TIMER_FNS_STRUCT_SIZE;
  timer_fns->ext = nullptr;
  timer_fns->nanoseconds = &MPSTimerNanoseconds;
  TfOk(status);
}

void DestroyTimerFns(const SP_Platform* /*platform*/, SP_TimerFns* /*timer_fns*/) {}

void DestroyPlatform(SP_Platform* /*platform*/) {}
void DestroyPlatformFns(SP_PlatformFns* /*platform_fns*/) {}

}  // namespace

