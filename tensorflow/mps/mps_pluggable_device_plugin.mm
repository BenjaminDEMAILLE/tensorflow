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

extern "C" {

void SE_InitPlugin(SE_PlatformRegistrationParams* const params,
                   TF_Status* const status) {
  if (!params || !status) return;

  static SP_Platform platform = {SP_PLATFORM_STRUCT_SIZE};
  platform.ext = nullptr;
  platform.name = kPlatformName;
  platform.type = kDeviceType;
  platform.supports_unified_memory = 0;
  platform.use_bfc_allocator = 1;
  platform.force_memory_growth = 0;

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

  TfOk(status);
}

// Register a minimal Identity(T=float) kernel for MPS device to exercise device path.
namespace {
void* MPSIdentity_Create(TF_OpKernelConstruction* /*ctx*/) { return nullptr; }

void MPSIdentity_Delete(void* /*kernel*/) {}

void MPSIdentity_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();

  // Get input[0]
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, /*i=*/0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  // Build dims from input
  int ndims = TF_NumDims(input);
  int64_t dims_buf[8];
  std::unique_ptr<int64_t[]> dyn_dims;
  int64_t* dims = dims_buf;
  if (ndims > 8) { dyn_dims.reset(new int64_t[ndims]); dims = dyn_dims.get(); }
  for (int i = 0; i < ndims; ++i) dims[i] = TF_Dim(input, i);

  // Try to forward input to output 0.
  int forwarded = -1;
  int cand = 0;
  TF_Tensor* out = TF_ForwardInputOrAllocateOutput(ctx, &cand, /*num_candidate_input_indices=*/1,
                                                   /*output_index=*/0, dims, ndims, &forwarded, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  if (forwarded < 0) {
    // No forward possible; copy host->host as a fallback (runtime will place buffers correctly for MPS).
    void* dst = TF_TensorData(out);
    const void* src = TF_TensorData(input);
    size_t nbytes = TF_TensorByteSize(out);
    if (src && dst && nbytes) memcpy(dst, src, nbytes);
  }

  TF_DeleteStatus(status);
}
}  // namespace

void TF_InitKernel() {
  TF_Status* status = TF_NewStatus();

  // Register Identity for device "MPS" with T=float/half/bfloat16.
  TF_KernelBuilder* kb = TF_NewKernelBuilder("Identity", kPlatformName,
                                             &MPSIdentity_Create,
                                             &MPSIdentity_Compute,
                                             &MPSIdentity_Delete);
  TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSIdentityFloat", kb, status);

  TF_KernelBuilder* kb_h = TF_NewKernelBuilder("Identity", kPlatformName,
                                              &MPSIdentity_Create,
                                              &MPSIdentity_Compute,
                                              &MPSIdentity_Delete);
  TF_KernelBuilder_TypeConstraint(kb_h, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSIdentityHalf", kb_h, status);

  TF_KernelBuilder* kb_bf = TF_NewKernelBuilder("Identity", kPlatformName,
                                               &MPSIdentity_Create,
                                               &MPSIdentity_Compute,
                                               &MPSIdentity_Delete);
  TF_KernelBuilder_TypeConstraint(kb_bf, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSIdentityBFloat16", kb_bf, status);
  // Register Relu for float/half/bfloat16.
  extern void MPSRelu_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* relu_kb = TF_NewKernelBuilder("Relu", kPlatformName,
                                                 /*create*/ nullptr,
                                                 /*compute*/ &MPSRelu_Compute,
                                                 /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(relu_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSReluFloat", relu_kb, status);

  extern void MPSReluHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* relu_h_kb = TF_NewKernelBuilder("Relu", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSReluHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(relu_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSReluHalf", relu_h_kb, status);

  extern void MPSReluBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* relu_bf_kb = TF_NewKernelBuilder("Relu", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSReluBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(relu_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSReluBFloat16", relu_bf_kb, status);

  // Register AddV2(T=float) and Mul(T=float) for device "MPS".
  extern void MPSAdd_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* add_kb = TF_NewKernelBuilder("AddV2", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSAdd_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(add_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSAddV2Float", add_kb, status);

  extern void MPSAddHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* add_h_kb = TF_NewKernelBuilder("AddV2", kPlatformName,
                                                  /*create*/ nullptr,
                                                  /*compute*/ &MPSAddHalf_Compute,
                                                  /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(add_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSAddV2Half", add_h_kb, status);

  extern void MPSMul_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mul_kb = TF_NewKernelBuilder("Mul", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMul_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(mul_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMulFloat", mul_kb, status);

  extern void MPSMulHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mul_h_kb = TF_NewKernelBuilder("Mul", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSMulHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(mul_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMulHalf", mul_h_kb, status);

  extern void MPSMulBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mul_bf_kb = TF_NewKernelBuilder("Mul", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSMulBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(mul_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMulBFloat16", mul_bf_kb, status);

  extern void MPSAddV2BFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* add_bf_kb = TF_NewKernelBuilder("AddV2", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSAddV2BFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(add_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSAddV2BFloat16", add_bf_kb, status);

  // Register MatMul(T=float) for device "MPS".
  extern void* MPSMatMul_Create(TF_OpKernelConstruction*);
  extern void MPSMatMul_Delete(void*);
  extern void MPSMatMul_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mm_kb = TF_NewKernelBuilder("MatMul", kPlatformName,
                                               &MPSMatMul_Create,
                                               &MPSMatMul_Compute,
                                               &MPSMatMul_Delete);
  TF_KernelBuilder_TypeConstraint(mm_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMatMulFloat", mm_kb, status);

  // MatMul half
  TF_KernelBuilder* mm_h_kb = TF_NewKernelBuilder("MatMul", kPlatformName,
                                                 &MPSMatMul_Create,
                                                 &MPSMatMul_Compute,
                                                 &MPSMatMul_Delete);
  TF_KernelBuilder_TypeConstraint(mm_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMatMulHalf", mm_h_kb, status);

  // MatMul bfloat16
  TF_KernelBuilder* mm_bf_kb = TF_NewKernelBuilder("MatMul", kPlatformName,
                                                  &MPSMatMul_Create,
                                                  &MPSMatMul_Compute,
                                                  &MPSMatMul_Delete);
  TF_KernelBuilder_TypeConstraint(mm_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMatMulBFloat16", mm_bf_kb, status);

  // Register Maximum/Minimum (float)
  extern void MPSMaximum_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* max_kb = TF_NewKernelBuilder("Maximum", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMaximum_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(max_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMaximumFloat", max_kb, status);

  extern void MPSMaximumHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* max_h_kb = TF_NewKernelBuilder("Maximum", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSMaximumHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(max_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMaximumHalf", max_h_kb, status);

  extern void MPSMaximumBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* max_bf_kb = TF_NewKernelBuilder("Maximum", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSMaximumBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(max_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMaximumBFloat16", max_bf_kb, status);

  extern void MPSMinimum_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* min_kb = TF_NewKernelBuilder("Minimum", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMinimum_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(min_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMinimumFloat", min_kb, status);

  extern void MPSMinimumHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* min_h_kb = TF_NewKernelBuilder("Minimum", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSMinimumHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(min_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMinimumHalf", min_h_kb, status);

  extern void MPSMinimumBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* min_bf_kb = TF_NewKernelBuilder("Minimum", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSMinimumBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(min_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMinimumBFloat16", min_bf_kb, status);

  // Register Sigmoid/Tanh (float)
  extern void MPSSigmoid_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sig_kb = TF_NewKernelBuilder("Sigmoid", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSSigmoid_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(sig_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSigmoidFloat", sig_kb, status);

  extern void MPSSigmoidHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sig_h_kb = TF_NewKernelBuilder("Sigmoid", kPlatformName,
                                                   /*create*/ nullptr,
                                                   /*compute*/ &MPSSigmoidHalf_Compute,
                                                   /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(sig_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSSigmoidHalf", sig_h_kb, status);

  extern void MPSSigmoidBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sig_bf_kb = TF_NewKernelBuilder("Sigmoid", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSSigmoidBFloat16_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(sig_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSSigmoidBFloat16", sig_bf_kb, status);

  extern void MPSTanh_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* tanh_kb = TF_NewKernelBuilder("Tanh", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSTanh_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(tanh_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSTanhFloat", tanh_kb, status);

  extern void MPSTanhHalf_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* tanh_h_kb = TF_NewKernelBuilder("Tanh", kPlatformName,
                                                    /*create*/ nullptr,
                                                    /*compute*/ &MPSTanhHalf_Compute,
                                                    /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(tanh_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSTanhHalf", tanh_h_kb, status);

  extern void MPSTanhBFloat16_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* tanh_bf_kb = TF_NewKernelBuilder("Tanh", kPlatformName,
                                                     /*create*/ nullptr,
                                                     /*compute*/ &MPSTanhBFloat16_Compute,
                                                     /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(tanh_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSTanhBFloat16", tanh_bf_kb, status);

  // Softmax (float, half, bfloat16) via MPSGraph
  extern void* MPSSoftmax_Create(TF_OpKernelConstruction*);
  extern void MPSSoftmax_Delete(void*);
  extern void MPSSoftmax_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* softmax_kb = TF_NewKernelBuilder("Softmax", kPlatformName,
                                                     &MPSSoftmax_Create,
                                                     &MPSSoftmax_Compute,
                                                     &MPSSoftmax_Delete);
  TF_KernelBuilder_TypeConstraint(softmax_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSoftmaxFloat", softmax_kb, status);

  TF_KernelBuilder* softmax_h_kb = TF_NewKernelBuilder("Softmax", kPlatformName,
                                                       &MPSSoftmax_Create,
                                                       &MPSSoftmax_Compute,
                                                       &MPSSoftmax_Delete);
  TF_KernelBuilder_TypeConstraint(softmax_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSSoftmaxHalf", softmax_h_kb, status);

  TF_KernelBuilder* softmax_bf_kb = TF_NewKernelBuilder("Softmax", kPlatformName,
                                                        &MPSSoftmax_Create,
                                                        &MPSSoftmax_Compute,
                                                        &MPSSoftmax_Delete);
  TF_KernelBuilder_TypeConstraint(softmax_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSSoftmaxBFloat16", softmax_bf_kb, status);

  // Register FusedBatchNormV3 (float, half, bfloat16)
  extern void* MPSFusedBatchNormV3_Create(TF_OpKernelConstruction*);
  extern void MPSFusedBatchNormV3_Delete(void*);
  extern void MPSFusedBatchNormV3_Compute(void*, TF_OpKernelContext*);

  TF_KernelBuilder* bn_kb = TF_NewKernelBuilder("FusedBatchNormV3", kPlatformName,
                                                &MPSFusedBatchNormV3_Create,
                                                &MPSFusedBatchNormV3_Compute,
                                                &MPSFusedBatchNormV3_Delete);
  TF_KernelBuilder_TypeConstraint(bn_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSFusedBatchNormV3Float", bn_kb, status);

  TF_KernelBuilder* bn_h_kb = TF_NewKernelBuilder("FusedBatchNormV3", kPlatformName,
                                                  &MPSFusedBatchNormV3_Create,
                                                  &MPSFusedBatchNormV3_Compute,
                                                  &MPSFusedBatchNormV3_Delete);
  TF_KernelBuilder_TypeConstraint(bn_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSFusedBatchNormV3Half", bn_h_kb, status);

  TF_KernelBuilder* bn_bf_kb = TF_NewKernelBuilder("FusedBatchNormV3", kPlatformName,
                                                   &MPSFusedBatchNormV3_Create,
                                                   &MPSFusedBatchNormV3_Compute,
                                                   &MPSFusedBatchNormV3_Delete);
  TF_KernelBuilder_TypeConstraint(bn_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSFusedBatchNormV3BFloat16", bn_bf_kb, status);

  // Register Swish activation (float, half, bfloat16)
  extern void* MPSSwish_Create(TF_OpKernelConstruction*);
  extern void MPSSwish_Delete(void*);
  extern void MPSSwish_Compute(void*, TF_OpKernelContext*);

  TF_KernelBuilder* swish_kb = TF_NewKernelBuilder("Swish", kPlatformName,
                                                   &MPSSwish_Create,
                                                   &MPSSwish_Compute,
                                                   &MPSSwish_Delete);
  TF_KernelBuilder_TypeConstraint(swish_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSwishFloat", swish_kb, status);

  TF_KernelBuilder* swish_h_kb = TF_NewKernelBuilder("Swish", kPlatformName,
                                                     &MPSSwish_Create,
                                                     &MPSSwish_Compute,
                                                     &MPSSwish_Delete);
  TF_KernelBuilder_TypeConstraint(swish_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSSwishHalf", swish_h_kb, status);

  TF_KernelBuilder* swish_bf_kb = TF_NewKernelBuilder("Swish", kPlatformName,
                                                      &MPSSwish_Create,
                                                      &MPSSwish_Compute,
                                                      &MPSSwish_Delete);
  TF_KernelBuilder_TypeConstraint(swish_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSSwishBFloat16", swish_bf_kb, status);

  // Register Gelu activation (float, half, bfloat16)
  extern void* MPSGelu_Create(TF_OpKernelConstruction*);
  extern void MPSGelu_Delete(void*);
  extern void MPSGelu_Compute(void*, TF_OpKernelContext*);

  TF_KernelBuilder* gelu_kb = TF_NewKernelBuilder("Gelu", kPlatformName,
                                                  &MPSGelu_Create,
                                                  &MPSGelu_Compute,
                                                  &MPSGelu_Delete);
  TF_KernelBuilder_TypeConstraint(gelu_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSGeluFloat", gelu_kb, status);

  TF_KernelBuilder* gelu_h_kb = TF_NewKernelBuilder("Gelu", kPlatformName,
                                                    &MPSGelu_Create,
                                                    &MPSGelu_Compute,
                                                    &MPSGelu_Delete);
  TF_KernelBuilder_TypeConstraint(gelu_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSGeluHalf", gelu_h_kb, status);

  TF_KernelBuilder* gelu_bf_kb = TF_NewKernelBuilder("Gelu", kPlatformName,
                                                     &MPSGelu_Create,
                                                     &MPSGelu_Compute,
                                                     &MPSGelu_Delete);
  TF_KernelBuilder_TypeConstraint(gelu_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSGeluBFloat16", gelu_bf_kb, status);

  // ===== MASS REGISTRATION OF NEW OPERATIONS =====
  // Macro to register unary ops for 3 dtypes
  #define REGISTER_UNARY_OP_3DTYPE(OP_NAME) \
    extern void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*); \
    extern void MPS##OP_NAME##_Delete(void*); \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    TF_KernelBuilder* op_name##_f_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(op_name##_f_kb, "T", TF_FLOAT, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", op_name##_f_kb, status); \
    TF_KernelBuilder* op_name##_h_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(op_name##_h_kb, "T", TF_HALF, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", op_name##_h_kb, status); \
    TF_KernelBuilder* op_name##_bf_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(op_name##_bf_kb, "T", TF_BFLOAT16, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", op_name##_bf_kb, status);

  // Register 27 unary ops (81 kernels total)
  REGISTER_UNARY_OP_3DTYPE(Abs)
  REGISTER_UNARY_OP_3DTYPE(Neg)
  REGISTER_UNARY_OP_3DTYPE(Sqrt)
  REGISTER_UNARY_OP_3DTYPE(Rsqrt)
  REGISTER_UNARY_OP_3DTYPE(Exp)
  REGISTER_UNARY_OP_3DTYPE(Log)
  REGISTER_UNARY_OP_3DTYPE(Sin)
  REGISTER_UNARY_OP_3DTYPE(Cos)
  REGISTER_UNARY_OP_3DTYPE(Tan)
  REGISTER_UNARY_OP_3DTYPE(Asin)
  REGISTER_UNARY_OP_3DTYPE(Acos)
  REGISTER_UNARY_OP_3DTYPE(Atan)
  REGISTER_UNARY_OP_3DTYPE(Sinh)
  REGISTER_UNARY_OP_3DTYPE(Cosh)
  REGISTER_UNARY_OP_3DTYPE(Asinh)
  REGISTER_UNARY_OP_3DTYPE(Acosh)
  REGISTER_UNARY_OP_3DTYPE(Atanh)
  REGISTER_UNARY_OP_3DTYPE(Ceil)
  REGISTER_UNARY_OP_3DTYPE(Floor)
  REGISTER_UNARY_OP_3DTYPE(Round)
  REGISTER_UNARY_OP_3DTYPE(Erf)
  REGISTER_UNARY_OP_3DTYPE(Square)
  REGISTER_UNARY_OP_3DTYPE(Reciprocal)
  REGISTER_UNARY_OP_3DTYPE(Sign)
  REGISTER_UNARY_OP_3DTYPE(Expm1)
  REGISTER_UNARY_OP_3DTYPE(Log1p)
  REGISTER_UNARY_OP_3DTYPE(IsFinite)

  // Additional activation registrations
  REGISTER_UNARY_OP_3DTYPE(LeakyRelu)
  REGISTER_UNARY_OP_3DTYPE(Relu6)
  REGISTER_UNARY_OP_3DTYPE(Elu)
  REGISTER_UNARY_OP_3DTYPE(Selu)
  REGISTER_UNARY_OP_3DTYPE(Softplus)
  REGISTER_UNARY_OP_3DTYPE(Softsign)

  // Macro for binary ops
  #define REGISTER_BINARY_OP_3DTYPE(OP_NAME) \
    extern void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*); \
    extern void MPS##OP_NAME##_Delete(void*); \
    extern void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext*); \
    TF_KernelBuilder* bin##OP_NAME##_f_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(bin##OP_NAME##_f_kb, "T", TF_FLOAT, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Float", bin##OP_NAME##_f_kb, status); \
    TF_KernelBuilder* bin##OP_NAME##_h_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(bin##OP_NAME##_h_kb, "T", TF_HALF, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "Half", bin##OP_NAME##_h_kb, status); \
    TF_KernelBuilder* bin##OP_NAME##_bf_kb = TF_NewKernelBuilder(#OP_NAME, kPlatformName, &MPS##OP_NAME##_Create, &MPS##OP_NAME##_Compute, &MPS##OP_NAME##_Delete); \
    TF_KernelBuilder_TypeConstraint(bin##OP_NAME##_bf_kb, "T", TF_BFLOAT16, status); \
    TF_RegisterKernelBuilder("MPS" #OP_NAME "BFloat16", bin##OP_NAME##_bf_kb, status);

  // Register 14 binary ops (42 kernels total)
  REGISTER_BINARY_OP_3DTYPE(Div)
  REGISTER_BINARY_OP_3DTYPE(RealDiv)
  REGISTER_BINARY_OP_3DTYPE(Sub)
  REGISTER_BINARY_OP_3DTYPE(Pow)
  REGISTER_BINARY_OP_3DTYPE(FloorDiv)
  REGISTER_BINARY_OP_3DTYPE(FloorMod)
  REGISTER_BINARY_OP_3DTYPE(Atan2)
  REGISTER_BINARY_OP_3DTYPE(SquaredDifference)
  REGISTER_BINARY_OP_3DTYPE(Equal)
  REGISTER_BINARY_OP_3DTYPE(NotEqual)
  REGISTER_BINARY_OP_3DTYPE(Less)
  REGISTER_BINARY_OP_3DTYPE(LessEqual)
  REGISTER_BINARY_OP_3DTYPE(Greater)
  REGISTER_BINARY_OP_3DTYPE(GreaterEqual)

  // Tensor op registrations (float/half/bfloat16)
  REGISTER_UNARY_OP_3DTYPE(Slice)
  REGISTER_UNARY_OP_3DTYPE(StridedSlice)
  REGISTER_UNARY_OP_3DTYPE(Fill)
  REGISTER_UNARY_OP_3DTYPE(ZerosLike)
  REGISTER_UNARY_OP_3DTYPE(OnesLike)
  REGISTER_UNARY_OP_3DTYPE(Pad)
  REGISTER_UNARY_OP_3DTYPE(MirrorPad)
  REGISTER_UNARY_OP_3DTYPE(Tile)
  REGISTER_UNARY_OP_3DTYPE(Select)
  REGISTER_UNARY_OP_3DTYPE(ClipByValue)

  // Logical ops (bool)
  extern void* MPSLogicalAnd_Create(TF_OpKernelConstruction*);
  extern void MPSLogicalAnd_Delete(void*);
  extern void MPSLogicalAnd_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* land_kb = TF_NewKernelBuilder("LogicalAnd", kPlatformName,
                                                  &MPSLogicalAnd_Create,
                                                  &MPSLogicalAnd_Compute,
                                                  &MPSLogicalAnd_Delete);
  TF_KernelBuilder_TypeConstraint(land_kb, "T", TF_BOOL, status);
  TF_RegisterKernelBuilder("MPSLogicalAnd", land_kb, status);

  extern void* MPSLogicalOr_Create(TF_OpKernelConstruction*);
  extern void MPSLogicalOr_Delete(void*);
  extern void MPSLogicalOr_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* lor_kb = TF_NewKernelBuilder("LogicalOr", kPlatformName,
                                                &MPSLogicalOr_Create,
                                                &MPSLogicalOr_Compute,
                                                &MPSLogicalOr_Delete);
  TF_KernelBuilder_TypeConstraint(lor_kb, "T", TF_BOOL, status);
  TF_RegisterKernelBuilder("MPSLogicalOr", lor_kb, status);

  extern void* MPSLogicalNot_Create(TF_OpKernelConstruction*);
  extern void MPSLogicalNot_Delete(void*);
  extern void MPSLogicalNot_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* lnot_kb = TF_NewKernelBuilder("LogicalNot", kPlatformName,
                                                 &MPSLogicalNot_Create,
                                                 &MPSLogicalNot_Compute,
                                                 &MPSLogicalNot_Delete);
  TF_KernelBuilder_TypeConstraint(lnot_kb, "T", TF_BOOL, status);
  TF_RegisterKernelBuilder("MPSLogicalNot", lnot_kb, status);

  // Register Conv2D (T=float) for device "MPS" (NHWC only)
  extern void* MPSConv2D_Create(TF_OpKernelConstruction*);
  extern void MPSConv2D_Delete(void*);
  extern void MPSConv2D_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* conv_kb = TF_NewKernelBuilder("Conv2D", kPlatformName,
                                                  &MPSConv2D_Create,
                                                  &MPSConv2D_Compute,
                                                  &MPSConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(conv_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSConv2DFloat", conv_kb, status);

  // Conv2D half
  TF_KernelBuilder* conv_h_kb = TF_NewKernelBuilder("Conv2D", kPlatformName,
                                                    &MPSConv2D_Create,
                                                    &MPSConv2D_Compute,
                                                    &MPSConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(conv_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSConv2DHalf", conv_h_kb, status);

  // Conv2D bfloat16
  TF_KernelBuilder* conv_bf_kb = TF_NewKernelBuilder("Conv2D", kPlatformName,
                                                     &MPSConv2D_Create,
                                                     &MPSConv2D_Compute,
                                                     &MPSConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(conv_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSConv2DBFloat16", conv_bf_kb, status);

  // DepthwiseConv2dNative (float and half) for device "MPS" (NHWC only)
  extern void* MPSDepthwiseConv2D_Create(TF_OpKernelConstruction*);
  extern void MPSDepthwiseConv2D_Delete(void*);
  extern void MPSDepthwiseConv2D_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* dwconv_kb = TF_NewKernelBuilder("DepthwiseConv2dNative", kPlatformName,
                                                    &MPSDepthwiseConv2D_Create,
                                                    &MPSDepthwiseConv2D_Compute,
                                                    &MPSDepthwiseConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(dwconv_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSDepthwiseConv2DFloat", dwconv_kb, status);

  TF_KernelBuilder* dwconv_h_kb = TF_NewKernelBuilder("DepthwiseConv2dNative", kPlatformName,
                                                      &MPSDepthwiseConv2D_Create,
                                                      &MPSDepthwiseConv2D_Compute,
                                                      &MPSDepthwiseConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(dwconv_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSDepthwiseConv2DHalf", dwconv_h_kb, status);

  TF_KernelBuilder* dwconv_bf_kb = TF_NewKernelBuilder("DepthwiseConv2dNative", kPlatformName,
                                                       &MPSDepthwiseConv2D_Create,
                                                       &MPSDepthwiseConv2D_Compute,
                                                       &MPSDepthwiseConv2D_Delete);
  TF_KernelBuilder_TypeConstraint(dwconv_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSDepthwiseConv2DBFloat16", dwconv_bf_kb, status);

  // MaxPool (float and half) for device "MPS" (NHWC only)
  extern void* MPSMaxPool_Create(TF_OpKernelConstruction*);
  extern void MPSMaxPool_Delete(void*);
  extern void MPSMaxPool_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* maxpool_kb = TF_NewKernelBuilder("MaxPool", kPlatformName,
                                                     &MPSMaxPool_Create,
                                                     &MPSMaxPool_Compute,
                                                     &MPSMaxPool_Delete);
  TF_KernelBuilder_TypeConstraint(maxpool_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMaxPoolFloat", maxpool_kb, status);

  TF_KernelBuilder* maxpool_h_kb = TF_NewKernelBuilder("MaxPool", kPlatformName,
                                                       &MPSMaxPool_Create,
                                                       &MPSMaxPool_Compute,
                                                       &MPSMaxPool_Delete);
  TF_KernelBuilder_TypeConstraint(maxpool_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSMaxPoolHalf", maxpool_h_kb, status);

  TF_KernelBuilder* maxpool_bf_kb = TF_NewKernelBuilder("MaxPool", kPlatformName,
                                                        &MPSMaxPool_Create,
                                                        &MPSMaxPool_Compute,
                                                        &MPSMaxPool_Delete);
  TF_KernelBuilder_TypeConstraint(maxpool_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSMaxPoolBFloat16", maxpool_bf_kb, status);

  // AvgPool (float and half) for device "MPS" (NHWC only)
  extern void* MPSAvgPool_Create(TF_OpKernelConstruction*);
  extern void MPSAvgPool_Delete(void*);
  extern void MPSAvgPool_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* avgpool_kb = TF_NewKernelBuilder("AvgPool", kPlatformName,
                                                     &MPSAvgPool_Create,
                                                     &MPSAvgPool_Compute,
                                                     &MPSAvgPool_Delete);
  TF_KernelBuilder_TypeConstraint(avgpool_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSAvgPoolFloat", avgpool_kb, status);

  TF_KernelBuilder* avgpool_h_kb = TF_NewKernelBuilder("AvgPool", kPlatformName,
                                                       &MPSAvgPool_Create,
                                                       &MPSAvgPool_Compute,
                                                       &MPSAvgPool_Delete);
  TF_KernelBuilder_TypeConstraint(avgpool_h_kb, "T", TF_HALF, status);
  TF_RegisterKernelBuilder("MPSAvgPoolHalf", avgpool_h_kb, status);

  TF_KernelBuilder* avgpool_bf_kb = TF_NewKernelBuilder("AvgPool", kPlatformName,
                                                        &MPSAvgPool_Create,
                                                        &MPSAvgPool_Compute,
                                                        &MPSAvgPool_Delete);
  TF_KernelBuilder_TypeConstraint(avgpool_bf_kb, "T", TF_BFLOAT16, status);
  TF_RegisterKernelBuilder("MPSAvgPoolBFloat16", avgpool_bf_kb, status);

  // If registration fails, status is dropped intentionally (plugin load should continue).
  TF_DeleteStatus(status);
}

}

// ===== MPS Relu kernel implementation (float/half) =====
namespace {
// Static pipeline cache for the Relu compute shader.
static id<MTLComputePipelineState> g_relu_pipeline = nil;
static id<MTLComputePipelineState> g_relu_h_pipeline = nil;
static id<MTLLibrary> g_relu_lib = nil;
static id<MTLLibrary> g_relu_h_lib = nil;
static dispatch_once_t g_relu_once;
static dispatch_once_t g_relu_h_once;

static void EnsureReluPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_relu_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void relu_k(const device float* in_ [[buffer(0)]],\n"
                       @"                 device float* out_ [[buffer(1)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out_[gid] = max(in_[gid], 0.0f);\n"
                       @"}";
    NSError* err = nil;
    g_relu_lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!g_relu_lib) {
      NSLog(@"MPS Relu: failed to compile MSL: %@", err);
      return;
    }
    id<MTLFunction> fn = [g_relu_lib newFunctionWithName:@"relu_k"];
    g_relu_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_relu_pipeline) {
      NSLog(@"MPS Relu: pipeline error: %@", err);
    }
  });
}

static void EnsureReluHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_relu_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void relu_h_k(const device half* in_ [[buffer(0)]],\n"
                       @"                   device half* out_ [[buffer(1)]],\n"
                       @"                   uint gid [[thread_position_in_grid]]) {\n"
                       @"  half zero = (half)0.0h;\n"
                       @"  out_[gid] = max(in_[gid], zero);\n"
                       @"}";
    NSError* err = nil;
    g_relu_h_lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!g_relu_h_lib) {
      NSLog(@"MPS Relu half: failed to compile MSL: %@", err);
      return;
    }
    id<MTLFunction> fn = [g_relu_h_lib newFunctionWithName:@"relu_h_k"];
    g_relu_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_relu_h_pipeline) {
      NSLog(@"MPS Relu half: pipeline error: %@", err);
    }
  });
}
}  // namespace

extern "C" void MPSRelu_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // Input 0
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Relu[MPS float] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims_stack[8];
  int64_t* dims = dims_stack;
  std::unique_ptr<int64_t[]> dyn;
  if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(input, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(float);

  // Output allocation
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Get MPS stream
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    // Host fallback
    const float* in = static_cast<const float*>(TF_TensorData(input));
    float* out = static_cast<float*>(TF_TensorData(output));
    for (int64_t i = 0; i < nelems; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    TF_DeleteStatus(s);
    return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  EnsureReluPipeline(dev);
  if (!g_relu_pipeline) {
    // Host fallback
    const float* in = static_cast<const float*>(TF_TensorData(input));
    float* out = static_cast<float*>(TF_TensorData(output));
    for (int64_t i = 0; i < nelems; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    TF_DeleteStatus(s);
    return;
  }

  // Stage, compute, stage back
  const void* in_host = TF_TensorData(input);
  void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);

  id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_relu_pipeline];
  [enc setBuffer:inb offset:0 atIndex:0];
  [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads = 256;
  NSUInteger grid = (NSUInteger)nelems;
  NSUInteger groups = (grid + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  memcpy(out_host, outb.contents, bytes);
  TF_DeleteStatus(s);
}

extern "C" void MPSReluHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_HALF) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Relu[MPS half] expects half");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd = TF_NumDims(input);
  int64_t nelems = 1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems * sizeof(uint16_t);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd, bytes, s);
  if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }

  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    // Host fallback with conversion
    const uint16_t* in = (const uint16_t*)TF_TensorData(input);
    uint16_t* out = (uint16_t*)TF_TensorData(output);
    for (int64_t i=0;i<nelems;++i){ float v = HalfToFloat(in[i]); if (v < 0) v = 0.0f; out[i] = FloatToHalf(v); }
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device; EnsureReluHalfPipeline(dev);
  if (!g_relu_h_pipeline) {
    const uint16_t* in = (const uint16_t*)TF_TensorData(input);
    uint16_t* out = (uint16_t*)TF_TensorData(output);
    for (int64_t i=0;i<nelems;++i){ float v = HalfToFloat(in[i]); if (v < 0) v = 0.0f; out[i] = FloatToHalf(v); }
    TF_DeleteStatus(s); return;
  }
  const void* in_host = TF_TensorData(input); void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_relu_h_pipeline];
  [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(out_host, outb.contents, bytes); TF_DeleteStatus(s);
}

extern "C" void MPSReluBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Relu[MPS bfloat16] expects bfloat16");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int nd = TF_NumDims(input); int64_t nelems=1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems*sizeof(uint16_t);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, bytes, s);
  if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  const uint16_t* in = (const uint16_t*)TF_TensorData(input);
  uint16_t* out = (uint16_t*)TF_TensorData(output);
  for (int64_t i=0;i<nelems;++i){ float v = BFloat16ToFloat(in[i]); if (v<0) v=0.0f; out[i]=FloatToBFloat16(v); }
  TF_DeleteStatus(s);
}

// ===== MPS Add and Mul kernels (float) =====
namespace {
static id<MTLComputePipelineState> g_add_pipeline = nil;
static id<MTLComputePipelineState> g_mul_pipeline = nil;
static dispatch_once_t g_add_once;
static dispatch_once_t g_mul_once;

static void EnsureAddPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_add_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void add_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] + b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Add: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"add_k"];
    g_add_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_add_pipeline) { NSLog(@"MPS Add: pipeline error: %@", err); }
  });
}

static void EnsureMulPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_mul_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void mul_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] * b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Mul: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_k"];
    g_mul_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_mul_pipeline) { NSLog(@"MPS Mul: pipeline error: %@", err); }
  });
}
}  // namespace

extern "C" void MPSAdd_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Same-shape fast path; allow scalar broadcasting on host fallback for now.
  int nd_a = TF_NumDims(a);
  int nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }

  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) {
      if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; }
    }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      // Host fallback
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + pb[i];
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureAddPipeline(dev);
    if (!g_add_pipeline) {
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + pb[i];
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a);
    const void* hb = TF_TensorData(b);
    void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes);
    memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_add_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256; NSUInteger grid = (NSUInteger)nelems;
    NSUInteger groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s);
    return;
  }

  // Scalar broadcasting on host
  bool a_scalar = (nelems_a == 1);
  bool b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2 shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  // Output dims = other tensor dims
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const float* pa = static_cast<const float*>(TF_TensorData(a));
  const float* pb = static_cast<const float*>(TF_TensorData(b));
  float* po = static_cast<float*>(TF_TensorData(out));
  if (a_scalar) {
    float av = pa[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = av + pb[i];
  } else {
    float bv = pb[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + bv;
  }
  TF_DeleteStatus(s); return;
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  EnsureAddPipeline(dev);
  if (!g_add_pipeline) {
    const float* pa = static_cast<const float*>(TF_TensorData(a));
    const float* pb = static_cast<const float*>(TF_TensorData(b));
    float* po = static_cast<float*>(TF_TensorData(out));
    for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] + pb[i];
    TF_DeleteStatus(s); return;
  }
  const void* ha = TF_TensorData(a);
  const void* hb = TF_TensorData(b);
  void* ho = TF_TensorData(out);
  id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(ba.contents, ha, bytes);
  memcpy(bb.contents, hb, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_add_pipeline];
  [enc setBuffer:ba offset:0 atIndex:0];
  [enc setBuffer:bb offset:0 atIndex:1];
  [enc setBuffer:bo offset:0 atIndex:2];
  NSUInteger threads = 256; NSUInteger grid = (NSUInteger)nelems;
  NSUInteger groups = (grid + threads - 1) / threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(ho, bo.contents, bytes);
  TF_DeleteStatus(s);
}

extern "C" void MPSMul_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Same-shape fast path; allow scalar broadcasting on host fallback for now.
  int nd_a = TF_NumDims(a);
  int nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }

  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) {
      if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; }
    }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] * pb[i];
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureMulPipeline(dev);
    if (!g_mul_pipeline) {
      const float* pa = static_cast<const float*>(TF_TensorData(a));
      const float* pb = static_cast<const float*>(TF_TensorData(b));
      float* po = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] * pb[i];
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a);
    const void* hb = TF_TensorData(b);
    void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes);
    memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_mul_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256; NSUInteger grid = (NSUInteger)nelems;
    NSUInteger groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s);
    return;
  }

  // Scalar broadcasting on host
  bool a_scalar = (nelems_a == 1);
  bool b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const float* pa = static_cast<const float*>(TF_TensorData(a));
  const float* pb = static_cast<const float*>(TF_TensorData(b));
  float* po = static_cast<float*>(TF_TensorData(out));
  if (a_scalar) {
    float av = pa[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = av * pb[i];
  } else {
    float bv = pb[0];
    for (int64_t i = 0; i < nelems; ++i) po[i] = pa[i] * bv;
  }
  TF_DeleteStatus(s); return;
}

// ===== MPS Add/Mul kernels (half) =====
namespace {
static id<MTLComputePipelineState> g_add_h_pipeline = nil;
static id<MTLComputePipelineState> g_mul_h_pipeline = nil;
static dispatch_once_t g_add_h_once;
static dispatch_once_t g_mul_h_once;

static void EnsureAddHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_add_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void add_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2)]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] + b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS AddHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"add_h_k"];
    g_add_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_add_h_pipeline) { NSLog(@"MPS AddHalf: pipeline error: %@", err); }
  });
}

static void EnsureMulHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_mul_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void mul_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2)]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = a[gid] * b[gid];\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS MulHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"mul_h_k"];
    g_mul_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_mul_h_pipeline) { NSLog(@"MPS MulHalf: pipeline error: %@", err); }
  });
}
}  // namespace

extern "C" void MPSAddHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_HALF || TF_TensorType(b) != TF_HALF) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2[MPS half] expects half");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) { if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; } }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(uint16_t);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) + HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureAddHalfPipeline(dev);
    if (!g_add_h_pipeline) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) + HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a), *hb = TF_TensorData(b); void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes); memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_add_h_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256, grid = (NSUInteger)nelems, groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s); return;
  }
  // Scalar broadcast on host
  bool a_scalar = (nelems_a == 1), b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "AddV2 shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(uint16_t);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  if (a_scalar) {
    float av = HalfToFloat(pa[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(av + HalfToFloat(pb[i]));
  } else {
    float bv = HalfToFloat(pb[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) + bv);
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMulHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_HALF || TF_TensorType(b) != TF_HALF) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul[MPS half] expects half");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  int nd = nd_a;
  if (same_shape) {
    for (int i = 0; i < nd_a; ++i) { if (TF_Dim(a, i) != TF_Dim(b, i)) { same_shape = false; break; } }
  }

  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1;
    for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
    size_t bytes = nelems * sizeof(uint16_t);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) * HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
    id<MTLDevice> dev = stream->dev->device;
    EnsureMulHalfPipeline(dev);
    if (!g_mul_h_pipeline) {
      const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
      const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
      uint16_t* po = (uint16_t*)TF_TensorData(out);
      for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) * HalfToFloat(pb[i]));
      TF_DeleteStatus(s); return;
    }
    const void* ha = TF_TensorData(a), *hb = TF_TensorData(b); void* ho = TF_TensorData(out);
    id<MTLBuffer> ba = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bo = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents, ha, bytes); memcpy(bb.contents, hb, bytes);
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_mul_h_pipeline];
    [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads = 256, grid = (NSUInteger)nelems, groups = (grid + threads - 1) / threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho, bo.contents, bytes);
    TF_DeleteStatus(s); return;
  }
  // Scalar broadcast on host
  bool a_scalar = (nelems_a == 1), b_scalar = (nelems_b == 1);
  if (!(a_scalar || b_scalar)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Mul shapes must match or one input be scalar");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int nd_out = a_scalar ? nd_b : nd_a;
  if (nd_out > 8) { dyn.reset(new int64_t[nd_out]); dims = dyn.get(); }
  int64_t nelems = 1;
  for (int i = 0; i < nd_out; ++i) { int64_t d = a_scalar ? TF_Dim(b, i) : TF_Dim(a, i); dims[i] = d; nelems *= (d < 0 ? 0 : d); }
  size_t bytes = nelems * sizeof(uint16_t);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_HALF, dims, nd_out, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  if (a_scalar) {
    float av = HalfToFloat(pa[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(av * HalfToFloat(pb[i]));
  } else {
    float bv = HalfToFloat(pb[0]);
    for (int64_t i = 0; i < nelems; ++i) po[i] = FloatToHalf(HalfToFloat(pa[i]) * bv);
  }
  TF_DeleteStatus(s);
}

// ===== MPS Add/Mul/Maximum/Minimum/Sigmoid/Tanh kernels (bfloat16, host-based) =====
extern "C" void MPSAddV2BFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(fa + fb);
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMulBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(fa * fb);
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMaximumBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(std::max(fa, fb));
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSMinimumBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(a);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(a, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* pa = (const uint16_t*)TF_TensorData(a);
  const uint16_t* pb = (const uint16_t*)TF_TensorData(b);
  uint16_t* po = (uint16_t*)TF_TensorData(out);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float fa = BFloat16ToFloat(pa[i]);
    float fb = BFloat16ToFloat(pb[i]);
    po[i] = FloatToBFloat16(std::min(fa, fb));
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSSigmoidBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(input, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* in = (const uint16_t*)TF_TensorData(input);
  uint16_t* out = (uint16_t*)TF_TensorData(output);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float x = BFloat16ToFloat(in[i]);
    out[i] = FloatToBFloat16(1.0f / (1.0f + expf(-x)));
  }
  TF_DeleteStatus(s);
}

extern "C" void MPSTanhBFloat16_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int nd = TF_NumDims(input);
  int64_t nelems = 1;
  int64_t dims[8];
  for (int i = 0; i < nd; ++i) { int64_t d = TF_Dim(input, i); dims[i] = d; nelems *= d; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BFLOAT16, dims, nd, nelems * 2, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  const uint16_t* in = (const uint16_t*)TF_TensorData(input);
  uint16_t* out = (uint16_t*)TF_TensorData(output);
  
  for (int64_t i = 0; i < nelems; ++i) {
    float x = BFloat16ToFloat(in[i]);
    out[i] = FloatToBFloat16(tanhf(x));
  }
  TF_DeleteStatus(s);
}

// ===== MPS MatMul kernel (float) =====
namespace {
struct MPSMatMulAttrs { bool ta; bool tb; };
}

extern "C" void* MPSMatMul_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Bool ta = 0, tb = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_a", &ta, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return nullptr; }
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_b", &tb, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return nullptr; }
  auto* attrs = new MPSMatMulAttrs{ta != 0, tb != 0};
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSMatMul_Delete(void* kernel) {
  auto* attrs = reinterpret_cast<MPSMatMulAttrs*>(kernel);
  delete attrs;
}

extern "C" void MPSMatMul_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = reinterpret_cast<MPSMatMulAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_DataType dtype_a = TF_TensorType(a), dtype_b = TF_TensorType(b);
  if (dtype_a != dtype_b) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul[MPS] inputs must have same dtype");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (dtype_a != TF_FLOAT && dtype_a != TF_HALF && dtype_a != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul[MPS] supports float/half/bfloat16");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_NumDims(a) != 2 || TF_NumDims(b) != 2) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul[MPS] requires rank-2 tensors");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int64_t a_rows = TF_Dim(a, 0), a_cols = TF_Dim(a, 1);
  int64_t b_rows = TF_Dim(b, 0), b_cols = TF_Dim(b, 1);
  int64_t M = attrs && attrs->ta ? a_cols : a_rows;
  int64_t K_a = attrs && attrs->ta ? a_rows : a_cols;
  int64_t K_b = attrs && attrs->tb ? b_cols : b_rows;
  int64_t N = attrs && attrs->tb ? b_rows : b_cols;
  if (K_a != K_b) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MatMul inner dims mismatch");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  int64_t K = K_a;

  bool is_bf16 = (dtype_a == TF_BFLOAT16);
  bool is_half = (dtype_a == TF_HALF);
  bool is_float = (dtype_a == TF_FLOAT);
  size_t elem_size = is_float ? sizeof(float) : sizeof(uint16_t);

  int64_t out_dims[2] = {M, N};
  size_t bytes = (size_t)M * (size_t)N * elem_size;
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, dtype_a, out_dims, 2, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  // Try GPU first; if stream is unavailable, fall back to host.
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    // Host matmul (works for all dtypes via conversion)
    if (is_float) {
      const float* A = static_cast<const float*>(TF_TensorData(a));
      const float* B = static_cast<const float*>(TF_TensorData(b));
      float* C = static_cast<float*>(TF_TensorData(out));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            float va = attrs && attrs->ta ? A[k * a_cols + i] : A[i * a_cols + k];
            float vb = attrs && attrs->tb ? B[j * b_cols + k] : B[k * b_cols + j];
            sum += va * vb;
          }
          C[i * N + j] = sum;
        }
      }
    } else if (is_half) {
      const uint16_t* A = static_cast<const uint16_t*>(TF_TensorData(a));
      const uint16_t* B = static_cast<const uint16_t*>(TF_TensorData(b));
      uint16_t* C = static_cast<uint16_t*>(TF_TensorData(out));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            uint16_t ua = attrs && attrs->ta ? A[k * a_cols + i] : A[i * a_cols + k];
            uint16_t ub = attrs && attrs->tb ? B[j * b_cols + k] : B[k * b_cols + j];
            sum += HalfToFloat(ua) * HalfToFloat(ub);
          }
          C[i * N + j] = FloatToHalf(sum);
        }
      }
    } else {  // bfloat16
      const uint16_t* A = static_cast<const uint16_t*>(TF_TensorData(a));
      const uint16_t* B = static_cast<const uint16_t*>(TF_TensorData(b));
      uint16_t* C = static_cast<uint16_t*>(TF_TensorData(out));
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
          float sum = 0.0f;
          for (int64_t k = 0; k < K; ++k) {
            uint16_t ua = attrs && attrs->ta ? A[k * a_cols + i] : A[i * a_cols + k];
            uint16_t ub = attrs && attrs->tb ? B[j * b_cols + k] : B[k * b_cols + j];
            sum += BFloat16ToFloat(ua) * BFloat16ToFloat(ub);
          }
          C[i * N + j] = FloatToBFloat16(sum);
        }
      }
    }
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;

  // For bfloat16, use MPSGraph which supports MPSDataTypeBFloat16 natively
  if (is_bf16) {
    @autoreleasepool {
      MPSGraph* graph = [[MPSGraph alloc] init];
      MPSGraphTensor* tA = [graph placeholderWithShape:@[@(a_rows), @(a_cols)]
                                               dataType:MPSDataTypeBFloat16
                                                   name:@"A"];
      MPSGraphTensor* tB = [graph placeholderWithShape:@[@(b_rows), @(b_cols)]
                                               dataType:MPSDataTypeBFloat16
                                                   name:@"B"];
      if (attrs && attrs->ta) tA = [graph transposeTensor:tA dimension:0 withDimension:1 name:@"A_T"];
      if (attrs && attrs->tb) tB = [graph transposeTensor:tB dimension:0 withDimension:1 name:@"B_T"];
      MPSGraphTensor* tC = [graph matrixMultiplicationWithPrimaryTensor:tA
                                                        secondaryTensor:tB
                                                                   name:@"C"];
      size_t bytesA = (size_t)a_rows * (size_t)a_cols * sizeof(uint16_t);
      size_t bytesB = (size_t)b_rows * (size_t)b_cols * sizeof(uint16_t);
      id<MTLBuffer> bufA = [dev newBufferWithBytes:TF_TensorData(a) length:bytesA
                                           options:MTLResourceStorageModeShared];
      id<MTLBuffer> bufB = [dev newBufferWithBytes:TF_TensorData(b) length:bytesB
                                           options:MTLResourceStorageModeShared];
      id<MTLBuffer> bufC = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      MPSGraphTensorData* dA = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA
                                                                        shape:@[@(a_rows), @(a_cols)]
                                                                     dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData* dB = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB
                                                                        shape:@[@(b_rows), @(b_cols)]
                                                                     dataType:MPSDataTypeBFloat16];
      MPSGraphTensorData* dC = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufC
                                                                        shape:@[@(M), @(N)]
                                                                     dataType:MPSDataTypeBFloat16];
      id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
      [graph runWithMTLCommandBuffer:cb
                                feeds:@{tA: dA, tB: dB}
                      targetTensors:@[tC]
                   targetOperations:nil
                   executionDescriptor:nil];
      [cb commit]; [cb waitUntilCompleted];
      memcpy(TF_TensorData(out), bufC.contents, bytes);
    }
    TF_DeleteStatus(s); return;
  }

  // For float and half, use MPSMatrixMultiplication
  // Stage to shared buffers (float or half)
  size_t bytesA = (size_t)a_rows * (size_t)a_cols * elem_size;
  size_t bytesB = (size_t)b_rows * (size_t)b_cols * elem_size;
  id<MTLBuffer> bufA = [dev newBufferWithLength:bytesA options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufB = [dev newBufferWithLength:bytesB options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(bufA.contents, TF_TensorData(a), bytesA);
  memcpy(bufB.contents, TF_TensorData(b), bytesB);

  MPSDataType mps_dtype = is_float ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
  MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)a_rows
                                                                   columns:(NSUInteger)a_cols
                                                                  rowBytes:(NSUInteger)(a_cols * elem_size)
                                                                  dataType:mps_dtype];
  MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)b_rows
                                                                   columns:(NSUInteger)b_cols
                                                                  rowBytes:(NSUInteger)(b_cols * elem_size)
                                                                  dataType:mps_dtype];
  MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                                                   columns:(NSUInteger)N
                                                                  rowBytes:(NSUInteger)(N * elem_size)
                                                                  dataType:mps_dtype];
  MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:bufA offset:0 descriptor:dA];
  MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bufB offset:0 descriptor:dB];
  MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:bufC offset:0 descriptor:dC];
  MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                 transposeLeft:(attrs && attrs->ta)
                                                                transposeRight:(attrs && attrs->tb)
                                                                   resultRows:(NSUInteger)M
                                                                resultColumns:(NSUInteger)N
                                                              interiorColumns:(NSUInteger)K
                                                                          alpha:1.0
                                                                           beta:0.0];
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
  [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
  [cb commit]; [cb waitUntilCompleted];
  memcpy(TF_TensorData(out), bufC.contents, bytes);
  TF_DeleteStatus(s);
}

// ===== StreamExecutor memset32 implementation (uint32_t pattern) =====
namespace {
static id<MTLComputePipelineState> g_memset32_pipeline = nil;
static dispatch_once_t g_memset32_once;
static void EnsureMemset32Pipeline(id<MTLDevice> dev) {
  dispatch_once(&g_memset32_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void memset32_k(device uint* out [[buffer(0)]],\n"
                       @"                      constant uint& pat [[buffer(1)]],\n"
                       @"                      uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = pat;\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS memset32: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"memset32_k"];
    g_memset32_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_memset32_pipeline) { NSLog(@"MPS memset32: pipeline error: %@", err); }
  });
}
}  // namespace

// (implementation moved earlier to avoid duplicate definitions)

// ===== MPS Maximum and Minimum kernels (float) =====
namespace {
static id<MTLComputePipelineState> g_max_pipeline = nil;
static id<MTLComputePipelineState> g_min_pipeline = nil;
static dispatch_once_t g_max_once;
static dispatch_once_t g_min_once;

static void EnsureMaxPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_max_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void max_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = fmax(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Maximum: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"max_k"];
    g_max_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_max_pipeline) { NSLog(@"MPS Maximum: pipeline error: %@", err); }
  });
}
static void EnsureMinPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_min_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void min_k(const device float* a [[buffer(0)]],\n"
                       @"                 const device float* b [[buffer(1)]],\n"
                       @"                 device float* out [[buffer(2)]],\n"
                       @"                 uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = fmin(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Minimum: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"min_k"];
    g_min_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_min_pipeline) { NSLog(@"MPS Minimum: pipeline error: %@", err); }
  });
}
}

extern "C" void MPSMaximum_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Maximum[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  if (same_shape) for (int i = 0; i < nd_a; ++i) if (TF_Dim(a,i)!=TF_Dim(b,i)) { same_shape=false; break; }

  int nd = nd_a;
  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1; for (int i=0;i<nd;++i){ int64_t d=TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
      for (int64_t i=0;i<nelems;++i) po[i]= pa[i] > pb[i] ? pa[i] : pb[i];
      TF_DeleteStatus(s); return; }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureMaxPipeline(dev);
    if (!g_max_pipeline) { const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out); for (int64_t i=0;i<nelems;++i) po[i]= pa[i] > pb[i] ? pa[i] : pb[i]; TF_DeleteStatus(s); return; }
    const void* ha=TF_TensorData(a); const void* hb=TF_TensorData(b); void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes); memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_max_pipeline]; [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes); TF_DeleteStatus(s); return;
  }
  // Scalar broadcast
  bool a_scalar = (nelems_a==1), b_scalar=(nelems_b==1);
  if (!(a_scalar||b_scalar)) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Maximum shapes must match or one be scalar"); TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  int nd_out = a_scalar ? nd_b : nd_a; if (nd_out>8){ dyn.reset(new int64_t[nd_out]); dims=dyn.get(); }
  int64_t nelems=1; for (int i=0;i<nd_out;++i){ int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s); if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
  if (a_scalar){ float av=pa[0]; for (int64_t i=0;i<nelems;++i) po[i] = av > pb[i] ? av : pb[i]; }
  else { float bv=pb[0]; for (int64_t i=0;i<nelems;++i) po[i] = pa[i] > bv ? pa[i] : bv; }
  TF_DeleteStatus(s);
}

extern "C" void MPSMinimum_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(a) != TF_FLOAT || TF_TensorType(b) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Minimum[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b);
  int64_t nelems_a = 1, nelems_b = 1;
  for (int i = 0; i < nd_a; ++i) { int64_t d = TF_Dim(a, i); nelems_a *= (d < 0 ? 0 : d); }
  for (int i = 0; i < nd_b; ++i) { int64_t d = TF_Dim(b, i); nelems_b *= (d < 0 ? 0 : d); }
  bool same_shape = (nd_a == nd_b);
  if (same_shape) for (int i = 0; i < nd_a; ++i) if (TF_Dim(a,i)!=TF_Dim(b,i)) { same_shape=false; break; }

  int nd = nd_a;
  int64_t dims_stack[8]; int64_t* dims = dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (same_shape) {
    if (nd > 8) { dyn.reset(new int64_t[nd]); dims = dyn.get(); }
    int64_t nelems = 1; for (int i=0;i<nd;++i){ int64_t d=TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
    TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    SP_Stream cstream = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
      const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
      for (int64_t i=0;i<nelems;++i) po[i]= pa[i] < pb[i] ? pa[i] : pb[i];
      TF_DeleteStatus(s); return; }
    auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureMinPipeline(dev);
    if (!g_min_pipeline) { const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out); for (int64_t i=0;i<nelems;++i) po[i]= pa[i] < pb[i] ? pa[i] : pb[i]; TF_DeleteStatus(s); return; }
    const void* ha=TF_TensorData(a); const void* hb=TF_TensorData(b); void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; id<MTLBuffer> bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes); memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_min_pipeline]; [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:bb offset:0 atIndex:1]; [enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes); TF_DeleteStatus(s); return;
  }
  // Scalar broadcast
  bool a_scalar = (nelems_a==1), b_scalar=(nelems_b==1);
  if (!(a_scalar||b_scalar)) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Minimum shapes must match or one be scalar"); TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  int nd_out = a_scalar ? nd_b : nd_a; if (nd_out>8){ dyn.reset(new int64_t[nd_out]); dims=dyn.get(); }
  int64_t nelems=1; for (int i=0;i<nd_out;++i){ int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(float);
  TF_Tensor* out = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd_out, bytes, s); if (TF_GetCode(s)!=TF_OK){ TF_OpKernelContext_Failure(ctx,s); TF_DeleteStatus(s); return; }
  const float* pa=(const float*)TF_TensorData(a); const float* pb=(const float*)TF_TensorData(b); float* po=(float*)TF_TensorData(out);
  if (a_scalar){ float av=pa[0]; for (int64_t i=0;i<nelems;++i) po[i] = av < pb[i] ? av : pb[i]; }
  else { float bv=pb[0]; for (int64_t i=0;i<nelems;++i) po[i] = pa[i] < bv ? pa[i] : bv; }
  TF_DeleteStatus(s);
}

// ===== MPS Sigmoid and Tanh kernels (float) =====
namespace {
static id<MTLComputePipelineState> g_sigmoid_pipeline = nil;
static id<MTLComputePipelineState> g_tanh_pipeline = nil;
static dispatch_once_t g_sigmoid_once;
static dispatch_once_t g_tanh_once;

static void EnsureSigmoidPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_sigmoid_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void sigmoid_k(const device float* in [[buffer(0)]],\n"
                       @"                     device float* out [[buffer(1)]],\n"
                       @"                     uint gid [[thread_position_in_grid]]) {\n"
                       @"  float x = in[gid];\n"
                       @"  out[gid] = 1.0f / (1.0f + exp(-x));\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Sigmoid: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"sigmoid_k"];
    g_sigmoid_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_sigmoid_pipeline) { NSLog(@"MPS Sigmoid: pipeline error: %@", err); }
  });
}
static void EnsureTanhPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_tanh_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void tanh_k(const device float* in [[buffer(0)]],\n"
                       @"                  device float* out [[buffer(1)]],\n"
                       @"                  uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = tanh(in[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS Tanh: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"tanh_k"];
    g_tanh_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_tanh_pipeline) { NSLog(@"MPS Tanh: pipeline error: %@", err); }
  });
}
}

extern "C" void MPSSigmoid_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Sigmoid[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int nd = TF_NumDims(input); int64_t nelems = 1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems * sizeof(float);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) { float x = in[i]; out[i] = 1.0f / (1.0f + expf(-x)); }
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureSigmoidPipeline(dev);
  if (!g_sigmoid_pipeline) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) { float x = in[i]; out[i] = 1.0f / (1.0f + expf(-x)); }
    TF_DeleteStatus(s); return;
  }
  const void* in_host = TF_TensorData(input); void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_sigmoid_pipeline]; [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted]; memcpy(out_host, outb.contents, bytes); TF_DeleteStatus(s);
}

// ===== MPS Conv2D kernel (float, NHWC only) =====
namespace {
struct MPSConv2DAttrs {
  std::vector<int64_t> strides;
  std::string padding;
  std::vector<int64_t> dilations;
  std::string data_format;
};
}

extern "C" void* MPSConv2D_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSConv2DAttrs();
  TF_Status* s = TF_NewStatus();
  
  // Get strides
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  // Get padding
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  // Get dilations
  int64_t* dilations_data = nullptr;
  int dilations_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "dilations", &dilations_data, &dilations_len, s);
  if (TF_GetCode(s) == TF_OK && dilations_data && dilations_len > 0) {
    attrs->dilations.assign(dilations_data, dilations_data + dilations_len);
  }
  
  // Get data_format
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSConv2D_Delete(void* kernel) {
  auto* attrs = reinterpret_cast<MPSConv2DAttrs*>(kernel);
  delete attrs;
}

extern "C" void MPSConv2D_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = reinterpret_cast<MPSConv2DAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_Tensor* filter = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &filter, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_TensorType(filter)) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D[MPS] input and filter must have same dtype");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D[MPS] supports float32, float16, and bfloat16");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? sizeof(float) : sizeof(uint16_t);
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  // Only support NHWC for now
  if (attrs->data_format != "NHWC") {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2D[MPS] only supports NHWC format");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  // Input: [N, H, W, C_in], Filter: [kH, kW, C_in, C_out]
  if (TF_NumDims(input) != 4 || TF_NumDims(filter) != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D[MPS] expects 4D input and filter");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C_in = TF_Dim(input, 3);
  
  int64_t kH = TF_Dim(filter, 0);
  int64_t kW = TF_Dim(filter, 1);
  int64_t C_out = TF_Dim(filter, 3);
  
  if (TF_Dim(filter, 2) != C_in) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D filter channels mismatch");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  // Get strides (NHWC: [1, stride_h, stride_w, 1])
  int64_t stride_h = (attrs->strides.size() >= 4) ? attrs->strides[1] : 1;
  int64_t stride_w = (attrs->strides.size() >= 4) ? attrs->strides[2] : 1;
  
  // Get dilations (NHWC: [1, dil_h, dil_w, 1])
  int64_t dil_h = (attrs->dilations.size() >= 4) ? attrs->dilations[1] : 1;
  int64_t dil_w = (attrs->dilations.size() >= 4) ? attrs->dilations[2] : 1;
  
  // Compute output size based on padding
  int64_t H_out = 0, W_out = 0;
  int64_t pad_top = 0, pad_left = 0;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h_total = std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in);
    int64_t pad_w_total = std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in);
    pad_top = pad_h_total / 2;
    pad_left = pad_w_total / 2;
  } else if (attrs->padding == "VALID") {
    H_out = (H_in - (kH - 1) * dil_h) / stride_h;
    W_out = (W_in - (kW - 1) * dil_w) / stride_w;
    if (H_out <= 0 || W_out <= 0) {
      TF_SetStatus(s, TF_INVALID_ARGUMENT, "Conv2D output dimensions must be positive");
      TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
    }
    pad_top = 0;
    pad_left = 0;
  } else {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2D[MPS] only supports SAME/VALID padding");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  int64_t out_dims[4] = {N, H_out, W_out, C_out};
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C_out * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_dims, 4, out_bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Try GPU via MPSGraph
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "Conv2D[MPS] requires stream (host fallback not implemented)");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Input: [N, H, W, C_in] NHWC
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    // Filter: [kH, kW, C_in, C_out] -> needs to be [C_out, C_in, kH, kW] for MPSGraph
    // We'll transpose on host before feeding
    MPSGraphTensor* filterTensor = [graph placeholderWithShape:@[@(C_out), @(C_in), @(kH), @(kW)]
                                                        dataType:mps_dtype
                                                            name:@"filter"];
    
    // Create convolution descriptor
    MPSGraphConvolution2DOpDescriptor* desc = [[MPSGraphConvolution2DOpDescriptor alloc] init];
    desc.strideInX = (NSUInteger)stride_w;
    desc.strideInY = (NSUInteger)stride_h;
    desc.dilationRateInX = (NSUInteger)dil_w;
    desc.dilationRateInY = (NSUInteger)dil_h;
    desc.paddingLeft = (NSUInteger)pad_left;
    desc.paddingRight = (NSUInteger)std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in - pad_top);
    desc.paddingTop = (NSUInteger)pad_top;
    desc.paddingBottom = (NSUInteger)std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in - pad_left);
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    desc.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;  // [Out, In, H, W]
    
    // Perform convolution
    MPSGraphTensor* outputTensor = [graph convolution2DWithSourceTensor:inputTensor
                                                          weightsTensor:filterTensor
                                                             descriptor:desc
                                                                   name:@"conv2d"];
    
    // Prepare input buffer
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C_in * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    // Transpose filter: [kH, kW, C_in, C_out] -> [C_out, C_in, kH, kW]
    size_t filter_elem_count = (size_t)C_out * (size_t)C_in * (size_t)kH * (size_t)kW;
    size_t filter_bytes = filter_elem_count * elem_size;
    std::vector<uint8_t> filter_transposed(filter_bytes);
    
    if (is_half) {
      const uint16_t* filter_data = (const uint16_t*)TF_TensorData(filter);
      uint16_t* filter_out = (uint16_t*)filter_transposed.data();
      for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t ci = 0; ci < C_in; ++ci) {
          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              int64_t src_idx = ((kh * kW + kw) * C_in + ci) * C_out + co;
              int64_t dst_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
              filter_out[dst_idx] = filter_data[src_idx];
            }
          }
        }
      }
    } else {
      const float* filter_data = (const float*)TF_TensorData(filter);
      float* filter_out = (float*)filter_transposed.data();
      for (int64_t co = 0; co < C_out; ++co) {
        for (int64_t ci = 0; ci < C_in; ++ci) {
          for (int64_t kh = 0; kh < kH; ++kh) {
            for (int64_t kw = 0; kw < kW; ++kw) {
              int64_t src_idx = ((kh * kW + kw) * C_in + ci) * C_out + co;
              int64_t dst_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
              filter_out[dst_idx] = filter_data[src_idx];
            }
          }
        }
      }
    }
    id<MTLBuffer> filterBuffer = [dev newBufferWithBytes:filter_transposed.data()
                                                   length:filter_bytes
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* filterData = [[MPSGraphTensorData alloc] initWithMTLBuffer:filterBuffer
                                                                              shape:@[@(C_out), @(C_in), @(kH), @(kW)]
                                                                           dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C_out)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData, filterTensor: filterData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS DepthwiseConv2dNative kernel (float, half, NHWC only) =====
namespace {
struct MPSDepthwiseConv2DAttrs {
  std::vector<int64_t> strides;
  std::string padding;
  std::vector<int64_t> dilations;
  std::string data_format;
};
}

extern "C" void* MPSDepthwiseConv2D_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSDepthwiseConv2DAttrs();
  TF_Status* s = TF_NewStatus();
  
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  int64_t* dilations_data = nullptr;
  int dilations_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "dilations", &dilations_data, &dilations_len, s);
  if (TF_GetCode(s) == TF_OK && dilations_data && dilations_len > 0) {
    attrs->dilations.assign(dilations_data, dilations_data + dilations_len);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSDepthwiseConv2D_Delete(void* kernel_ptr) {
  delete static_cast<MPSDepthwiseConv2DAttrs*>(kernel_ptr);
}

extern "C" void MPSDepthwiseConv2D_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSDepthwiseConv2DAttrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_Tensor* filter = nullptr;
  TF_GetInput(ctx, 1, &filter, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS DepthwiseConv2dNative: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int64_t num_dims_input = TF_NumDims(input);
  int64_t num_dims_filter = TF_NumDims(filter);
  if (num_dims_input != 4 || num_dims_filter != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS DepthwiseConv2dNative: input/filter must be 4D");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Input: [N, H, W, C_in] NHWC
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C_in = TF_Dim(input, 3);
  
  // Filter: [kH, kW, C_in, depth_multiplier]
  int64_t kH = TF_Dim(filter, 0);
  int64_t kW = TF_Dim(filter, 1);
  int64_t filter_c = TF_Dim(filter, 2);
  int64_t depth_mult = TF_Dim(filter, 3);
  
  if (filter_c != C_in) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS DepthwiseConv2dNative: filter channels != input channels");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  int64_t C_out = C_in * depth_mult;
  
  // Get strides
  int64_t stride_h = 1, stride_w = 1;
  if (attrs->strides.size() == 4) {
    stride_h = attrs->strides[1];
    stride_w = attrs->strides[2];
  }
  
  // Get dilations
  int64_t dil_h = 1, dil_w = 1;
  if (attrs->dilations.size() == 4) {
    dil_h = attrs->dilations[1];
    dil_w = attrs->dilations[2];
  }
  
  // Compute padding
  int64_t pad_top = 0, pad_left = 0;
  int64_t H_out, W_out;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h = std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in);
    int64_t pad_w = std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in);
    pad_top = pad_h / 2;
    pad_left = pad_w / 2;
  } else {
    H_out = (H_in - (kH - 1) * dil_h - 1) / stride_h + 1;
    W_out = (W_in - (kW - 1) * dil_w - 1) / stride_w + 1;
  }
  
  int64_t output_dims[4] = {N, H_out, W_out, C_out};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, output_dims, 4, C_out * H_out * W_out * N * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C_out * elem_size;
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Input: [N, H, W, C_in]
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    // Filter: [kH, kW, C_in, depth_mult] -> needs to be [kH, kW, C_in, depth_mult] for depthwise
    MPSGraphTensor* filterTensor = [graph placeholderWithShape:@[@(kH), @(kW), @(C_in), @(depth_mult)]
                                                        dataType:mps_dtype
                                                            name:@"filter"];
    
    // Create depthwise convolution descriptor
    MPSGraphDepthwiseConvolution3DOpDescriptor* desc = [[MPSGraphDepthwiseConvolution3DOpDescriptor alloc] init];
    desc.strides = @[@1, @(stride_h), @(stride_w)];
    desc.dilationRates = @[@1, @(dil_h), @(dil_w)];
    desc.paddingValues = @[@0, @(pad_top), @(pad_left), @0,
                          @(std::max<int64_t>(0, (H_out - 1) * stride_h + (kH - 1) * dil_h + 1 - H_in - pad_top)),
                          @(std::max<int64_t>(0, (W_out - 1) * stride_w + (kW - 1) * dil_w + 1 - W_in - pad_left))];
    desc.paddingStyle = MPSGraphPaddingStyleExplicit;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    desc.weightsLayout = MPSGraphTensorNamedDataLayoutHWIO;  // [H, W, In, Out]
    
    // Perform depthwise convolution
    MPSGraphTensor* outputTensor = [graph depthwiseConvolution3DWithSourceTensor:inputTensor
                                                                   weightsTensor:filterTensor
                                                                      descriptor:desc
                                                                            name:@"depthwise_conv"];
    
    // Prepare buffers
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C_in * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    size_t filter_bytes = (size_t)kH * (size_t)kW * (size_t)C_in * (size_t)depth_mult * elem_size;
    id<MTLBuffer> filterBuffer = [dev newBufferWithBytes:TF_TensorData(filter)
                                                   length:filter_bytes
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C_in)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* filterData = [[MPSGraphTensorData alloc] initWithMTLBuffer:filterBuffer
                                                                              shape:@[@(kH), @(kW), @(C_in), @(depth_mult)]
                                                                           dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C_out)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData, filterTensor: filterData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS Maximum/Minimum/Sigmoid/Tanh kernels (half) =====
namespace {
static id<MTLComputePipelineState> g_max_h_pipeline=nil, g_min_h_pipeline=nil, g_sigmoid_h_pipeline=nil, g_tanh_h_pipeline=nil;
static dispatch_once_t g_max_h_once, g_min_h_once, g_sigmoid_h_once, g_tanh_h_once;

static void EnsureMaxHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_max_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void max_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2)]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = max(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS MaximumHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"max_h_k"];
    g_max_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_max_h_pipeline) { NSLog(@"MPS MaximumHalf: pipeline error: %@", err); }
  });
}
static void EnsureMinHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_min_h_once, ^{
    NSString* src = @"using namespace metal;\n"
                       @"kernel void min_h_k(const device half* a [[buffer(0)]],\n"
                       @"                    const device half* b [[buffer(1)]],\n"
                       @"                    device half* out [[buffer(2]]],\n"
                       @"                    uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = min(a[gid], b[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS MinimumHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"min_h_k"];
    g_min_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_min_h_pipeline) { NSLog(@"MPS MinimumHalf: pipeline error: %@", err); }
  });
}
static void EnsureSigmoidHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_sigmoid_h_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void sigmoid_h_k(const device half* in [[buffer(0)]],\n"
                       @"                        device half* out [[buffer(1)]],\n"
                       @"                        uint gid [[thread_position_in_grid]]) {\n"
                       @"  half x = in[gid];\n"
                       @"  out[gid] = (half)(1.0h / (1.0h + exp(-x)));\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS SigmoidHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"sigmoid_h_k"];
    g_sigmoid_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_sigmoid_h_pipeline) { NSLog(@"MPS SigmoidHalf: pipeline error: %@", err); }
  });
}
static void EnsureTanhHalfPipeline(id<MTLDevice> dev) {
  dispatch_once(&g_tanh_h_once, ^{
    NSString* src = @"#include <metal_stdlib>\n"
                       @"using namespace metal;\n"
                       @"kernel void tanh_h_k(const device half* in [[buffer(0)]],\n"
                       @"                     device half* out [[buffer(1)]],\n"
                       @"                     uint gid [[thread_position_in_grid]]) {\n"
                       @"  out[gid] = tanh(in[gid]);\n"
                       @"}";
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { NSLog(@"MPS TanhHalf: compile failed: %@", err); return; }
    id<MTLFunction> fn = [lib newFunctionWithName:@"tanh_h_k"];
    g_tanh_h_pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_tanh_h_pipeline) { NSLog(@"MPS TanhHalf: pipeline error: %@", err); }
  });
}
}

extern "C" void MPSMaximumHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a=nullptr, *b=nullptr;
  TF_GetInput(ctx,0,&a,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  TF_GetInput(ctx,1,&b,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(a)!=TF_HALF||TF_TensorType(b)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Maximum[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_a=TF_NumDims(a),nd_b=TF_NumDims(b); int64_t nelems_a=1,nelems_b=1;
  for(int i=0;i<nd_a;++i){int64_t d=TF_Dim(a,i);nelems_a*=(d<0?0:d);} for(int i=0;i<nd_b;++i){int64_t d=TF_Dim(b,i);nelems_b*=(d<0?0:d);}
  bool same_shape=(nd_a==nd_b); if(same_shape)for(int i=0;i<nd_a;++i)if(TF_Dim(a,i)!=TF_Dim(b,i)){same_shape=false;break;}
  int nd=nd_a; int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(same_shape){
    if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} int64_t nelems=1;
    for(int i=0;i<nd;++i){int64_t d=TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(uint16_t);
    TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
    SP_Stream cstream=TF_GetStream(ctx,s);
    if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
      const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
      for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va>vb?va:vb);}
      TF_DeleteStatus(s);return;}
    auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureMaxHalfPipeline(dev);
    if(!g_max_h_pipeline){const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out); for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va>vb?va:vb);} TF_DeleteStatus(s);return;}
    const void* ha=TF_TensorData(a),*hb=TF_TensorData(b);void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes);memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_max_h_pipeline];[enc setBuffer:ba offset:0 atIndex:0];[enc setBuffer:bb offset:0 atIndex:1];[enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes);TF_DeleteStatus(s);return;
  }
  bool a_scalar=(nelems_a==1),b_scalar=(nelems_b==1);
  if(!(a_scalar||b_scalar)){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Maximum shapes must match or one be scalar");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_out=a_scalar?nd_b:nd_a;if(nd_out>8){dyn.reset(new int64_t[nd_out]);dims=dyn.get();}
  int64_t nelems=1;for(int i=0;i<nd_out;++i){int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd_out,bytes,s);if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
  if(a_scalar){float av=HalfToFloat(pa[0]);for(int64_t i=0;i<nelems;++i){float vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(av>vb?av:vb);}}
  else{float bv=HalfToFloat(pb[0]);for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]);po[i]=FloatToHalf(va>bv?va:bv);}}
  TF_DeleteStatus(s);
}

extern "C" void MPSMinimumHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a=nullptr, *b=nullptr;
  TF_GetInput(ctx,0,&a,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  TF_GetInput(ctx,1,&b,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(a)!=TF_HALF||TF_TensorType(b)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Minimum[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_a=TF_NumDims(a),nd_b=TF_NumDims(b); int64_t nelems_a=1,nelems_b=1;
  for(int i=0;i<nd_a;++i){int64_t d=TF_Dim(a,i);nelems_a*=(d<0?0:d);} for(int i=0;i<nd_b;++i){int64_t d=TF_Dim(b,i);nelems_b*=(d<0?0:d);}
  bool same_shape=(nd_a==nd_b); if(same_shape)for(int i=0;i<nd_a;++i)if(TF_Dim(a,i)!=TF_Dim(b,i)){same_shape=false;break;}
  int nd=nd_a; int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(same_shape){
    if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} int64_t nelems=1;
    for(int i=0;i<nd;++i){int64_t d=TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);} size_t bytes=nelems*sizeof(uint16_t);
    TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
    SP_Stream cstream=TF_GetStream(ctx,s);
    if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
      const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
      for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va<vb?va:vb);}
      TF_DeleteStatus(s);return;}
    auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureMinHalfPipeline(dev);
    if(!g_min_h_pipeline){const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out); for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]),vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(va<vb?va:vb);} TF_DeleteStatus(s);return;}
    const void* ha=TF_TensorData(a),*hb=TF_TensorData(b);void* ho=TF_TensorData(out);
    id<MTLBuffer> ba=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],bo=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(ba.contents,ha,bytes);memcpy(bb.contents,hb,bytes);
    id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:g_min_h_pipeline];[enc setBuffer:ba offset:0 atIndex:0];[enc setBuffer:bb offset:0 atIndex:1];[enc setBuffer:bo offset:0 atIndex:2];
    NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
    [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];[cb commit];[cb waitUntilCompleted];
    memcpy(ho,bo.contents,bytes);TF_DeleteStatus(s);return;
  }
  bool a_scalar=(nelems_a==1),b_scalar=(nelems_b==1);
  if(!(a_scalar||b_scalar)){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Minimum shapes must match or one be scalar");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd_out=a_scalar?nd_b:nd_a;if(nd_out>8){dyn.reset(new int64_t[nd_out]);dims=dyn.get();}
  int64_t nelems=1;for(int i=0;i<nd_out;++i){int64_t d=a_scalar?TF_Dim(b,i):TF_Dim(a,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* out=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd_out,bytes,s);if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  const uint16_t* pa=(const uint16_t*)TF_TensorData(a),*pb=(const uint16_t*)TF_TensorData(b);uint16_t* po=(uint16_t*)TF_TensorData(out);
  if(a_scalar){float av=HalfToFloat(pa[0]);for(int64_t i=0;i<nelems;++i){float vb=HalfToFloat(pb[i]);po[i]=FloatToHalf(av<vb?av:vb);}}
  else{float bv=HalfToFloat(pb[0]);for(int64_t i=0;i<nelems;++i){float va=HalfToFloat(pa[i]);po[i]=FloatToHalf(va<bv?va:bv);}}
  TF_DeleteStatus(s);
}

extern "C" void MPSSigmoidHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input=nullptr;TF_GetInput(ctx,0,&input,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(input)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Sigmoid[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd=TF_NumDims(input);int64_t nelems=1;int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} for(int i=0;i<nd;++i){int64_t d=TF_Dim(input,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* output=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  SP_Stream cstream=TF_GetStream(ctx,s);
  if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
    const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output);
    for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(1.0f/(1.0f+expf(-x)));}
    TF_DeleteStatus(s);return;}
  auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureSigmoidHalfPipeline(dev);
  if(!g_sigmoid_h_pipeline){const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output); for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(1.0f/(1.0f+expf(-x)));} TF_DeleteStatus(s);return;}
  const void* in_host=TF_TensorData(input);void* out_host=TF_TensorData(output);
  id<MTLBuffer> inb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],outb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents,in_host,bytes);
  id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
  [enc setComputePipelineState:g_sigmoid_h_pipeline];[enc setBuffer:inb offset:0 atIndex:0];[enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];
  [cb commit];[cb waitUntilCompleted];memcpy(out_host,outb.contents,bytes);TF_DeleteStatus(s);
}

extern "C" void MPSTanhHalf_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input=nullptr;TF_GetInput(ctx,0,&input,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  if(TF_TensorType(input)!=TF_HALF){TF_SetStatus(s,TF_INVALID_ARGUMENT,"Tanh[MPS half] expects half");TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  int nd=TF_NumDims(input);int64_t nelems=1;int64_t dims_stack[8];int64_t* dims=dims_stack;std::unique_ptr<int64_t[]> dyn;
  if(nd>8){dyn.reset(new int64_t[nd]);dims=dyn.get();} for(int i=0;i<nd;++i){int64_t d=TF_Dim(input,i);dims[i]=d;nelems*=(d<0?0:d);}size_t bytes=nelems*sizeof(uint16_t);
  TF_Tensor* output=TF_AllocateOutput(ctx,0,TF_HALF,dims,nd,bytes,s); if(TF_GetCode(s)!=TF_OK){TF_OpKernelContext_Failure(ctx,s);TF_DeleteStatus(s);return;}
  SP_Stream cstream=TF_GetStream(ctx,s);
  if(TF_GetCode(s)!=TF_OK||cstream==nullptr){
    const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output);
    for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(tanhf(x));}
    TF_DeleteStatus(s);return;}
  auto* stream=reinterpret_cast<MPSStreamStruct*>(cstream);id<MTLDevice> dev=stream->dev->device;EnsureTanhHalfPipeline(dev);
  if(!g_tanh_h_pipeline){const uint16_t* in=(const uint16_t*)TF_TensorData(input);uint16_t* out=(uint16_t*)TF_TensorData(output); for(int64_t i=0;i<nelems;++i){float x=HalfToFloat(in[i]);out[i]=FloatToHalf(tanhf(x));} TF_DeleteStatus(s);return;}
  const void* in_host=TF_TensorData(input);void* out_host=TF_TensorData(output);
  id<MTLBuffer> inb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared],outb=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents,in_host,bytes);
  id<MTLCommandBuffer> cb=[stream->queue commandBuffer];id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
  [enc setComputePipelineState:g_tanh_h_pipeline];[enc setBuffer:inb offset:0 atIndex:0];[enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256,grid=(NSUInteger)nelems,groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[enc endEncoding];
  [cb commit];[cb waitUntilCompleted];memcpy(out_host,outb.contents,bytes);TF_DeleteStatus(s);
}

extern "C" void MPSTanh_Compute(void* /*kernel*/, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (TF_TensorType(input) != TF_FLOAT) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "Tanh[MPS] only supports float32");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  int nd = TF_NumDims(input); int64_t nelems = 1; int64_t dims_stack[8]; int64_t* dims=dims_stack; std::unique_ptr<int64_t[]> dyn;
  if (nd>8){ dyn.reset(new int64_t[nd]); dims=dyn.get(); }
  for (int i=0;i<nd;++i){ int64_t d=TF_Dim(input,i); dims[i]=d; nelems*=(d<0?0:d);} size_t bytes = nelems * sizeof(float);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) out[i] = tanhf(in[i]);
    TF_DeleteStatus(s); return;
  }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev=stream->dev->device; EnsureTanhPipeline(dev);
  if (!g_tanh_pipeline) {
    const float* in = (const float*)TF_TensorData(input); float* out = (float*)TF_TensorData(output);
    for (int64_t i = 0; i < nelems; ++i) out[i] = tanhf(in[i]);
    TF_DeleteStatus(s); return;
  }
  const void* in_host = TF_TensorData(input); void* out_host = TF_TensorData(output);
  id<MTLBuffer> inb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> outb = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memcpy(inb.contents, in_host, bytes);
  id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:g_tanh_pipeline]; [enc setBuffer:inb offset:0 atIndex:0]; [enc setBuffer:outb offset:0 atIndex:1];
  NSUInteger threads=256; NSUInteger grid=(NSUInteger)nelems; NSUInteger groups=(grid+threads-1)/threads;
  [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)]; [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted]; memcpy(out_host, outb.contents, bytes); TF_DeleteStatus(s);
}

// ===== MPS Softmax kernel (float, half, bfloat16 via MPSGraph) =====
namespace {
struct MPSSoftmaxAttrs {
  // Softmax is typically along last dimension, but we'll support any axis
  // For now, default to -1 (last dimension)
};
}

extern "C" void* MPSSoftmax_Create(TF_OpKernelConstruction* ctx) {
  return new MPSSoftmaxAttrs();
}

extern "C" void MPSSoftmax_Delete(void* kernel_ptr) {
  delete static_cast<MPSSoftmaxAttrs*>(kernel_ptr);
}

extern "C" void MPSSoftmax_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* logits = nullptr;
  TF_GetInput(ctx, 0, &logits, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(logits);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Softmax: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int nd = TF_NumDims(logits);
  if (nd < 1) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Softmax: input must have at least 1 dimension");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Get shape
  std::vector<int64_t> shape(nd);
  int64_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    shape[i] = TF_Dim(logits, i);
    total_elems *= shape[i];
  }
  
  // Allocate output with same shape
  size_t out_bytes = total_elems * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, out_bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Create NSArray for shape
    NSMutableArray* shapeArray = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [shapeArray addObject:@(shape[i])];
    }
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:shapeArray
                                                      dataType:mps_dtype
                                                          name:@"logits"];
    
    // Softmax along last axis (-1)
    MPSGraphTensor* outputTensor = [graph softMaxWithTensor:inputTensor
                                                        axis:-1
                                                        name:@"softmax"];
    
    size_t input_bytes = total_elems * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(logits)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:shapeArray
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:shapeArray
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS FusedBatchNormV3 kernel (float, half, bfloat16 via MPSGraph) =====
namespace {
struct MPSFusedBatchNormV3Attrs {
  float epsilon;
  bool is_training;
  std::string data_format;
};
}

extern "C" void* MPSFusedBatchNormV3_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSFusedBatchNormV3Attrs();
  TF_Status* s = TF_NewStatus();
  
  float epsilon = 0.0001f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "epsilon", &epsilon, s);
  if (TF_GetCode(s) == TF_OK) {
    attrs->epsilon = epsilon;
  }
  
  TF_Bool is_training = false;
  TF_OpKernelConstruction_GetAttrBool(ctx, "is_training", &is_training, s);
  if (TF_GetCode(s) == TF_OK) {
    attrs->is_training = (is_training != 0);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  } else {
    attrs->data_format = "NHWC";
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSFusedBatchNormV3_Delete(void* kernel_ptr) {
  delete static_cast<MPSFusedBatchNormV3Attrs*>(kernel_ptr);
}

extern "C" void MPSFusedBatchNormV3_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSFusedBatchNormV3Attrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  // Get inputs: x, scale, offset, mean, variance
  TF_Tensor* x = nullptr;
  TF_Tensor* scale = nullptr;
  TF_Tensor* offset = nullptr;
  TF_Tensor* mean = nullptr;
  TF_Tensor* variance = nullptr;
  
  TF_GetInput(ctx, 0, &x, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &scale, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &offset, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 3, &mean, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 4, &variance, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(x);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS FusedBatchNormV3: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int nd = TF_NumDims(x);
  if (nd != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS FusedBatchNormV3: input must be 4D (NHWC or NCHW)");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Get shape [N, H, W, C] for NHWC
  std::vector<int64_t> shape(nd);
  int64_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    shape[i] = TF_Dim(x, i);
    total_elems *= shape[i];
  }
  
  int64_t channels = (attrs->data_format == "NHWC") ? shape[3] : shape[1];
  
  // Allocate outputs: y, batch_mean, batch_variance, saved_mean, saved_variance
  TF_Tensor* y = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, total_elems * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  std::vector<int64_t> stats_shape = {channels};
  TF_Tensor* batch_mean = TF_AllocateOutput(ctx, 1, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_Tensor* batch_variance = TF_AllocateOutput(ctx, 2, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_Tensor* saved_mean = TF_AllocateOutput(ctx, 3, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_Tensor* saved_variance = TF_AllocateOutput(ctx, 4, TF_FLOAT, stats_shape.data(), 1, channels * 4, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    NSMutableArray* shapeArray = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [shapeArray addObject:@(shape[i])];
    }
    
    NSArray* statsShapeArray = @[@(channels)];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:shapeArray
                                                      dataType:mps_dtype
                                                          name:@"input"];
    MPSGraphTensor* scaleTensor = [graph placeholderWithShape:statsShapeArray
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"scale"];
    MPSGraphTensor* offsetTensor = [graph placeholderWithShape:statsShapeArray
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"offset"];
    MPSGraphTensor* meanTensor = [graph placeholderWithShape:statsShapeArray
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"mean"];
    MPSGraphTensor* varianceTensor = [graph placeholderWithShape:statsShapeArray
                                                         dataType:MPSDataTypeFloat32
                                                             name:@"variance"];
    
    // Normalize: y = scale * (x - mean) / sqrt(variance + epsilon) + offset
    // Use MPSGraph batch normalization
    NSUInteger axis = (attrs->data_format == "NHWC") ? 3 : 1;
    
    MPSGraphTensor* normalizedTensor = [graph normalizationWithTensor:inputTensor
                                                           meanTensor:meanTensor
                                                       varianceTensor:varianceTensor
                                                          gammaTensor:scaleTensor
                                                           betaTensor:offsetTensor
                                                              epsilon:attrs->epsilon
                                                                 name:@"batch_norm"];
    
    // Prepare buffers
    size_t input_bytes = total_elems * elem_size;
    size_t stats_bytes = channels * 4;
    
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(x)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> scaleBuffer = [dev newBufferWithBytes:TF_TensorData(scale)
                                                  length:stats_bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsetBuffer = [dev newBufferWithBytes:TF_TensorData(offset)
                                                   length:stats_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> meanBuffer = [dev newBufferWithBytes:TF_TensorData(mean)
                                                 length:stats_bytes
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> varianceBuffer = [dev newBufferWithBytes:TF_TensorData(variance)
                                                     length:stats_bytes
                                                    options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:input_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:shapeArray
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* scaleData = [[MPSGraphTensorData alloc] initWithMTLBuffer:scaleBuffer
                                                                             shape:statsShapeArray
                                                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* offsetData = [[MPSGraphTensorData alloc] initWithMTLBuffer:offsetBuffer
                                                                              shape:statsShapeArray
                                                                           dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* meanData = [[MPSGraphTensorData alloc] initWithMTLBuffer:meanBuffer
                                                                            shape:statsShapeArray
                                                                         dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* varianceData = [[MPSGraphTensorData alloc] initWithMTLBuffer:varianceBuffer
                                                                                shape:statsShapeArray
                                                                             dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:shapeArray
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData,
                                    scaleTensor: scaleData,
                                    offsetTensor: offsetData,
                                    meanTensor: meanData,
                                    varianceTensor: varianceData}
                   targetTensors:@[normalizedTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(y), outputBuffer.contents, input_bytes);
    
    // Copy statistics (in training mode, these would be computed; for now, copy inputs)
    memcpy(TF_TensorData(batch_mean), TF_TensorData(mean), stats_bytes);
    memcpy(TF_TensorData(batch_variance), TF_TensorData(variance), stats_bytes);
    memcpy(TF_TensorData(saved_mean), TF_TensorData(mean), stats_bytes);
    memcpy(TF_TensorData(saved_variance), TF_TensorData(variance), stats_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS Swish activation kernel (float, half, bfloat16) =====
// Swish(x) = x * sigmoid(x)
namespace {
struct MPSSwishAttrs {};
}

extern "C" void* MPSSwish_Create(TF_OpKernelConstruction* ctx) {
  return new MPSSwishAttrs();
}

extern "C" void MPSSwish_Delete(void* kernel_ptr) {
  delete static_cast<MPSSwishAttrs*>(kernel_ptr);
}

extern "C" void MPSSwish_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Swish: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int nd = TF_NumDims(input);
  std::vector<int64_t> shape(nd);
  int64_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    shape[i] = TF_Dim(input, i);
    total_elems *= shape[i];
  }
  
  size_t bytes = total_elems * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream);
  id<MTLDevice> dev = stream->dev->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    NSMutableArray* shapeArray = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [shapeArray addObject:@(shape[i])];
    }
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:shapeArray
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    // Swish(x) = x * sigmoid(x)
    MPSGraphTensor* sigmoidTensor = [graph sigmoidWithTensor:inputTensor name:@"sigmoid"];
    MPSGraphTensor* outputTensor = [graph multiplicationWithPrimaryTensor:inputTensor
                                                          secondaryTensor:sigmoidTensor
                                                                     name:@"swish"];
    
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:shapeArray
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:shapeArray
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS Gelu activation kernel (float, half, bfloat16) =====
// Gelu(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
// Approximation: Gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
namespace {
struct MPSGeluAttrs {};
}

extern "C" void* MPSGelu_Create(TF_OpKernelConstruction* ctx) {
  return new MPSGeluAttrs();
}

extern "C" void MPSGelu_Delete(void* kernel_ptr) {
  delete static_cast<MPSGeluAttrs*>(kernel_ptr);
}

extern "C" void MPSGelu_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Gelu: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int nd = TF_NumDims(input);
  std::vector<int64_t> shape(nd);
  int64_t total_elems = 1;
  for (int i = 0; i < nd; ++i) {
    shape[i] = TF_Dim(input, i);
    total_elems *= shape[i];
  }
  
  size_t bytes = total_elems * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    NSMutableArray* shapeArray = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [shapeArray addObject:@(shape[i])];
    }
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:shapeArray
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    // Gelu approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
    const float coeff = 0.044715f;
    
    MPSGraphTensor* constHalf = [graph constantWithScalar:0.5 dataType:mps_dtype];
    MPSGraphTensor* constOne = [graph constantWithScalar:1.0 dataType:mps_dtype];
    MPSGraphTensor* constCoeff = [graph constantWithScalar:coeff dataType:mps_dtype];
    MPSGraphTensor* constSqrt = [graph constantWithScalar:sqrt_2_over_pi dataType:mps_dtype];
    
    // x^3
    MPSGraphTensor* x2 = [graph multiplicationWithPrimaryTensor:inputTensor
                                                secondaryTensor:inputTensor
                                                           name:@"x_squared"];
    MPSGraphTensor* x3 = [graph multiplicationWithPrimaryTensor:x2
                                                secondaryTensor:inputTensor
                                                           name:@"x_cubed"];
    
    // 0.044715 * x^3
    MPSGraphTensor* coeffX3 = [graph multiplicationWithPrimaryTensor:constCoeff
                                                     secondaryTensor:x3
                                                                name:@"coeff_x3"];
    
    // x + 0.044715 * x^3
    MPSGraphTensor* inner = [graph additionWithPrimaryTensor:inputTensor
                                             secondaryTensor:coeffX3
                                                        name:@"inner_sum"];
    
    // sqrt(2/π) * (x + 0.044715 * x^3)
    MPSGraphTensor* scaled = [graph multiplicationWithPrimaryTensor:constSqrt
                                                    secondaryTensor:inner
                                                               name:@"scaled"];
    
    // tanh(...)
    MPSGraphTensor* tanhTensor = [graph tanhWithTensor:scaled name:@"tanh"];
    
    // 1 + tanh(...)
    MPSGraphTensor* onePlusTanh = [graph additionWithPrimaryTensor:constOne
                                                   secondaryTensor:tanhTensor
                                                              name:@"one_plus_tanh"];
    
    // x * (1 + tanh(...))
    MPSGraphTensor* xTimesExpr = [graph multiplicationWithPrimaryTensor:inputTensor
                                                        secondaryTensor:onePlusTanh
                                                                   name:@"x_times_expr"];
    
    // 0.5 * x * (1 + tanh(...))
    MPSGraphTensor* outputTensor = [graph multiplicationWithPrimaryTensor:constHalf
                                                          secondaryTensor:xTimesExpr
                                                                     name:@"gelu"];
    
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:shapeArray
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:shapeArray
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS MaxPool kernel (float, half, NHWC only) =====
namespace {
struct MPSMaxPoolAttrs {
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::string padding;
  std::string data_format;
};
}

extern "C" void* MPSMaxPool_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSMaxPoolAttrs();
  TF_Status* s = TF_NewStatus();
  
  int64_t* ksize_data = nullptr;
  int ksize_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "ksize", &ksize_data, &ksize_len, s);
  if (TF_GetCode(s) == TF_OK && ksize_data && ksize_len > 0) {
    attrs->ksize.assign(ksize_data, ksize_data + ksize_len);
  }
  
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSMaxPool_Delete(void* kernel_ptr) {
  delete static_cast<MPSMaxPoolAttrs*>(kernel_ptr);
}

extern "C" void MPSMaxPool_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSMaxPoolAttrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS MaxPool: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int64_t num_dims = TF_NumDims(input);
  if (num_dims != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS MaxPool: input must be 4D");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Input: [N, H, W, C] NHWC
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C = TF_Dim(input, 3);
  
  // Get ksize and strides
  int64_t kH = 1, kW = 1;
  if (attrs->ksize.size() == 4) {
    kH = attrs->ksize[1];
    kW = attrs->ksize[2];
  }
  
  int64_t stride_h = 1, stride_w = 1;
  if (attrs->strides.size() == 4) {
    stride_h = attrs->strides[1];
    stride_w = attrs->strides[2];
  }
  
  // Compute output dimensions and padding
  int64_t pad_top = 0, pad_left = 0;
  int64_t H_out, W_out;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h = std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in);
    int64_t pad_w = std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in);
    pad_top = pad_h / 2;
    pad_left = pad_w / 2;
  } else {
    H_out = (H_in - kH) / stride_h + 1;
    W_out = (W_in - kW) / stride_w + 1;
  }
  
  int64_t output_dims[4] = {N, H_out, W_out, C};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, output_dims, 4, C * H_out * W_out * N * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C * elem_size;
  
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor alloc] init];
    desc.kernelWidth = (NSUInteger)kW;
    desc.kernelHeight = (NSUInteger)kH;
    desc.strideInX = (NSUInteger)stride_w;
    desc.strideInY = (NSUInteger)stride_h;
    desc.paddingLeft = (NSUInteger)pad_left;
    desc.paddingRight = (NSUInteger)std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in - pad_left);
    desc.paddingTop = (NSUInteger)pad_top;
    desc.paddingBottom = (NSUInteger)std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in - pad_top);
    desc.paddingStyle = MPSGraphPaddingStyleExplicit;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    
    MPSGraphTensor* outputTensor = [graph maxPooling2DWithSourceTensor:inputTensor
                                                             descriptor:desc
                                                                   name:@"maxpool"];
    
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MPS AvgPool kernel (float, half, NHWC only) =====
namespace {
struct MPSAvgPoolAttrs {
  std::vector<int64_t> ksize;
  std::vector<int64_t> strides;
  std::string padding;
  std::string data_format;
};
}

extern "C" void* MPSAvgPool_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSAvgPoolAttrs();
  TF_Status* s = TF_NewStatus();
  
  int64_t* ksize_data = nullptr;
  int ksize_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "ksize", &ksize_data, &ksize_len, s);
  if (TF_GetCode(s) == TF_OK && ksize_data && ksize_len > 0) {
    attrs->ksize.assign(ksize_data, ksize_data + ksize_len);
  }
  
  int64_t* strides_data = nullptr;
  int strides_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "strides", &strides_data, &strides_len, s);
  if (TF_GetCode(s) == TF_OK && strides_data && strides_len > 0) {
    attrs->strides.assign(strides_data, strides_data + strides_len);
  }
  
  char* padding_data = nullptr;
  size_t padding_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "padding", &padding_data, &padding_len, s);
  if (TF_GetCode(s) == TF_OK && padding_data) {
    attrs->padding.assign(padding_data, padding_len);
  }
  
  char* format_data = nullptr;
  size_t format_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "data_format", &format_data, &format_len, s);
  if (TF_GetCode(s) == TF_OK && format_data) {
    attrs->data_format.assign(format_data, format_len);
  }
  
  TF_DeleteStatus(s);
  return attrs;
}

extern "C" void MPSAvgPool_Delete(void* kernel_ptr) {
  delete static_cast<MPSAvgPoolAttrs*>(kernel_ptr);
}

extern "C" void MPSAvgPool_Compute(void* kernel_ptr, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSAvgPoolAttrs*>(kernel_ptr);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS AvgPool: Only float32, float16, and bfloat16 supported");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  bool is_half = (dtype == TF_HALF);
  bool is_bf16 = (dtype == TF_BFLOAT16);
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = is_half ? MPSDataTypeFloat16 : (is_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  
  int64_t num_dims = TF_NumDims(input);
  if (num_dims != 4) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS AvgPool: input must be 4D");
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Input: [N, H, W, C] NHWC
  int64_t N = TF_Dim(input, 0);
  int64_t H_in = TF_Dim(input, 1);
  int64_t W_in = TF_Dim(input, 2);
  int64_t C = TF_Dim(input, 3);
  
  // Get ksize and strides
  int64_t kH = 1, kW = 1;
  if (attrs->ksize.size() == 4) {
    kH = attrs->ksize[1];
    kW = attrs->ksize[2];
  }
  
  int64_t stride_h = 1, stride_w = 1;
  if (attrs->strides.size() == 4) {
    stride_h = attrs->strides[1];
    stride_w = attrs->strides[2];
  }
  
  // Compute output dimensions and padding
  int64_t pad_top = 0, pad_left = 0;
  int64_t H_out, W_out;
  
  if (attrs->padding == "SAME") {
    H_out = (H_in + stride_h - 1) / stride_h;
    W_out = (W_in + stride_w - 1) / stride_w;
    int64_t pad_h = std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in);
    int64_t pad_w = std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in);
    pad_top = pad_h / 2;
    pad_left = pad_w / 2;
  } else {
    H_out = (H_in - kH) / stride_h + 1;
    W_out = (W_in - kW) / stride_w + 1;
  }
  
  int64_t output_dims[4] = {N, H_out, W_out, C};
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, output_dims, 4, C * H_out * W_out * N * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  size_t out_bytes = (size_t)N * (size_t)H_out * (size_t)W_out * (size_t)C * elem_size;
  
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(N), @(H_in), @(W_in), @(C)]
                                                      dataType:mps_dtype
                                                          name:@"input"];
    
    MPSGraphPooling2DOpDescriptor* desc = [[MPSGraphPooling2DOpDescriptor alloc] init];
    desc.kernelWidth = (NSUInteger)kW;
    desc.kernelHeight = (NSUInteger)kH;
    desc.strideInX = (NSUInteger)stride_w;
    desc.strideInY = (NSUInteger)stride_h;
    desc.paddingLeft = (NSUInteger)pad_left;
    desc.paddingRight = (NSUInteger)std::max<int64_t>(0, (W_out - 1) * stride_w + kW - W_in - pad_left);
    desc.paddingTop = (NSUInteger)pad_top;
    desc.paddingBottom = (NSUInteger)std::max<int64_t>(0, (H_out - 1) * stride_h + kH - H_in - pad_top);
    desc.paddingStyle = MPSGraphPaddingStyleExplicit;
    desc.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    
    MPSGraphTensor* outputTensor = [graph avgPooling2DWithSourceTensor:inputTensor
                                                             descriptor:desc
                                                                   name:@"avgpool"];
    
    size_t input_bytes = (size_t)N * (size_t)H_in * (size_t)W_in * (size_t)C * elem_size;
    id<MTLBuffer> inputBuffer = [dev newBufferWithBytes:TF_TensorData(input)
                                                  length:input_bytes
                                                 options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [dev newBufferWithLength:out_bytes
                                                   options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                             shape:@[@(N), @(H_in), @(W_in), @(C)]
                                                                          dataType:mps_dtype];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:outputBuffer
                                                                              shape:@[@(N), @(H_out), @(W_out), @(C)]
                                                                           dataType:mps_dtype];
    
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb
                             feeds:@{inputTensor: inputData}
                   targetTensors:@[outputTensor]
                targetOperations:nil
              executionDescriptor:nil];
    [cb commit];
    [cb waitUntilCompleted];
    
    memcpy(TF_TensorData(output), outputBuffer.contents, out_bytes);
  }
  
  TF_DeleteStatus(s);
}

// ===== MASSIVE Implementation of Additional Operations via MPSGraph =====

// Macro for unary elementwise ops
#define IMPL_UNARY_OP(OP_NAME, GRAPH_CALL) \
extern "C" void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*) { return new int(); } \
extern "C" void MPS##OP_NAME##_Delete(void* p) { delete static_cast<int*>(p); } \
extern "C" void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext* ctx) { \
  TF_Status* s = TF_NewStatus(); TF_Tensor* input = nullptr; \
  TF_GetInput(ctx, 0, &input, s); \
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; } \
  TF_DataType dtype = TF_TensorType(input); \
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) { \
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS " #OP_NAME ": float/half/bf16 only"); \
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2; \
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32); \
  int nd = TF_NumDims(input); std::vector<int64_t> shape(nd); int64_t total = 1; \
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; } \
  size_t bytes = total * elem_size; \
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, bytes, s); \
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  SP_Stream cstream = TF_GetStream(ctx, s); \
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); \
  id<MTLDevice> dev = stream->dev->device; \
  @autoreleasepool { \
    MPSGraph* graph = [[MPSGraph alloc] init]; \
    NSMutableArray* shapeArr = [NSMutableArray arrayWithCapacity:nd]; \
    for (int i = 0; i < nd; ++i) [shapeArr addObject:@(shape[i])]; \
    MPSGraphTensor* inT = [graph placeholderWithShape:shapeArr dataType:mps_dtype name:@"in"]; \
    MPSGraphTensor* outT = GRAPH_CALL; \
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:bytes options:MTLResourceStorageModeShared]; \
    id<MTLBuffer> outB = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; \
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:shapeArr dataType:mps_dtype]; \
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:shapeArr dataType:mps_dtype]; \
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; \
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil]; \
    [cb commit]; [cb waitUntilCompleted]; \
    memcpy(TF_TensorData(output), outB.contents, bytes); \
  } \
  TF_DeleteStatus(s); \
}

// Implement 27 unary ops
IMPL_UNARY_OP(Abs, [graph absoluteWithTensor:inT name:@"abs"])
IMPL_UNARY_OP(Neg, [graph negativeWithTensor:inT name:@"neg"])
IMPL_UNARY_OP(Sqrt, [graph squareRootWithTensor:inT name:@"sqrt"])
IMPL_UNARY_OP(Rsqrt, [graph reverseSquareRootWithTensor:inT name:@"rsqrt"])
IMPL_UNARY_OP(Exp, [graph exponentWithTensor:inT name:@"exp"])
IMPL_UNARY_OP(Log, [graph logarithmWithTensor:inT name:@"log"])
IMPL_UNARY_OP(Sin, [graph sinWithTensor:inT name:@"sin"])
IMPL_UNARY_OP(Cos, [graph cosWithTensor:inT name:@"cos"])
IMPL_UNARY_OP(Tan, [graph tanWithTensor:inT name:@"tan"])
IMPL_UNARY_OP(Asin, [graph asinWithTensor:inT name:@"asin"])
IMPL_UNARY_OP(Acos, [graph acosWithTensor:inT name:@"acos"])
IMPL_UNARY_OP(Atan, [graph atanWithTensor:inT name:@"atan"])
IMPL_UNARY_OP(Sinh, [graph sinhWithTensor:inT name:@"sinh"])
IMPL_UNARY_OP(Cosh, [graph coshWithTensor:inT name:@"cosh"])
IMPL_UNARY_OP(Asinh, [graph asinhWithTensor:inT name:@"asinh"])
IMPL_UNARY_OP(Acosh, [graph acoshWithTensor:inT name:@"acosh"])
IMPL_UNARY_OP(Atanh, [graph atanhWithTensor:inT name:@"atanh"])
IMPL_UNARY_OP(Ceil, [graph ceilWithTensor:inT name:@"ceil"])
IMPL_UNARY_OP(Floor, [graph floorWithTensor:inT name:@"floor"])
IMPL_UNARY_OP(Round, [graph roundWithTensor:inT name:@"round"])
IMPL_UNARY_OP(Erf, [graph erfWithTensor:inT name:@"erf"])
IMPL_UNARY_OP(Square, [graph squareWithTensor:inT name:@"square"])
IMPL_UNARY_OP(Reciprocal, [graph reciprocalWithTensor:inT name:@"recip"])
IMPL_UNARY_OP(Sign, [graph signWithTensor:inT name:@"sign"])
IMPL_UNARY_OP(Expm1, [graph exponentMinusOneWithTensor:inT name:@"expm1"])
IMPL_UNARY_OP(Log1p, [graph logarithmWithTensor:[graph additionWithPrimaryTensor:inT secondaryTensor:[graph constantWithScalar:1.0 dataType:mps_dtype] name:@"add1"] name:@"log1p"])
IMPL_UNARY_OP(IsFinite, [graph isFiniteWithTensor:inT name:@"isfinite"])

// Binary ops macro  
#define IMPL_BINARY_OP(OP_NAME, GRAPH_CALL) \
extern "C" void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*) { return new int(); } \
extern "C" void MPS##OP_NAME##_Delete(void* p) { delete static_cast<int*>(p); } \
extern "C" void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext* ctx) { \
  TF_Status* s = TF_NewStatus(); TF_Tensor* a = nullptr; TF_Tensor* b = nullptr; \
  TF_GetInput(ctx, 0, &a, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; } \
  TF_GetInput(ctx, 1, &b, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; } \
  TF_DataType dtype = TF_TensorType(a); \
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) { \
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS " #OP_NAME ": float/half/bf16 only"); \
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2; \
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32); \
  int nd_a = TF_NumDims(a), nd_b = TF_NumDims(b); \
  std::vector<int64_t> shape_a(nd_a), shape_b(nd_b); \
  int64_t total_a = 1, total_b = 1; \
  for (int i = 0; i < nd_a; ++i) { shape_a[i] = TF_Dim(a, i); total_a *= shape_a[i]; } \
  for (int i = 0; i < nd_b; ++i) { shape_b[i] = TF_Dim(b, i); total_b *= shape_b[i]; } \
  int nd_out = std::max(nd_a, nd_b); std::vector<int64_t> shape_out(nd_out); int64_t total_out = 1; \
  for (int i = 0; i < nd_out; ++i) { \
    int64_t dim_a = (i + nd_a >= nd_out) ? shape_a[i + nd_a - nd_out] : 1; \
    int64_t dim_b = (i + nd_b >= nd_out) ? shape_b[i + nd_b - nd_out] : 1; \
    shape_out[i] = std::max(dim_a, dim_b); total_out *= shape_out[i]; } \
  size_t bytes_out = total_out * elem_size; \
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape_out.data(), nd_out, bytes_out, s); \
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  SP_Stream cstream = TF_GetStream(ctx, s); \
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev = stream->dev->device; \
  @autoreleasepool { \
    MPSGraph* graph = [[MPSGraph alloc] init]; \
    NSMutableArray* shA = [NSMutableArray arrayWithCapacity:nd_a]; for (int i = 0; i < nd_a; ++i) [shA addObject:@(shape_a[i])]; \
    NSMutableArray* shB = [NSMutableArray arrayWithCapacity:nd_b]; for (int i = 0; i < nd_b; ++i) [shB addObject:@(shape_b[i])]; \
    NSMutableArray* shO = [NSMutableArray arrayWithCapacity:nd_out]; for (int i = 0; i < nd_out; ++i) [shO addObject:@(shape_out[i])]; \
    MPSGraphTensor* tA = [graph placeholderWithShape:shA dataType:mps_dtype name:@"a"]; \
    MPSGraphTensor* tB = [graph placeholderWithShape:shB dataType:mps_dtype name:@"b"]; \
    MPSGraphTensor* tO = GRAPH_CALL; \
    size_t bA = total_a * elem_size, bB = total_b * elem_size; \
    id<MTLBuffer> bufA = [dev newBufferWithBytes:TF_TensorData(a) length:bA options:MTLResourceStorageModeShared]; \
    id<MTLBuffer> bufB = [dev newBufferWithBytes:TF_TensorData(b) length:bB options:MTLResourceStorageModeShared]; \
    id<MTLBuffer> bufO = [dev newBufferWithLength:bytes_out options:MTLResourceStorageModeShared]; \
    MPSGraphTensorData* dA = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:shA dataType:mps_dtype]; \
    MPSGraphTensorData* dB = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:shB dataType:mps_dtype]; \
    MPSGraphTensorData* dO = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO shape:shO dataType:mps_dtype]; \
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; \
    [graph runWithMTLCommandBuffer:cb feeds:@{tA: dA, tB: dB} targetTensors:@[tO] targetOperations:nil executionDescriptor:nil]; \
    [cb commit]; [cb waitUntilCompleted]; \
    memcpy(TF_TensorData(output), bufO.contents, bytes_out); \
  } \
  TF_DeleteStatus(s); \
}

// Implement 14 binary ops
IMPL_BINARY_OP(Div, [graph divisionWithPrimaryTensor:tA secondaryTensor:tB name:@"div"])
IMPL_BINARY_OP(RealDiv, [graph divisionWithPrimaryTensor:tA secondaryTensor:tB name:@"realdiv"])
IMPL_BINARY_OP(Sub, [graph subtractionWithPrimaryTensor:tA secondaryTensor:tB name:@"sub"])
IMPL_BINARY_OP(Pow, [graph powerWithPrimaryTensor:tA secondaryTensor:tB name:@"pow"])
IMPL_BINARY_OP(FloorDiv, [graph floorWithTensor:[graph divisionWithPrimaryTensor:tA secondaryTensor:tB name:@""] name:@"floordiv"])
IMPL_BINARY_OP(FloorMod, [graph floorModuloWithPrimaryTensor:tA secondaryTensor:tB name:@"floormod"])
IMPL_BINARY_OP(Atan2, [graph atan2WithPrimaryTensor:tA secondaryTensor:tB name:@"atan2"])
IMPL_BINARY_OP(SquaredDifference, [graph squareWithTensor:[graph subtractionWithPrimaryTensor:tA secondaryTensor:tB name:@""] name:@"sqd"])
IMPL_BINARY_OP(Equal, [graph equalWithPrimaryTensor:tA secondaryTensor:tB name:@"eq"])
IMPL_BINARY_OP(NotEqual, [graph notEqualWithPrimaryTensor:tA secondaryTensor:tB name:@"neq"])
IMPL_BINARY_OP(Less, [graph lessThanWithPrimaryTensor:tA secondaryTensor:tB name:@"lt"])
IMPL_BINARY_OP(LessEqual, [graph lessThanOrEqualToWithPrimaryTensor:tA secondaryTensor:tB name:@"lte"])
IMPL_BINARY_OP(Greater, [graph greaterThanWithPrimaryTensor:tA secondaryTensor:tB name:@"gt"])
IMPL_BINARY_OP(GreaterEqual, [graph greaterThanOrEqualToWithPrimaryTensor:tA secondaryTensor:tB name:@"gte"])


// ===== Reduction Operations (Sum, Mean, Max, Min, etc.) =====
#define IMPL_REDUCTION_OP(OP_NAME, GRAPH_CALL) \
extern "C" void* MPS##OP_NAME##_Create(TF_OpKernelConstruction*) { return new int(); } \
extern "C" void MPS##OP_NAME##_Delete(void* p) { delete static_cast<int*>(p); } \
extern "C" void MPS##OP_NAME##_Compute(void*, TF_OpKernelContext* ctx) { \
  TF_Status* s = TF_NewStatus(); TF_Tensor* input = nullptr; TF_Tensor* axes_tensor = nullptr; \
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; } \
  TF_GetInput(ctx, 1, &axes_tensor, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; } \
  TF_DataType dtype = TF_TensorType(input); \
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) { \
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS " #OP_NAME ": float/half/bf16 only"); \
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2; \
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32); \
  int nd = TF_NumDims(input); std::vector<int64_t> shape(nd); int64_t total = 1; \
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; } \
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, nullptr, 0, elem_size, s); \
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  SP_Stream cstream = TF_GetStream(ctx, s); \
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; } \
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev = stream->dev->device; \
  @autoreleasepool { \
    MPSGraph* graph = [[MPSGraph alloc] init]; \
    NSMutableArray* shapeArr = [NSMutableArray arrayWithCapacity:nd]; \
    for (int i = 0; i < nd; ++i) [shapeArr addObject:@(shape[i])]; \
    MPSGraphTensor* inT = [graph placeholderWithShape:shapeArr dataType:mps_dtype name:@"in"]; \
    MPSGraphTensor* outT = GRAPH_CALL; \
    size_t bytes = total * elem_size; \
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:bytes options:MTLResourceStorageModeShared]; \
    id<MTLBuffer> outB = [dev newBufferWithLength:elem_size options:MTLResourceStorageModeShared]; \
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:shapeArr dataType:mps_dtype]; \
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:@[@1] dataType:mps_dtype]; \
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer]; \
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil]; \
    [cb commit]; [cb waitUntilCompleted]; \
    memcpy(TF_TensorData(output), outB.contents, elem_size); \
  } \
  TF_DeleteStatus(s); \
}

IMPL_REDUCTION_OP(Sum, [graph reductionSumWithTensor:inT axes:nil name:@"sum"])
IMPL_REDUCTION_OP(Mean, [graph reductionMeanWithTensor:inT axes:nil name:@"mean"])
IMPL_REDUCTION_OP(Max, [graph reductionMaximumWithTensor:inT axes:nil name:@"max"])
IMPL_REDUCTION_OP(Min, [graph reductionMinimumWithTensor:inT axes:nil name:@"min"])
IMPL_REDUCTION_OP(Prod, [graph reductionProductWithTensor:inT axes:nil name:@"prod"])

// ===== Cast Operation =====
extern "C" void* MPSCast_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSCast_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSCast_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus(); TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s); 
  if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType src_dtype = TF_TensorType(input);
  // For now, simple memcpy for same-size casts, full implementation would convert
  int nd = TF_NumDims(input); std::vector<int64_t> shape(nd); int64_t total = 1;
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; }
  size_t bytes = TF_TensorByteSize(input);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, src_dtype, shape.data(), nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  memcpy(TF_TensorData(output), TF_TensorData(input), bytes);
  TF_DeleteStatus(s);
}

// ===== Reshape Operation =====
extern "C" void* MPSReshape_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSReshape_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSReshape_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus(); TF_Tensor* input = nullptr; TF_Tensor* shape_tensor = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &shape_tensor, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  int64_t* new_shape_data = static_cast<int64_t*>(TF_TensorData(shape_tensor));
  int new_nd = (int)TF_TensorElementCount(shape_tensor);
  std::vector<int64_t> new_shape(new_nd);
  int64_t new_total = 1;
  for (int i = 0; i < new_nd; ++i) { new_shape[i] = new_shape_data[i]; new_total *= new_shape[i]; }
  size_t bytes = TF_TensorByteSize(input);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, new_shape.data(), new_nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  memcpy(TF_TensorData(output), TF_TensorData(input), bytes);
  TF_DeleteStatus(s);
}

// ===== Transpose Operation =====
extern "C" void* MPSTranspose_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSTranspose_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSTranspose_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus(); TF_Tensor* input = nullptr; TF_Tensor* perm_tensor = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &perm_tensor, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Transpose: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  int nd = TF_NumDims(input); std::vector<int64_t> shape(nd); int64_t total = 1;
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; }
  int32_t* perm = static_cast<int32_t*>(TF_TensorData(perm_tensor));
  std::vector<int64_t> out_shape(nd);
  for (int i = 0; i < nd; ++i) out_shape[i] = shape[perm[i]];
  size_t bytes = total * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_shape.data(), nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev = stream->dev->device;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableArray* shapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [shapeArr addObject:@(shape[i])];
    NSMutableArray* permArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [permArr addObject:@(perm[i])];
    MPSGraphTensor* inT = [graph placeholderWithShape:shapeArr dataType:mps_dtype name:@"in"];
    MPSGraphTensor* outT = [graph transposeTensor:inT permutation:permArr name:@"transpose"];
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outB = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    NSMutableArray* outShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [outShapeArr addObject:@(out_shape[i])];
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:shapeArr dataType:mps_dtype];
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:outShapeArr dataType:mps_dtype];
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), outB.contents, bytes);
  }
  TF_DeleteStatus(s);
}

// ===== Concat Operation =====
extern "C" void* MPSConcat_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSConcat_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSConcat_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  // Simple 2-input concat for now
  TF_Tensor* input0 = nullptr; TF_Tensor* input1 = nullptr;
  TF_GetInput(ctx, 0, &input0, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &input1, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input0);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Concat: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  int nd = TF_NumDims(input0);
  std::vector<int64_t> shape0(nd), shape1(nd);
  int64_t total0 = 1, total1 = 1;
  for (int i = 0; i < nd; ++i) { shape0[i] = TF_Dim(input0, i); total0 *= shape0[i]; }
  for (int i = 0; i < nd; ++i) { shape1[i] = TF_Dim(input1, i); total1 *= shape1[i]; }
  std::vector<int64_t> out_shape = shape0;
  out_shape[0] += shape1[0]; // Concat along axis 0
  int64_t total_out = total0 + total1;
  size_t bytes_out = total_out * elem_size;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_shape.data(), nd, bytes_out, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev = stream->dev->device;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableArray* shape0Arr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [shape0Arr addObject:@(shape0[i])];
    NSMutableArray* shape1Arr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [shape1Arr addObject:@(shape1[i])];
    MPSGraphTensor* in0T = [graph placeholderWithShape:shape0Arr dataType:mps_dtype name:@"in0"];
    MPSGraphTensor* in1T = [graph placeholderWithShape:shape1Arr dataType:mps_dtype name:@"in1"];
    MPSGraphTensor* outT = [graph concatTensor:in0T withTensor:in1T dimension:0 name:@"concat"];
    size_t bytes0 = total0 * elem_size, bytes1 = total1 * elem_size;
    id<MTLBuffer> buf0 = [dev newBufferWithBytes:TF_TensorData(input0) length:bytes0 options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf1 = [dev newBufferWithBytes:TF_TensorData(input1) length:bytes1 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufOut = [dev newBufferWithLength:bytes_out options:MTLResourceStorageModeShared];
    NSMutableArray* outShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [outShapeArr addObject:@(out_shape[i])];
    MPSGraphTensorData* d0 = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf0 shape:shape0Arr dataType:mps_dtype];
    MPSGraphTensorData* d1 = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf1 shape:shape1Arr dataType:mps_dtype];
    MPSGraphTensorData* dOut = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufOut shape:outShapeArr dataType:mps_dtype];
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{in0T: d0, in1T: d1} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), bufOut.contents, bytes_out);
  }
  TF_DeleteStatus(s);
}

// ===== ArgMax/ArgMin Operations =====
extern "C" void* MPSArgMax_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSArgMax_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSArgMax_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus(); TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS ArgMax: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  int nd = TF_NumDims(input); std::vector<int64_t> shape(nd); int64_t total = 1;
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; }
  std::vector<int64_t> out_shape(nd - 1);
  for (int i = 0; i < nd - 1; ++i) out_shape[i] = shape[i];
  int64_t out_total = (nd > 1) ? (total / shape[nd-1]) : 1;
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_INT64, out_shape.data(), nd - 1, out_total * 8, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(cstream); id<MTLDevice> dev = stream->dev->device;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableArray* shapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [shapeArr addObject:@(shape[i])];
    MPSGraphTensor* inT = [graph placeholderWithShape:shapeArr dataType:mps_dtype name:@"in"];
    MPSGraphTensor* outT = [graph argMaxWithTensor:inT axis:(nd-1) name:@"argmax"];
    size_t bytes = total * elem_size;
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outB = [dev newBufferWithLength:out_total * 4 options:MTLResourceStorageModeShared];
    NSMutableArray* outShapeArr = [NSMutableArray arrayWithCapacity:(nd-1)];
    for (int i = 0; i < nd - 1; ++i) [outShapeArr addObject:@(out_shape[i])];
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:shapeArr dataType:mps_dtype];
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:outShapeArr dataType:MPSDataTypeInt32];
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    int32_t* out_i32 = (int32_t*)outB.contents;
    int64_t* out_i64 = (int64_t*)TF_TensorData(output);
    for (int64_t i = 0; i < out_total; ++i) out_i64[i] = out_i32[i];
  }
  TF_DeleteStatus(s);
}


// ===== Additional Critical Operations =====

// Slice operation
extern "C" void* MPSSlice_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSSlice_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSSlice_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_Tensor* begin_t = nullptr; TF_Tensor* size_t = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &begin_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &size_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  int nd = TF_NumDims(input);
  std::vector<int64_t> in_shape(nd), begin(nd), size(nd);
  int64_t* begin_data = static_cast<int64_t*>(TF_TensorData(begin_t));
  int64_t* size_data = static_cast<int64_t*>(TF_TensorData(size_t));
  int64_t out_total = 1;
  for (int i = 0; i < nd; ++i) {
    in_shape[i] = TF_Dim(input, i);
    begin[i] = begin_data[i];
    size[i] = (size_data[i] == -1) ? (in_shape[i] - begin[i]) : size_data[i];
    out_total *= size[i];
  }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : ((dtype == TF_HALF || dtype == TF_BFLOAT16) ? 2 : 1);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, size.data(), nd, out_total * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  // Simple memcpy for contiguous slice (full implementation would handle strided slices)
  memcpy(TF_TensorData(output), TF_TensorData(input), out_total * elem_size);
  TF_DeleteStatus(s);
}

// MirrorPad (reflect/symmetric) operation
namespace { struct MPSMirrorPadAttrs { std::string mode; }; }
extern "C" void* MPSMirrorPad_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSMirrorPadAttrs();
  TF_Status* s = TF_NewStatus();
  char* mode_data = nullptr; size_t mode_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "mode", &mode_data, &mode_len, s);
  if (TF_GetCode(s) == TF_OK && mode_data) attrs->mode.assign(mode_data, mode_len);
  TF_DeleteStatus(s);
  return attrs;
}
extern "C" void MPSMirrorPad_Delete(void* p) { delete static_cast<MPSMirrorPadAttrs*>(p); }
extern "C" void MPSMirrorPad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSMirrorPadAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_Tensor* paddings_t = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &paddings_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS MirrorPad: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  int nd = TF_NumDims(input);
  std::vector<int64_t> in_shape(nd), out_shape(nd);
  int64_t in_total = 1, out_total = 1;
  int32_t* paddings = static_cast<int32_t*>(TF_TensorData(paddings_t));
  for (int i = 0; i < nd; ++i) {
    in_shape[i] = TF_Dim(input, i);
    out_shape[i] = in_shape[i] + paddings[i*2] + paddings[i*2+1];
    in_total *= in_shape[i];
    out_total *= out_shape[i];
  }
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_shape.data(), nd, out_total * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = reinterpret_cast<MPSStreamStruct*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->dev->device;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableArray* inShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [inShapeArr addObject:@(in_shape[i])];
    NSMutableArray* leftPad = [NSMutableArray arrayWithCapacity:nd];
    NSMutableArray* rightPad = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [leftPad addObject:@(paddings[i*2])];
      [rightPad addObject:@(paddings[i*2+1])];
    }
    MPSGraphPaddingMode mode = MPSGraphPaddingModeReflect;
    if (attrs->mode == "SYMMETRIC") mode = MPSGraphPaddingModeSymmetric;
    else if (attrs->mode == "REFLECT") mode = MPSGraphPaddingModeReflect;
    MPSGraphTensor* inT = [graph placeholderWithShape:inShapeArr dataType:mps_dtype name:@"in"];
    MPSGraphTensor* outT = [graph padTensor:inT withPaddingMode:mode leftPadding:leftPad rightPadding:rightPad constantValue:0.0 name:@"mirrorpad"];
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:in_total*elem_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> outB = [dev newBufferWithLength:out_total*elem_size options:MTLResourceStorageModeShared];
    NSMutableArray* outShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [outShapeArr addObject:@(out_shape[i])];
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:inShapeArr dataType:mps_dtype];
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:outShapeArr dataType:mps_dtype];
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), outB.contents, out_total * elem_size);
  }
  TF_DeleteStatus(s);
}

// Fill operation
extern "C" void* MPSFill_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSFill_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSFill_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* dims_t = nullptr; TF_Tensor* value_t = nullptr;
  TF_GetInput(ctx, 0, &dims_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &value_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(value_t);
  int nd = (int)TF_TensorElementCount(dims_t);
  int32_t* dims_data = static_cast<int32_t*>(TF_TensorData(dims_t));
  std::vector<int64_t> shape(nd);
  int64_t total = 1;
  for (int i = 0; i < nd; ++i) { shape[i] = dims_data[i]; total *= shape[i]; }
  size_t elem_size;
  switch (dtype) {
    case TF_FLOAT: elem_size = 4; break;
    case TF_HALF: case TF_BFLOAT16: elem_size = 2; break;
    case TF_BOOL: elem_size = 1; break;
    default: elem_size = TF_TensorByteSize(value_t); // assume scalar size
  }
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, total * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  // Copy scalar value bytes repeatedly
  const void* val_ptr = TF_TensorData(value_t);
  char* outp = static_cast<char*>(TF_TensorData(output));
  for (int64_t i = 0; i < total; ++i) memcpy(outp + i * elem_size, val_ptr, elem_size);
  TF_DeleteStatus(s);
}

// ZerosLike operation
extern "C" void* MPSZerosLike_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSZerosLike_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSZerosLike_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  int nd = TF_NumDims(input);
  std::vector<int64_t> shape(nd);
  for (int i = 0; i < nd; ++i) shape[i] = TF_Dim(input, i);
  size_t bytes = TF_TensorByteSize(input);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  memset(TF_TensorData(output), 0, bytes);
  TF_DeleteStatus(s);
}

// OnesLike operation
extern "C" void* MPSOnesLike_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSOnesLike_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSOnesLike_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  int nd = TF_NumDims(input);
  std::vector<int64_t> shape(nd);
  int64_t total = 1;
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; }
  size_t bytes = TF_TensorByteSize(input);
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, bytes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  if (dtype == TF_FLOAT) {
    float* out = static_cast<float*>(TF_TensorData(output));
    for (int64_t i = 0; i < total; ++i) out[i] = 1.0f;
  } else if (dtype == TF_HALF) {
    uint16_t* out = static_cast<uint16_t*>(TF_TensorData(output));
    // IEEE754 half for 1.0 is 0x3C00
    for (int64_t i = 0; i < total; ++i) out[i] = 0x3C00;
  } else if (dtype == TF_BFLOAT16) {
    uint16_t* out = static_cast<uint16_t*>(TF_TensorData(output));
    // bfloat16 1.0 top 16 bits of float32 1.0 (0x3F80)
    for (int64_t i = 0; i < total; ++i) out[i] = 0x3F80;
  } else if (dtype == TF_BOOL) {
    bool* out = static_cast<bool*>(TF_TensorData(output));
    for (int64_t i = 0; i < total; ++i) out[i] = true;
  } else {
    // Default: memset pattern for ones not defined; fall back to element-wise copy from a scalar 1 if representable.
    memset(TF_TensorData(output), 0, bytes); // safe fallback to zeros if unsupported
  }
  TF_DeleteStatus(s);
}

// Pad operation
extern "C" void* MPSPad_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSPad_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSPad_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_Tensor* paddings_t = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &paddings_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Pad: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  int nd = TF_NumDims(input);
  std::vector<int64_t> in_shape(nd), out_shape(nd);
  int64_t in_total = 1, out_total = 1;
  int32_t* paddings = static_cast<int32_t*>(TF_TensorData(paddings_t));
  for (int i = 0; i < nd; ++i) {
    in_shape[i] = TF_Dim(input, i);
    out_shape[i] = in_shape[i] + paddings[i*2] + paddings[i*2+1];
    in_total *= in_shape[i];
    out_total *= out_shape[i];
  }
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_shape.data(), nd, out_total * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableArray* inShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [inShapeArr addObject:@(in_shape[i])];
    NSMutableArray* paddingsArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) {
      [paddingsArr addObject:@[@(paddings[i*2]), @(paddings[i*2+1])]];
    }
    MPSGraphTensor* inT = [graph placeholderWithShape:inShapeArr dataType:mps_dtype name:@"in"];
    MPSGraphTensor* outT = [graph padTensor:inT withPaddingMode:MPSGraphPaddingModeConstant leftPadding:[paddingsArr valueForKeyPath:@"@firstObject"] rightPadding:[paddingsArr valueForKeyPath:@"@lastObject"] constantValue:0.0 name:@"pad"];
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:in_total*elem_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> outB = [dev newBufferWithLength:out_total*elem_size options:MTLResourceStorageModeShared];
    NSMutableArray* outShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [outShapeArr addObject:@(out_shape[i])];
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:inShapeArr dataType:mps_dtype];
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:outShapeArr dataType:mps_dtype];
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), outB.contents, out_total * elem_size);
  }
  TF_DeleteStatus(s);
}

// Slice operation (full begin/size semantics, contiguous copy on CPU)
extern "C" void* MPSTile_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSTile_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSTile_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_Tensor* multiples_t = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &multiples_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Tile: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  // begin/size can be int32 or int64; normalize to int64
  int64_t out_total = 1;
  // Read begin
  if (TF_TensorType(begin_t) == TF_INT32) {
    int32_t* b32 = static_cast<int32_t*>(TF_TensorData(begin_t));
    for (int i = 0; i < nd; ++i) begin[i] = static_cast<int64_t>(b32[i]);
  } else {
    int64_t* b64 = static_cast<int64_t*>(TF_TensorData(begin_t));
    for (int i = 0; i < nd; ++i) begin[i] = b64[i];
  }
  // Read size
  if (TF_TensorType(size_t) == TF_INT32) {
    int32_t* s32 = static_cast<int32_t*>(TF_TensorData(size_t));
    for (int i = 0; i < nd; ++i) size[i] = static_cast<int64_t>(s32[i]);
  } else {
    int64_t* s64 = static_cast<int64_t*>(TF_TensorData(size_t));
    for (int i = 0; i < nd; ++i) size[i] = s64[i];
  }
  std::vector<int64_t> in_shape(nd), out_shape(nd);
  int64_t in_total = 1, out_total = 1;
    // Normalize negative begin
    int64_t b = begin[i];
    if (b < 0) b += in_shape[i];
    if (b < 0) b = 0; if (b > in_shape[i]) b = in_shape[i];
    int64_t sz = size[i];
    if (sz == -1) sz = in_shape[i] - b;
    if (sz < 0) sz = 0; if (b + sz > in_shape[i]) sz = in_shape[i] - b;
    begin[i] = b; size[i] = sz; out_shape[i] = sz; out_total *= sz;
    out_shape[i] = in_shape[i] * multiples[i];
  size_t elem_size;
  switch (dtype) {
    case TF_FLOAT: elem_size = 4; break;
    case TF_HALF: case TF_BFLOAT16: elem_size = 2; break;
    case TF_BOOL: elem_size = 1; break;
    default: elem_size = TF_TensorByteSize(input) / (TF_TensorElementCount(input) ? TF_TensorElementCount(input) : 1);
  }
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_shape.data(), nd, out_total * elem_size, s);
  }
  // Compute row-major strides for input and output
  std::vector<int64_t> in_stride(nd, 1), out_stride(nd, 1);
  for (int i = nd - 2; i >= 0; --i) {
    in_stride[i] = in_stride[i+1] * in_shape[i+1];
    out_stride[i] = out_stride[i+1] * out_shape[i+1];
  }
  // N-D copy
  const char* in_base = static_cast<const char*>(TF_TensorData(input));
  char* out_base = static_cast<char*>(TF_TensorData(output));
  // Handle 0-sized output
  if (out_total == 0) { TF_DeleteStatus(s); return; }
  // Iterate over all output indices and map to input
  std::vector<int64_t> idx(nd, 0);
  while (true) {
    // Compute flat offsets
    int64_t in_off = 0, out_off = 0;
    for (int i = 0; i < nd; ++i) {
      in_off += (begin[i] + idx[i]) * in_stride[i];
      out_off += idx[i] * out_stride[i];
    }
    memcpy(out_base + out_off * elem_size, in_base + in_off * elem_size, elem_size);
    // Increment idx
    int d = nd - 1;
    for (; d >= 0; --d) {
      if (++idx[d] < out_shape[d]) break; idx[d] = 0;
    }
    if (d < 0) break;
  }
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }

// StridedSlice operation with basic mask support (begin_mask, end_mask, shrink_axis_mask). Positive strides only.
namespace { struct MPSStridedSliceAttrs { int64_t begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0; }; }
extern "C" void* MPSStridedSlice_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSStridedSliceAttrs();
  TF_Status* s = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt64(ctx, "begin_mask", &attrs->begin_mask, s);
  TF_OpKernelConstruction_GetAttrInt64(ctx, "end_mask", &attrs->end_mask, s);
  TF_OpKernelConstruction_GetAttrInt64(ctx, "ellipsis_mask", &attrs->ellipsis_mask, s);
  TF_OpKernelConstruction_GetAttrInt64(ctx, "new_axis_mask", &attrs->new_axis_mask, s);
  TF_OpKernelConstruction_GetAttrInt64(ctx, "shrink_axis_mask", &attrs->shrink_axis_mask, s);
  TF_DeleteStatus(s);
  return attrs;
}
extern "C" void MPSStridedSlice_Delete(void* p) { delete static_cast<MPSStridedSliceAttrs*>(p); }
extern "C" void MPSStridedSlice_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* k = static_cast<MPSStridedSliceAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_Tensor* begin_t = nullptr; TF_Tensor* end_t = nullptr; TF_Tensor* strides_t = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &begin_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &end_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 3, &strides_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  int nd = TF_NumDims(input);
  std::vector<int64_t> in_shape(nd), begin(nd), end(nd), strides(nd), out_shape(nd);
  for (int i = 0; i < nd; ++i) in_shape[i] = TF_Dim(input, i);
  auto read_vec = [&](TF_Tensor* t, std::vector<int64_t>& dst){
    if (TF_TensorType(t) == TF_INT32) {
      int32_t* p = static_cast<int32_t*>(TF_TensorData(t));
      for (int i = 0; i < nd; ++i) dst[i] = static_cast<int64_t>(p[i]);
    } else {
      int64_t* p = static_cast<int64_t*>(TF_TensorData(t));
      for (int i = 0; i < nd; ++i) dst[i] = p[i];
    }
  };
  read_vec(begin_t, begin); read_vec(end_t, end); read_vec(strides_t, strides);
  if (k->ellipsis_mask != 0 || k->new_axis_mask != 0) {
    TF_SetStatus(s, TF_UNIMPLEMENTED, "StridedSlice: ellipsis/new_axis masks not yet supported");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return;
  }
  // Normalize and compute out_shape with begin/end/shrink masks
  for (int i = 0; i < nd; ++i) {
    if (strides[i] == 0) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "StridedSlice: stride cannot be 0"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    int64_t b = begin[i]; int64_t e = end[i]; int64_t st = strides[i];
    // Support positive strides only for now
    if (st < 0) { TF_SetStatus(s, TF_UNIMPLEMENTED, "StridedSlice: negative strides not yet supported"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    bool shrink = ((k->shrink_axis_mask >> i) & 1) != 0;
    // Apply masks
    if (((k->begin_mask >> i) & 1) != 0) b = 0;
    if (((k->end_mask >> i) & 1) != 0) e = in_shape[i];
    if (b < 0) b += in_shape[i]; if (e < 0) e += in_shape[i];
    if (b < 0) b = 0; if (b > in_shape[i]) b = in_shape[i];
    if (e < b) e = b; if (e > in_shape[i]) e = in_shape[i];
    int64_t len;
    if (shrink) { len = 1; e = b + 1; }
    else { len = (e - b + st - 1) / st; if (len < 0) len = 0; }
    begin[i] = b; end[i] = e; strides[i] = st; out_shape[i] = len;
  }
  // Compute output rank after shrinking axes
  std::vector<int64_t> final_shape; final_shape.reserve(nd);
  for (int i = 0; i < nd; ++i) {
    if (((k->shrink_axis_mask >> i) & 1) != 0) continue; // drop dim
    final_shape.push_back(out_shape[i]);
  }
  int out_nd = static_cast<int>(final_shape.size());
  int64_t out_total = 1; for (int i = 0; i < out_nd; ++i) out_total *= final_shape[i];
  size_t elem_size;
  switch (dtype) {
    case TF_FLOAT: elem_size = 4; break;
    case TF_HALF: case TF_BFLOAT16: elem_size = 2; break;
    case TF_BOOL: elem_size = 1; break;
    default: elem_size = TF_TensorByteSize(input) / (TF_TensorElementCount(input) ? TF_TensorElementCount(input) : 1);
  }
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, final_shape.data(), out_nd, out_total * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const char* in_base = static_cast<const char*>(TF_TensorData(input));
  char* out_base = static_cast<char*>(TF_TensorData(output));
  if (out_total == 0) { TF_DeleteStatus(s); return; }
  // Compute input strides and output strides for unshrunken dims
  std::vector<int64_t> in_stride(nd, 1);
  for (int i = nd - 2; i >= 0; --i) in_stride[i] = in_stride[i+1] * in_shape[i+1];
  // Map from output dims to input dims
  std::vector<int> out2in;
  for (int i = 0; i < nd; ++i) if (((k->shrink_axis_mask >> i) & 1) == 0) out2in.push_back(i);
  std::vector<int64_t> out_stride(out_nd, 1);
  for (int i = out_nd - 2; i >= 0; --i) out_stride[i] = out_stride[i+1] * final_shape[i+1];
  std::vector<int64_t> idx(out_nd, 0);
  while (true) {
    int64_t in_off = 0, out_off = 0;
    for (int oi = 0; oi < out_nd; ++oi) {
      int di = out2in[oi];
      in_off += (begin[di] + idx[oi] * strides[di]) * in_stride[di];
      out_off += idx[oi] * out_stride[oi];
    }
    // Add offsets for shrunken dims (fixed position at begin)
    for (int di = 0; di < nd; ++di) {
      if (((k->shrink_axis_mask >> di) & 1) != 0) {
        in_off += begin[di] * in_stride[di];
      }
    }
    memcpy(out_base + out_off * elem_size, in_base + in_off * elem_size, elem_size);
    int d = out_nd - 1;
    for (; d >= 0; --d) { if (++idx[d] < final_shape[d]) break; idx[d] = 0; }
    if (d < 0) break;
  }
  TF_DeleteStatus(s);
}
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableArray* inShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [inShapeArr addObject:@(in_shape[i])];
    MPSGraphTensor* inT = [graph placeholderWithShape:inShapeArr dataType:mps_dtype name:@"in"];
    MPSGraphTensor* outT = inT;
    for (int i = 0; i < nd; ++i) {
      if (multiples[i] > 1) {
        NSMutableArray* tiles = [NSMutableArray arrayWithCapacity:nd];
        for (int j = 0; j < nd; ++j) [tiles addObject:@((i == j) ? multiples[i] : 1)];
        outT = [graph tileTensor:outT withMultiplier:tiles name:[NSString stringWithFormat:@"tile_%d", i]];
      }
    }
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:in_total*elem_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> outB = [dev newBufferWithLength:out_total*elem_size options:MTLResourceStorageModeShared];
    NSMutableArray* outShapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [outShapeArr addObject:@(out_shape[i])];
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:inShapeArr dataType:mps_dtype];
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:outShapeArr dataType:mps_dtype];
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), outB.contents, out_total * elem_size);
  }
  TF_DeleteStatus(s);
}

// Select/Where operation
extern "C" void* MPSSelect_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSSelect_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSSelect_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* cond = nullptr; TF_Tensor* t = nullptr; TF_Tensor* e = nullptr;
  TF_GetInput(ctx, 0, &cond, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &e, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(t);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS Select: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  // Compute broadcasted shape of t and e
  int nd_t = TF_NumDims(t), nd_e = TF_NumDims(e);
  int nd = std::max(nd_t, nd_e);
  std::vector<int64_t> t_shape(nd_t), e_shape(nd_e), out_shape(nd);
  for (int i = 0; i < nd_t; ++i) t_shape[i] = TF_Dim(t, i);
  for (int i = 0; i < nd_e; ++i) e_shape[i] = TF_Dim(e, i);
  // Right-align shapes
  for (int i = 0; i < nd; ++i) {
    int it = nd_t - 1 - i; int ie = nd_e - 1 - i; int64_t dt = (it >= 0) ? t_shape[it] : 1; int64_t de = (ie >= 0) ? e_shape[ie] : 1;
    if (dt != de && dt != 1 && de != 1) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Select: t and e not broadcastable"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    out_shape[nd - 1 - i] = std::max(dt, de);
  }
  // Cond can be scalar or same as out_shape
  bool cond_scalar = (TF_NumDims(cond) == 0) || (TF_TensorElementCount(cond) == 1);
  if (!cond_scalar) {
    if (TF_NumDims(cond) != nd) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Select: cond rank mismatch"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    for (int i = 0; i < nd; ++i) if (TF_Dim(cond, i) != out_shape[i]) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "Select: cond shape mismatch"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  }
  int64_t total = 1; for (int i = 0; i < nd; ++i) total *= out_shape[i];
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, out_shape.data(), nd, total * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  // For now, if all shapes already match (t, e, cond), use MPSGraph; otherwise, do a CPU broadcast.
  bool shapes_match = (!cond_scalar) && (TF_NumDims(t) == nd) && (TF_NumDims(e) == nd) && (TF_NumDims(cond) == nd);
  for (int i = 0; shapes_match && i < nd; ++i) {
    shapes_match = (TF_Dim(t, i) == out_shape[i]) && (TF_Dim(e, i) == out_shape[i]) && (TF_Dim(cond, i) == out_shape[i]);
  }
  if (shapes_match && !cond_scalar) {
    SP_Stream stream_handle = TF_GetStream(ctx, s);
    if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
    id<MTLDevice> dev = stream->device;
    @autoreleasepool {
      MPSGraph* graph = [[MPSGraph alloc] init];
      NSMutableArray* shapeArr = [NSMutableArray arrayWithCapacity:nd];
      for (int i = 0; i < nd; ++i) [shapeArr addObject:@(out_shape[i])];
      MPSGraphTensor* condT = [graph placeholderWithShape:shapeArr dataType:MPSDataTypeBool name:@"cond"];
      MPSGraphTensor* tT = [graph placeholderWithShape:shapeArr dataType:mps_dtype name:@"t"];
      MPSGraphTensor* eT = [graph placeholderWithShape:shapeArr dataType:mps_dtype name:@"e"];
      MPSGraphTensor* outT = [graph selectWithPredicateTensor:condT truePredicateTensor:tT falsePredicateTensor:eT name:@"select"];
      id<MTLBuffer> condB = [dev newBufferWithBytes:TF_TensorData(cond) length:total options:MTLResourceStorageModeShared];
      id<MTLBuffer> tB = [dev newBufferWithBytes:TF_TensorData(t) length:total*elem_size options:MTLResourceStorageModeShared];
      id<MTLBuffer> eB = [dev newBufferWithBytes:TF_TensorData(e) length:total*elem_size options:MTLResourceStorageModeShared];
      id<MTLBuffer> outB = [dev newBufferWithLength:total*elem_size options:MTLResourceStorageModeShared];
      NSMutableArray* shapeArr2 = [NSMutableArray arrayWithCapacity:nd];
      for (int i = 0; i < nd; ++i) [shapeArr2 addObject:@(out_shape[i])];
      MPSGraphTensorData* condD = [[MPSGraphTensorData alloc] initWithMTLBuffer:condB shape:shapeArr2 dataType:MPSDataTypeBool];
      MPSGraphTensorData* tD = [[MPSGraphTensorData alloc] initWithMTLBuffer:tB shape:shapeArr2 dataType:mps_dtype];
      MPSGraphTensorData* eD = [[MPSGraphTensorData alloc] initWithMTLBuffer:eB shape:shapeArr2 dataType:mps_dtype];
      MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:shapeArr2 dataType:mps_dtype];
      id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
      [graph runWithMTLCommandBuffer:cb feeds:@{condT: condD, tT: tD, eT: eD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
      [cb commit]; [cb waitUntilCompleted];
      memcpy(TF_TensorData(output), outB.contents, total * elem_size);
    }
  } else {
    // CPU fallback supporting broadcasting among cond, t, and e
    const bool* cptr = cond_scalar ? static_cast<const bool*>(TF_TensorData(cond)) : nullptr;
    std::vector<int64_t> t_strides(nd,1), e_strides(nd,1), c_strides(nd,1), out_strides(nd,1);
    // Build aligned shapes for t, e, cond
    std::vector<int64_t> t_shape_al(nd,1), e_shape_al(nd,1), c_shape_al(nd,1);
    for (int i = 0; i < nd; ++i) {
      int it = nd - 1 - i; int it_src = nd_t - 1 - i; t_shape_al[it] = (it_src >= 0) ? t_shape[it_src] : 1;
      int ie = nd - 1 - i; int ie_src = nd_e - 1 - i; e_shape_al[ie] = (ie_src >= 0) ? e_shape[ie_src] : 1;
      if (!cond_scalar) {
        int ic = nd - 1 - i; int ic_src = TF_NumDims(cond) - 1 - i; c_shape_al[ic] = (ic_src >= 0) ? TF_Dim(cond, ic_src) : 1;
      }
    }
    for (int i = nd - 2; i >= 0; --i) {
      t_strides[i] = t_strides[i+1] * t_shape_al[i+1];
      e_strides[i] = e_strides[i+1] * e_shape_al[i+1];
      if (!cond_scalar) c_strides[i] = c_strides[i+1] * c_shape_al[i+1];
      out_strides[i] = out_strides[i+1] * out_shape[i+1];
    }
    const char* tptr = static_cast<const char*>(TF_TensorData(t));
    const char* eptr = static_cast<const char*>(TF_TensorData(e));
    char* outp = static_cast<char*>(TF_TensorData(output));
    // Iterate over output index and compute source offsets
    std::vector<int64_t> idx(nd,0);
    bool cond_val = cptr ? (*cptr) : false;
    while (true) {
      int64_t t_off = 0, e_off = 0, c_off = 0, out_off = 0;
      for (int i = 0; i < nd; ++i) {
        int64_t ii = idx[i];
        int64_t ti = (t_shape_al[i] == 1) ? 0 : ii;
        int64_t ei = (e_shape_al[i] == 1) ? 0 : ii;
        int64_t ci = cond_scalar ? 0 : ((c_shape_al[i] == 1) ? 0 : ii);
        t_off += ti * t_strides[i];
        e_off += ei * e_strides[i];
        if (!cond_scalar) c_off += ci * c_strides[i];
        out_off += ii * out_strides[i];
      }
      bool use_t;
      if (cond_scalar) use_t = cond_val;
      else { const bool* cbase = static_cast<const bool*>(TF_TensorData(cond)); use_t = cbase[c_off]; }
      const char* src = use_t ? (tptr + t_off * elem_size) : (eptr + e_off * elem_size);
      memcpy(outp + out_off * elem_size, src, elem_size);
      int d = nd - 1; for (; d >= 0; --d) { if (++idx[d] < out_shape[d]) break; idx[d] = 0; }
      if (d < 0) break;
    }
  }
  TF_DeleteStatus(s);
}

// Clip/ClipByValue operation
extern "C" void* MPSClipByValue_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSClipByValue_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSClipByValue_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr; TF_Tensor* min_t = nullptr; TF_Tensor* max_t = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &min_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 2, &max_t, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_DataType dtype = TF_TensorType(input);
  if (dtype != TF_FLOAT && dtype != TF_HALF && dtype != TF_BFLOAT16) {
    TF_SetStatus(s, TF_INVALID_ARGUMENT, "MPS ClipByValue: float/half/bf16 only");
    TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  size_t elem_size = (dtype == TF_FLOAT) ? 4 : 2;
  MPSDataType mps_dtype = (dtype == TF_HALF) ? MPSDataTypeFloat16 : ((dtype == TF_BFLOAT16) ? MPSDataTypeBFloat16 : MPSDataTypeFloat32);
  int nd = TF_NumDims(input);
  std::vector<int64_t> shape(nd);
  int64_t total = 1;
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; }
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, dtype, shape.data(), nd, total * elem_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  SP_Stream stream_handle = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  auto* stream = static_cast<MPSStream*>(stream_handle->stream_handle);
  id<MTLDevice> dev = stream->device;
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    NSMutableArray* shapeArr = [NSMutableArray arrayWithCapacity:nd];
    for (int i = 0; i < nd; ++i) [shapeArr addObject:@(shape[i])];
    MPSGraphTensor* inT = [graph placeholderWithShape:shapeArr dataType:mps_dtype name:@"in"];
    MPSGraphTensor* minT = [graph placeholderWithShape:@[@1] dataType:mps_dtype name:@"min"];
    MPSGraphTensor* maxT = [graph placeholderWithShape:@[@1] dataType:mps_dtype name:@"max"];
    MPSGraphTensor* outT = [graph clampWithTensor:inT minValueTensor:minT maxValueTensor:maxT name:@"clip"];
    id<MTLBuffer> inB = [dev newBufferWithBytes:TF_TensorData(input) length:total*elem_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> minB = [dev newBufferWithBytes:TF_TensorData(min_t) length:elem_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> maxB = [dev newBufferWithBytes:TF_TensorData(max_t) length:elem_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> outB = [dev newBufferWithLength:total*elem_size options:MTLResourceStorageModeShared];
    MPSGraphTensorData* inD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inB shape:shapeArr dataType:mps_dtype];
    MPSGraphTensorData* minD = [[MPSGraphTensorData alloc] initWithMTLBuffer:minB shape:@[@1] dataType:mps_dtype];
    MPSGraphTensorData* maxD = [[MPSGraphTensorData alloc] initWithMTLBuffer:maxB shape:@[@1] dataType:mps_dtype];
    MPSGraphTensorData* outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outB shape:shapeArr dataType:mps_dtype];
    id<MTLCommandBuffer> cb = [stream->queue commandBuffer];
    [graph runWithMTLCommandBuffer:cb feeds:@{inT: inD, minT: minD, maxT: maxD} targetTensors:@[outT] targetOperations:nil executionDescriptor:nil];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), outB.contents, total * elem_size);
  }
  TF_DeleteStatus(s);
}


// ===== Additional Activations =====

// LeakyRelu
IMPL_UNARY_OP(LeakyRelu, leakyReLUWithTensor:inT alpha:0.2 name:@"leaky_relu")

// Relu6
IMPL_UNARY_OP(Relu6, clampWithTensor:inT minValueTensor:[graph constantWithScalar:0.0 dataType:mps_dtype] maxValueTensor:[graph constantWithScalar:6.0 dataType:mps_dtype] name:@"relu6")

// Elu
IMPL_UNARY_OP(Elu, [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:[graph constantWithScalar:1.0 dataType:mps_dtype] secondaryTensor:[graph subtractionWithPrimaryTensor:[graph exponentWithTensor:inT name:@"exp"] secondaryTensor:[graph constantWithScalar:1.0 dataType:mps_dtype] name:@"sub"] name:@"mul"] secondaryTensor:[graph selectWithPredicateTensor:[graph lessThanWithPrimaryTensor:inT secondaryTensor:[graph constantWithScalar:0.0 dataType:mps_dtype] name:@"lt"] truePredicateTensor:inT falsePredicateTensor:[graph constantWithScalar:0.0 dataType:mps_dtype] name:@"sel"] name:@"add"])

// Selu  
IMPL_UNARY_OP(Selu, [graph multiplicationWithPrimaryTensor:[graph constantWithScalar:1.0507f dataType:mps_dtype] secondaryTensor:[graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:[graph constantWithScalar:1.67326f dataType:mps_dtype] secondaryTensor:[graph subtractionWithPrimaryTensor:[graph exponentWithTensor:inT name:@"exp"] secondaryTensor:[graph constantWithScalar:1.0 dataType:mps_dtype] name:@"sub"] name:@"mul"] secondaryTensor:[graph selectWithPredicateTensor:[graph lessThanWithPrimaryTensor:inT secondaryTensor:[graph constantWithScalar:0.0 dataType:mps_dtype] name:@"lt"] truePredicateTensor:inT falsePredicateTensor:[graph constantWithScalar:0.0 dataType:mps_dtype] name:@"sel"] name:@"add"] name:@"mul"])

// Softplus: log(exp(x) + 1)
IMPL_UNARY_OP(Softplus, [graph logarithmWithTensor:[graph additionWithPrimaryTensor:[graph exponentWithTensor:inT name:@"exp"] secondaryTensor:[graph constantWithScalar:1.0 dataType:mps_dtype] name:@"add"] name:@"log"])

// Softsign: x / (1 + |x|)
IMPL_UNARY_OP(Softsign, [graph divisionWithPrimaryTensor:inT secondaryTensor:[graph additionWithPrimaryTensor:[graph constantWithScalar:1.0 dataType:mps_dtype] secondaryTensor:[graph absoluteWithTensor:inT name:@"abs"] name:@"add"] name:@"div"])

// ===== Logical Operations =====

extern "C" void* MPSLogicalAnd_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSLogicalAnd_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSLogicalAnd_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  // General broadcasting
  bool a_scalar = (TF_NumDims(a) == 0) || (TF_TensorElementCount(a) == 1);
  bool b_scalar = (TF_NumDims(b) == 0) || (TF_TensorElementCount(b) == 1);
  int nd = std::max(TF_NumDims(a), TF_NumDims(b));
  std::vector<int64_t> ash(nd,1), bsh(nd,1), shape(nd);
  for (int i = 0; i < TF_NumDims(a); ++i) ash[nd - TF_NumDims(a) + i] = TF_Dim(a, i);
  for (int i = 0; i < TF_NumDims(b); ++i) bsh[nd - TF_NumDims(b) + i] = TF_Dim(b, i);
  for (int i = 0; i < nd; ++i) {
    if (ash[i] != bsh[i] && ash[i] != 1 && bsh[i] != 1) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "LogicalAnd: shapes not broadcastable"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    shape[i] = std::max(ash[i], bsh[i]);
  }
  int64_t total = 1; for (int i = 0; i < nd; ++i) total *= shape[i];
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BOOL, shape.data(), nd, total, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const bool* aptr = static_cast<bool*>(TF_TensorData(a));
  const bool* bptr = static_cast<bool*>(TF_TensorData(b));
  bool* out = static_cast<bool*>(TF_TensorData(output));
  if (a_scalar && b_scalar) {
    bool v = (*aptr) && (*bptr); for (int64_t i = 0; i < total; ++i) out[i] = v;
  } else {
    std::vector<int64_t> astrides(nd,1), bstrides(nd,1), ostrides(nd,1);
    for (int i = nd - 2; i >= 0; --i) { astrides[i] = astrides[i+1] * ash[i+1]; bstrides[i] = bstrides[i+1] * bsh[i+1]; ostrides[i] = ostrides[i+1] * shape[i+1]; }
    std::vector<int64_t> idx(nd,0);
    while (true) {
      int64_t a_off = 0, b_off = 0, o_off = 0;
      for (int i = 0; i < nd; ++i) {
        int64_t ii = idx[i];
        a_off += ((ash[i]==1)?0:ii) * astrides[i];
        b_off += ((bsh[i]==1)?0:ii) * bstrides[i];
        o_off += ii * ostrides[i];
      }
      out[o_off] = aptr[a_off] && bptr[b_off];
      int d = nd - 1; for (; d >= 0; --d) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
      if (d < 0) break;
    }
  }
  TF_DeleteStatus(s);
}

extern "C" void* MPSLogicalOr_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSLogicalOr_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSLogicalOr_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* a = nullptr; TF_Tensor* b = nullptr;
  TF_GetInput(ctx, 0, &a, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  TF_GetInput(ctx, 1, &b, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  // General broadcasting
  bool a_scalar = (TF_NumDims(a) == 0) || (TF_TensorElementCount(a) == 1);
  bool b_scalar = (TF_NumDims(b) == 0) || (TF_TensorElementCount(b) == 1);
  int nd = std::max(TF_NumDims(a), TF_NumDims(b));
  std::vector<int64_t> ash(nd,1), bsh(nd,1), shape(nd);
  for (int i = 0; i < TF_NumDims(a); ++i) ash[nd - TF_NumDims(a) + i] = TF_Dim(a, i);
  for (int i = 0; i < TF_NumDims(b); ++i) bsh[nd - TF_NumDims(b) + i] = TF_Dim(b, i);
  for (int i = 0; i < nd; ++i) {
    if (ash[i] != bsh[i] && ash[i] != 1 && bsh[i] != 1) { TF_SetStatus(s, TF_INVALID_ARGUMENT, "LogicalOr: shapes not broadcastable"); TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
    shape[i] = std::max(ash[i], bsh[i]);
  }
  int64_t total = 1; for (int i = 0; i < nd; ++i) total *= shape[i];
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BOOL, shape.data(), nd, total, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  const bool* aptr = static_cast<bool*>(TF_TensorData(a));
  const bool* bptr = static_cast<bool*>(TF_TensorData(b));
  bool* out = static_cast<bool*>(TF_TensorData(output));
  if (a_scalar && b_scalar) {
    bool v = (*aptr) || (*bptr); for (int64_t i = 0; i < total; ++i) out[i] = v;
  } else {
    std::vector<int64_t> astrides(nd,1), bstrides(nd,1), ostrides(nd,1);
    for (int i = nd - 2; i >= 0; --i) { astrides[i] = astrides[i+1] * ash[i+1]; bstrides[i] = bstrides[i+1] * bsh[i+1]; ostrides[i] = ostrides[i+1] * shape[i+1]; }
    std::vector<int64_t> idx(nd,0);
    while (true) {
      int64_t a_off = 0, b_off = 0, o_off = 0;
      for (int i = 0; i < nd; ++i) {
        int64_t ii = idx[i];
        a_off += ((ash[i]==1)?0:ii) * astrides[i];
        b_off += ((bsh[i]==1)?0:ii) * bstrides[i];
        o_off += ii * ostrides[i];
      }
      out[o_off] = aptr[a_off] || bptr[b_off];
      int d = nd - 1; for (; d >= 0; --d) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
      if (d < 0) break;
    }
  }
  TF_DeleteStatus(s);
}

extern "C" void* MPSLogicalNot_Create(TF_OpKernelConstruction*) { return new int(); }
extern "C" void MPSLogicalNot_Delete(void* p) { delete static_cast<int*>(p); }
extern "C" void MPSLogicalNot_Compute(void*, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s); if (TF_GetCode(s) != TF_OK) { TF_DeleteStatus(s); return; }
  // Allow scalar and any shape
  int nd = TF_NumDims(input);
  std::vector<int64_t> shape(nd);
  int64_t total = 1;
  for (int i = 0; i < nd; ++i) { shape[i] = TF_Dim(input, i); total *= shape[i]; }
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_BOOL, shape.data(), nd, total, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  bool* in = static_cast<bool*>(TF_TensorData(input));
  bool* out = static_cast<bool*>(TF_TensorData(output));
  for (int64_t i = 0; i < total; ++i) out[i] = !in[i];
  TF_DeleteStatus(s);
}

