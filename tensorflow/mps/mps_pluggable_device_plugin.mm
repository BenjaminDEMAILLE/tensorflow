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

  extern void MPSMul_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* mul_kb = TF_NewKernelBuilder("Mul", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMul_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(mul_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMulFloat", mul_kb, status);

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

  extern void MPSMinimum_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* min_kb = TF_NewKernelBuilder("Minimum", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSMinimum_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(min_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSMinimumFloat", min_kb, status);

  // Register Sigmoid/Tanh (float)
  extern void MPSSigmoid_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* sig_kb = TF_NewKernelBuilder("Sigmoid", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSSigmoid_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(sig_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSSigmoidFloat", sig_kb, status);

  extern void MPSTanh_Compute(void*, TF_OpKernelContext*);
  TF_KernelBuilder* tanh_kb = TF_NewKernelBuilder("Tanh", kPlatformName,
                                                /*create*/ nullptr,
                                                /*compute*/ &MPSTanh_Compute,
                                                /*delete*/ nullptr);
  TF_KernelBuilder_TypeConstraint(tanh_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSTanhFloat", tanh_kb, status);

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

  // Try GPU first; if stream is unavailable or bf16, fall back to host.
  SP_Stream cstream = TF_GetStream(ctx, s);
  if (TF_GetCode(s) != TF_OK || cstream == nullptr || is_bf16) {
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
