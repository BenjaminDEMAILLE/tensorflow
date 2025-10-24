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
#import <mach/mach_time.h>

#include "tensorflow/c/tf_status.h"

using namespace tensorflow::mps;

namespace tensorflow {
namespace mps {

// Helper functions for status
inline void TfOk(TF_Status* s) { TF_SetStatus(s, TF_OK, ""); }
inline void TfUnimpl(TF_Status* s, const char* msg) {
  TF_SetStatus(s, TF_UNIMPLEMENTED, msg);
}

// ---- MPSStreamStruct Implementation ----
MPSStreamStruct::MPSStreamStruct(MPSDevice* d)
    : dev(d), queue([d->device newCommandQueue]), last_cb(nil) {}

MPSStreamStruct::~MPSStreamStruct() {
  queue = nil;
  last_cb = nil;
}

// ---- MPSEventStruct Implementation ----
MPSEventStruct::MPSEventStruct() : sema(dispatch_semaphore_create(0)) {}

MPSEventStruct::~MPSEventStruct() {
  sema = nullptr;
}

// ---- Platform Callbacks ----
void GetDeviceCount(const SP_Platform* /*platform*/, int* device_count,
                    TF_Status* status) {
  @autoreleasepool {
    NSArray<id<MTLDevice>>* devices = nil;
    if (@available(macOS 10.11, *)) {
      devices = MTLCopyAllDevices();
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
    MPSDevice* wrapper = new MPSDevice(dev);
    params->device->device_handle = static_cast<void*>(wrapper);
    params->device->hardware_name = dev.name.UTF8String;
    params->device->device_vendor = "Apple";
    params->device->pci_bus_id = nullptr;
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

// ---- Stream Callbacks ----
void CreateStream(const SP_Device* device, SP_Stream* stream, 
                  TF_Status* status) {
  auto* w = static_cast<MPSDevice*>(device->device_handle);
  auto* s = new MPSStreamStruct(w);
  *stream = reinterpret_cast<SP_Stream>(s);
  TfOk(status);
}

void DestroyStream(const SP_Device* /*device*/, SP_Stream stream) {
  if (!stream) return;
  auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
  delete s;
}

void CreateStreamDependency(const SP_Device* /*device*/, SP_Stream dependent,
                           SP_Stream other, TF_Status* status) {
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

void GetStreamStatus(const SP_Device* /*device*/, SP_Stream stream,
                     TF_Status* status) {
  auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
  id<MTLCommandBuffer> last = s->last_cb;
  if (!last) { 
    TfOk(status); 
    return; 
  }
  if (last.status == MTLCommandBufferStatusError) {
    TF_SetStatus(status, TF_INTERNAL, "Metal command buffer error");
  } else {
    TfOk(status);
  }
}

// ---- Event Callbacks ----
void CreateEvent(const SP_Device* /*device*/, SP_Event* event, TF_Status* status) {
  *event = reinterpret_cast<SP_Event>(new MPSEventStruct());
  TfOk(status);
}

void DestroyEvent(const SP_Device* /*device*/, SP_Event event) {
  delete reinterpret_cast<MPSEventStruct*>(event);
}

TF_Bool EventGetPollStatus(const SP_Device* /*device*/, SP_Event /*event*/) {
  return 0;  // Cannot poll semaphore without affecting count
}

void RecordEvent(const SP_Device* /*device*/, SP_Stream stream, SP_Event event,
                 TF_Status* status) {
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

void WaitForEvent(const SP_Device* /*device*/, SP_Stream /*stream*/, SP_Event event,
                  TF_Status* status) {
  auto* ev = reinterpret_cast<MPSEventStruct*>(event);
  dispatch_semaphore_wait(ev->sema, DISPATCH_TIME_FOREVER);
  TfOk(status);
}

// ---- Memory Callbacks ----
void MemAllocate(const SP_Device* device, uint64_t size, int64_t /*memory_space*/,
                 SP_DeviceMemoryBase* mem) {
  @autoreleasepool {
    auto* w = static_cast<MPSDevice*>(device->device_handle);
    if (size == 0) {
      mem->opaque = nullptr;
      mem->size = 0;
      mem->payload = 0;
      return;
    }
    id<MTLBuffer> buf = [w->device newBufferWithLength:size 
                                              options:MTLResourceStorageModePrivate];
    if (!buf) {
      mem->opaque = nullptr;
      mem->size = 0;
      mem->payload = 0;
      return;
    }
    mem->opaque = RetainToOpaque<id<MTLBuffer>>(buf);
    mem->size = size;
    mem->payload = 0;
  }
}

void MemDeallocate(const SP_Device* /*device*/, SP_DeviceMemoryBase* memory) {
  if (!memory || !memory->opaque) return;
  (void)TransferFromOpaque<id<MTLBuffer>>(memory->opaque);
  memory->opaque = nullptr;
  memory->size = 0;
  memory->payload = 0;
}

void* HostMemAllocate(const SP_Device* /*device*/, uint64_t size) {
  if (size == 0) return nullptr;
  return malloc(size);
}

void HostMemDeallocate(const SP_Device* /*device*/, void* mem) {
  free(mem);
}

// ---- Memory Copy Callbacks ----
TF_Bool MemcpyDeviceToHost(const SP_Device* /*device*/, SP_Stream stream,
                          void* host_dst, const SP_DeviceMemoryBase* device_src,
                          uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> src = BridgeNoTransfer<id<MTLBuffer>>(device_src->opaque);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBuffer> staging = [s->dev->device newBufferWithLength:size 
                                                       options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:staging destinationOffset:0 size:size];
    [blit endEncoding];
    [cb addCompletedHandler:^(__unused id<MTLCommandBuffer>) {
      memcpy(host_dst, staging.contents, (size_t)size);
    }];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
    return 1;
  }
}

TF_Bool MemcpyHostToDevice(const SP_Device* /*device*/, SP_Stream stream,
                          SP_DeviceMemoryBase* device_dst, const void* host_src,
                          uint64_t size, TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(device_dst->opaque);
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBuffer> staging = [s->dev->device newBufferWithLength:size 
                                                       options:MTLResourceStorageModeShared];
    memcpy(staging.contents, host_src, (size_t)size);
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
    return 1;
  }
}

TF_Bool MemcpyDeviceToDevice(const SP_Device* /*device*/, SP_Stream stream,
                            SP_DeviceMemoryBase* device_dst,
                            const SP_DeviceMemoryBase* device_src,
                            uint64_t size, TF_Status* status) {
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
    return 1;
  }
}

// ---- Synchronous Memory Copy ----
TF_Bool SyncMemcpyDeviceToHost(const SP_Device* /*device*/, void* host_dst,
                              const SP_DeviceMemoryBase* device_src,
                              uint64_t size, TF_Status* status) {
  @autoreleasepool {
    id<MTLBuffer> src = BridgeNoTransfer<id<MTLBuffer>>(device_src->opaque);
    id<MTLDevice> dev = src.device;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBuffer> staging = [dev newBufferWithLength:size 
                                            options:MTLResourceStorageModeShared];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src sourceOffset:0 toBuffer:staging destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(host_dst, staging.contents, (size_t)size);
    TfOk(status);
    return 1;
  }
}

TF_Bool SyncMemcpyHostToDevice(const SP_Device* /*device*/,
                              SP_DeviceMemoryBase* device_dst,
                              const void* host_src, uint64_t size,
                              TF_Status* status) {
  @autoreleasepool {
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(device_dst->opaque);
    id<MTLCommandQueue> q = [dst.device newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBuffer> staging = [dst.device newBufferWithLength:size 
                                                   options:MTLResourceStorageModeShared];
    memcpy(staging.contents, host_src, (size_t)size);
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    TfOk(status);
    return 1;
  }
}

TF_Bool SyncMemcpyDeviceToDevice(const SP_Device* /*device*/,
                                SP_DeviceMemoryBase* device_dst,
                                const SP_DeviceMemoryBase* device_src,
                                uint64_t size, TF_Status* status) {
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
    return 1;
  }
}

// ---- Host Blocking ----
void BlockHostUntilDone(const SP_Device* /*device*/, SP_Stream stream,
                        TF_Status* status) {
  auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
  id<MTLCommandBuffer> last = s->last_cb;
  if (last) {
    [last waitUntilCompleted];
  }
  TfOk(status);
}

// ---- Memory Fill Operations ----
void MemZero(const SP_Device* /*device*/, SP_Stream stream,
             SP_DeviceMemoryBase* location, uint64_t size, TF_Status* status) {
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

void Memset(const SP_Device* /*device*/, SP_Stream stream,
            SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
            TF_Status* status) {
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

void Memset32(const SP_Device* /*device*/, SP_Stream stream,
              SP_DeviceMemoryBase* location, uint32_t pattern, uint64_t size,
              TF_Status* status) {
  @autoreleasepool {
    auto* s = reinterpret_cast<MPSStreamStruct*>(stream);
    id<MTLBuffer> dst = BridgeNoTransfer<id<MTLBuffer>>(location->opaque);
    
    // Metal's fillBuffer only supports byte patterns, so we need to use compute
    // For now, use a CPU-side memset then upload
    std::vector<uint32_t> data(size / sizeof(uint32_t));
    std::fill(data.begin(), data.end(), pattern);
    
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBuffer> staging = [s->dev->device newBufferWithLength:size 
                                                       options:MTLResourceStorageModeShared];
    memcpy(staging.contents, data.data(), size);
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:dst destinationOffset:0 size:size];
    [blit endEncoding];
    [cb commit];
    s->last_cb = cb;
    TfOk(status);
  }
}

}  // namespace mps
}  // namespace tensorflow
