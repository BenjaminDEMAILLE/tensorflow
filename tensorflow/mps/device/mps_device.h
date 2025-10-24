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

#ifndef TENSORFLOW_MPS_DEVICE_MPS_DEVICE_H_
#define TENSORFLOW_MPS_DEVICE_MPS_DEVICE_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"

namespace tensorflow {
namespace mps {

// Platform name
constexpr char kPlatformName[] = "MPS";
constexpr char kDeviceType[] = "MPS";

// Wrapper for Metal device
struct MPSDevice {
  id<MTLDevice> device;
  explicit MPSDevice(id<MTLDevice> d) : device(d) {}
};

// Stream structure (command queue wrapper)
struct MPSStreamStruct {
  MPSDevice* dev;
  id<MTLCommandQueue> queue;
  id<MTLCommandBuffer> last_cb;
  
  explicit MPSStreamStruct(MPSDevice* d);
  ~MPSStreamStruct();
};

// Event structure (synchronization primitive)
struct MPSEventStruct {
  dispatch_semaphore_t sema;
  
  MPSEventStruct();
  ~MPSEventStruct();
};

// ---- Platform Callbacks ----
void GetDeviceCount(const SP_Platform* platform, int* device_count,
                    TF_Status* status);

void CreateDevice(const SP_Platform* platform,
                  SE_CreateDeviceParams* params, TF_Status* status);

void DestroyDevice(const SP_Platform* platform, SP_Device* device);

int32_t DevGetNumaNode(const SP_Device* device);
int64_t DevGetBandwidth(const SP_Device* device);

// ---- Stream Callbacks ----
void CreateStream(const SP_Device* device, SP_Stream* stream, 
                  TF_Status* status);

void DestroyStream(const SP_Device* device, SP_Stream stream);

void CreateStreamDependency(const SP_Device* device, SP_Stream dependent,
                           SP_Stream other, TF_Status* status);

void GetStreamStatus(const SP_Device* device, SP_Stream stream,
                     TF_Status* status);

// ---- Event Callbacks ----
void CreateEvent(const SP_Device* device, SP_Event* event, TF_Status* status);

void DestroyEvent(const SP_Device* device, SP_Event event);

TF_Bool EventGetPollStatus(const SP_Device* device, SP_Event event);

void RecordEvent(const SP_Device* device, SP_Stream stream, SP_Event event,
                 TF_Status* status);

void WaitForEvent(const SP_Device* device, SP_Stream stream, SP_Event event,
                  TF_Status* status);

// ---- Memory Callbacks ----
void MemAllocate(const SP_Device* device, uint64_t size, int64_t memory_space,
                 SP_DeviceMemoryBase* mem);

void MemDeallocate(const SP_Device* device, SP_DeviceMemoryBase* mem);

void* HostMemAllocate(const SP_Device* device, uint64_t size);

void HostMemDeallocate(const SP_Device* device, void* mem);

TF_Bool MemcpyDeviceToHost(const SP_Device* device, SP_Stream stream,
                          void* host_dst, const SP_DeviceMemoryBase* device_src,
                          uint64_t size, TF_Status* status);

TF_Bool MemcpyHostToDevice(const SP_Device* device, SP_Stream stream,
                          SP_DeviceMemoryBase* device_dst, const void* host_src,
                          uint64_t size, TF_Status* status);

TF_Bool MemcpyDeviceToDevice(const SP_Device* device, SP_Stream stream,
                            SP_DeviceMemoryBase* device_dst,
                            const SP_DeviceMemoryBase* device_src,
                            uint64_t size, TF_Status* status);

TF_Bool SyncMemcpyDeviceToHost(const SP_Device* device, void* host_dst,
                              const SP_DeviceMemoryBase* device_src,
                              uint64_t size, TF_Status* status);

TF_Bool SyncMemcpyHostToDevice(const SP_Device* device,
                              SP_DeviceMemoryBase* device_dst,
                              const void* host_src, uint64_t size,
                              TF_Status* status);

TF_Bool SyncMemcpyDeviceToDevice(const SP_Device* device,
                                SP_DeviceMemoryBase* device_dst,
                                const SP_DeviceMemoryBase* device_src,
                                uint64_t size, TF_Status* status);

void BlockHostUntilDone(const SP_Device* device, SP_Stream stream,
                        TF_Status* status);

void MemZero(const SP_Device* device, SP_Stream stream,
             SP_DeviceMemoryBase* location, uint64_t size, TF_Status* status);

void Memset(const SP_Device* device, SP_Stream stream,
            SP_DeviceMemoryBase* location, uint8_t pattern, uint64_t size,
            TF_Status* status);

void Memset32(const SP_Device* device, SP_Stream stream,
              SP_DeviceMemoryBase* location, uint32_t pattern, uint64_t size,
              TF_Status* status);

}  // namespace mps
}  // namespace tensorflow

#endif  // TENSORFLOW_MPS_DEVICE_MPS_DEVICE_H_
