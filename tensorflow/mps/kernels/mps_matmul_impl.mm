/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// REAL IMPLEMENTATION: MatMul with Metal Performance Shaders

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <vector>

namespace tensorflow {
namespace mps {

// MatMul kernel context
struct MPSMatMulContext {
  bool transpose_a;
  bool transpose_b;
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLLibrary> library;
  id<MTLComputePipelineState> pipeline;
};

// Metal shader for optimized MatMul
static const char* kMatMulShader = R"(
#include <metal_stdlib>
using namespace metal;

// Tiled MatMul kernel - optimized for M1/M2/M3 chips
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant bool& transpose_a [[buffer(6)]],
    constant bool& transpose_b [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Tile size optimized for Apple Silicon
    constexpr uint TILE_SIZE = 32;
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Tiled multiplication
    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        threadgroup float As[TILE_SIZE][TILE_SIZE];
        threadgroup float Bs[TILE_SIZE][TILE_SIZE];
        
        // Load tiles into threadgroup memory
        uint a_idx = transpose_a ? (tile * TILE_SIZE + tid.x) * M + row : row * K + tile * TILE_SIZE + tid.x;
        uint b_idx = transpose_b ? col * K + tile * TILE_SIZE + tid.y : (tile * TILE_SIZE + tid.y) * N + col;
        
        if (tile * TILE_SIZE + tid.x < K && row < M)
            As[tid.y][tid.x] = A[a_idx];
        else
            As[tid.y][tid.x] = 0.0f;
            
        if (tile * TILE_SIZE + tid.y < K && col < N)
            Bs[tid.y][tid.x] = B[b_idx];
        else
            Bs[tid.y][tid.x] = 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    C[row * N + col] = sum;
}

// Fallback simple kernel
kernel void matmul_simple(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}
)";

extern "C" void* MPSMatMul_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSMatMulContext();
  TF_Status* status = TF_NewStatus();
  
  // Get transpose attributes
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_a", &kernel_ctx->transpose_a, status);
  TF_OpKernelConstruction_GetAttrBool(ctx, "transpose_b", &kernel_ctx->transpose_b, status);
  
  // Initialize Metal
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  // Compile shader
  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:kMatMulShader];
  kernel_ctx->library = [kernel_ctx->device newLibraryWithSource:shaderSource options:nil error:&error];
  
  if (error) {
    TF_SetStatus(status, TF_INTERNAL, [[error localizedDescription] UTF8String]);
    TF_DeleteStatus(status);
    return kernel_ctx;
  }
  
  id<MTLFunction> function = [kernel_ctx->library newFunctionWithName:@"matmul_tiled"];
  kernel_ctx->pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:function error:&error];
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSMatMul_Delete(void* kernel) {
  auto* ctx = static_cast<MPSMatMulContext*>(kernel);
  delete ctx;
}

extern "C" void MPSMatMul_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* matmul_ctx = static_cast<MPSMatMulContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    // Get input tensors
    TF_Tensor* a_tensor = nullptr;
    TF_Tensor* b_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &a_tensor, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    TF_GetInput(ctx, 1, &b_tensor, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get dimensions
    int64_t a_rows = TF_Dim(a_tensor, 0);
    int64_t a_cols = TF_Dim(a_tensor, 1);
    int64_t b_rows = TF_Dim(b_tensor, 0);
    int64_t b_cols = TF_Dim(b_tensor, 1);
    
    // Apply transposes
    uint32_t M = matmul_ctx->transpose_a ? a_cols : a_rows;
    uint32_t K = matmul_ctx->transpose_a ? a_rows : a_cols;
    uint32_t N = matmul_ctx->transpose_b ? b_rows : b_cols;
    
    // Validate dimensions
    uint32_t K_b = matmul_ctx->transpose_b ? b_cols : b_rows;
    if (K != K_b) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "Matrix dimensions incompatible for multiplication");
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Create Metal buffers
    size_t a_size = a_rows * a_cols * sizeof(float);
    size_t b_size = b_rows * b_cols * sizeof(float);
    size_t c_size = M * N * sizeof(float);
    
    id<MTLBuffer> a_buffer = [matmul_ctx->device newBufferWithBytes:TF_TensorData(a_tensor)
                                                              length:a_size
                                                             options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> b_buffer = [matmul_ctx->device newBufferWithBytes:TF_TensorData(b_tensor)
                                                              length:b_size
                                                             options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> c_buffer = [matmul_ctx->device newBufferWithLength:c_size
                                                              options:MTLResourceStorageModeShared];
    
    // Parameter buffers
    id<MTLBuffer> m_buffer = [matmul_ctx->device newBufferWithBytes:&M length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> n_buffer = [matmul_ctx->device newBufferWithBytes:&N length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> k_buffer = [matmul_ctx->device newBufferWithBytes:&K length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> ta_buffer = [matmul_ctx->device newBufferWithBytes:&matmul_ctx->transpose_a length:sizeof(bool) options:MTLResourceStorageModeShared];
    id<MTLBuffer> tb_buffer = [matmul_ctx->device newBufferWithBytes:&matmul_ctx->transpose_b length:sizeof(bool) options:MTLResourceStorageModeShared];
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [matmul_ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:matmul_ctx->pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:b_buffer offset:0 atIndex:1];
    [encoder setBuffer:c_buffer offset:0 atIndex:2];
    [encoder setBuffer:m_buffer offset:0 atIndex:3];
    [encoder setBuffer:n_buffer offset:0 atIndex:4];
    [encoder setBuffer:k_buffer offset:0 atIndex:5];
    [encoder setBuffer:ta_buffer offset:0 atIndex:6];
    [encoder setBuffer:tb_buffer offset:0 atIndex:7];
    
    // Dispatch threads
    MTLSize threadgroupSize = MTLSizeMake(32, 32, 1);
    MTLSize gridSize = MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Allocate output tensor
    int64_t output_dims[2] = {M, N};
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 2, c_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Copy result back
    memcpy(TF_TensorData(output_tf), [c_buffer contents], c_size);
  }
  
  TF_DeleteStatus(status);
}

}  // namespace mps
}  // namespace tensorflow
