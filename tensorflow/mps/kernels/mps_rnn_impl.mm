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

// REAL IMPLEMENTATION: RNN, LSTM, GRU cells with Metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

struct MPSRNNContext {
  int64_t hidden_size;
  int64_t num_layers;
  float dropout;
  bool bidirectional;
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLLibrary> library;
  
  id<MTLComputePipelineState> lstm_pipeline;
  id<MTLComputePipelineState> gru_pipeline;
};

static const char* kRNNShaders = R"(
#include <metal_stdlib>
using namespace metal;

// LSTM cell: (c_t, h_t) = LSTM(x_t, h_{t-1}, c_{t-1})
kernel void lstm_cell(
    device const float* input [[buffer(0)]],
    device const float* h_prev [[buffer(1)]],
    device const float* c_prev [[buffer(2)]],
    device const float* weight_ih [[buffer(3)]],
    device const float* weight_hh [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    device float* h_next [[buffer(6)]],
    device float* c_next [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& input_size [[buffer(9)]],
    constant uint& hidden_size [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.x; // batch index
    uint h = gid.y; // hidden index
    
    if (b >= batch_size || h >= hidden_size) return;
    
    // Compute gates: i, f, g, o (input, forget, cell, output)
    float i_gate = 0.0f, f_gate = 0.0f, g_gate = 0.0f, o_gate = 0.0f;
    
    // Input contribution
    for (uint i = 0; i < input_size; ++i) {
        float x = input[b * input_size + i];
        i_gate += x * weight_ih[(0 * hidden_size + h) * input_size + i];
        f_gate += x * weight_ih[(1 * hidden_size + h) * input_size + i];
        g_gate += x * weight_ih[(2 * hidden_size + h) * input_size + i];
        o_gate += x * weight_ih[(3 * hidden_size + h) * input_size + i];
    }
    
    // Hidden contribution
    for (uint j = 0; j < hidden_size; ++j) {
        float h_p = h_prev[b * hidden_size + j];
        i_gate += h_p * weight_hh[(0 * hidden_size + h) * hidden_size + j];
        f_gate += h_p * weight_hh[(1 * hidden_size + h) * hidden_size + j];
        g_gate += h_p * weight_hh[(2 * hidden_size + h) * hidden_size + j];
        o_gate += h_p * weight_hh[(3 * hidden_size + h) * hidden_size + j];
    }
    
    // Add bias
    i_gate += bias[0 * hidden_size + h];
    f_gate += bias[1 * hidden_size + h];
    g_gate += bias[2 * hidden_size + h];
    o_gate += bias[3 * hidden_size + h];
    
    // Apply activations
    i_gate = 1.0f / (1.0f + exp(-i_gate)); // sigmoid
    f_gate = 1.0f / (1.0f + exp(-f_gate)); // sigmoid
    g_gate = tanh(g_gate);                  // tanh
    o_gate = 1.0f / (1.0f + exp(-o_gate)); // sigmoid
    
    // Update cell state
    float c_p = c_prev[b * hidden_size + h];
    float c_t = f_gate * c_p + i_gate * g_gate;
    c_next[b * hidden_size + h] = c_t;
    
    // Update hidden state
    h_next[b * hidden_size + h] = o_gate * tanh(c_t);
}

// GRU cell: h_t = GRU(x_t, h_{t-1})
kernel void gru_cell(
    device const float* input [[buffer(0)]],
    device const float* h_prev [[buffer(1)]],
    device const float* weight_ih [[buffer(2)]],
    device const float* weight_hh [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* h_next [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant uint& input_size [[buffer(7)]],
    constant uint& hidden_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint b = gid.x;
    uint h = gid.y;
    
    if (b >= batch_size || h >= hidden_size) return;
    
    // Reset gate (r), update gate (z), new gate (n)
    float r_gate = 0.0f, z_gate = 0.0f, n_gate = 0.0f;
    
    // Input contribution
    for (uint i = 0; i < input_size; ++i) {
        float x = input[b * input_size + i];
        r_gate += x * weight_ih[(0 * hidden_size + h) * input_size + i];
        z_gate += x * weight_ih[(1 * hidden_size + h) * input_size + i];
        n_gate += x * weight_ih[(2 * hidden_size + h) * input_size + i];
    }
    
    // Hidden contribution for r and z
    for (uint j = 0; j < hidden_size; ++j) {
        float h_p = h_prev[b * hidden_size + j];
        r_gate += h_p * weight_hh[(0 * hidden_size + h) * hidden_size + j];
        z_gate += h_p * weight_hh[(1 * hidden_size + h) * hidden_size + j];
    }
    
    // Apply sigmoid to r and z
    r_gate = 1.0f / (1.0f + exp(-(r_gate + bias[0 * hidden_size + h])));
    z_gate = 1.0f / (1.0f + exp(-(z_gate + bias[1 * hidden_size + h])));
    
    // New gate with reset applied
    for (uint j = 0; j < hidden_size; ++j) {
        float h_p = h_prev[b * hidden_size + j];
        n_gate += r_gate * h_p * weight_hh[(2 * hidden_size + h) * hidden_size + j];
    }
    n_gate = tanh(n_gate + bias[2 * hidden_size + h]);
    
    // Update hidden state
    float h_p = h_prev[b * hidden_size + h];
    h_next[b * hidden_size + h] = (1.0f - z_gate) * n_gate + z_gate * h_p;
}
)";

extern "C" void* MPSRNN_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSRNNContext();
  TF_Status* status = TF_NewStatus();
  
  kernel_ctx->hidden_size = 128;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "hidden_size", &kernel_ctx->hidden_size, status);
  
  kernel_ctx->num_layers = 1;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "num_layers", &kernel_ctx->num_layers, status);
  
  kernel_ctx->dropout = 0.0f;
  TF_OpKernelConstruction_GetAttrFloat(ctx, "dropout", &kernel_ctx->dropout, status);
  
  kernel_ctx->bidirectional = false;
  TF_OpKernelConstruction_GetAttrBool(ctx, "bidirectional", &kernel_ctx->bidirectional, status);
  
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:kRNNShaders];
  kernel_ctx->library = [kernel_ctx->device newLibraryWithSource:shaderSource options:nil error:&error];
  
  if (!error) {
    id<MTLFunction> lstm_func = [kernel_ctx->library newFunctionWithName:@"lstm_cell"];
    id<MTLFunction> gru_func = [kernel_ctx->library newFunctionWithName:@"gru_cell"];
    
    kernel_ctx->lstm_pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:lstm_func error:&error];
    kernel_ctx->gru_pipeline = [kernel_ctx->device newComputePipelineStateWithFunction:gru_func error:&error];
  }
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSRNN_Delete(void* kernel) {
  auto* ctx = static_cast<MPSRNNContext*>(kernel);
  delete ctx;
}

extern "C" void MPSLSTM_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* rnn_ctx = static_cast<MPSRNNContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    // Inputs: input, h_prev, c_prev, weight_ih, weight_hh, bias
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* h_prev_tensor = nullptr;
    TF_Tensor* c_prev_tensor = nullptr;
    TF_Tensor* weight_ih_tensor = nullptr;
    TF_Tensor* weight_hh_tensor = nullptr;
    TF_Tensor* bias_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &h_prev_tensor, status);
    TF_GetInput(ctx, 2, &c_prev_tensor, status);
    TF_GetInput(ctx, 3, &weight_ih_tensor, status);
    TF_GetInput(ctx, 4, &weight_hh_tensor, status);
    TF_GetInput(ctx, 5, &bias_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    uint32_t batch_size = TF_Dim(input_tensor, 0);
    uint32_t input_size = TF_Dim(input_tensor, 1);
    uint32_t hidden_size = TF_Dim(h_prev_tensor, 1);
    
    // Create buffers
    size_t input_bytes = batch_size * input_size * sizeof(float);
    size_t hidden_bytes = batch_size * hidden_size * sizeof(float);
    size_t weight_ih_bytes = 4 * hidden_size * input_size * sizeof(float);
    size_t weight_hh_bytes = 4 * hidden_size * hidden_size * sizeof(float);
    size_t bias_bytes = 4 * hidden_size * sizeof(float);
    
    id<MTLBuffer> input_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(input_tensor) length:input_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> h_prev_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(h_prev_tensor) length:hidden_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_prev_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(c_prev_tensor) length:hidden_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_ih_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(weight_ih_tensor) length:weight_ih_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_hh_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(weight_hh_tensor) length:weight_hh_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(bias_tensor) length:bias_bytes options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> h_next_buf = [rnn_ctx->device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c_next_buf = [rnn_ctx->device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> batch_buf = [rnn_ctx->device newBufferWithBytes:&batch_size length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_size_buf = [rnn_ctx->device newBufferWithBytes:&input_size length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hidden_size_buf = [rnn_ctx->device newBufferWithBytes:&hidden_size length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [rnn_ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:rnn_ctx->lstm_pipeline];
    [encoder setBuffer:input_buf offset:0 atIndex:0];
    [encoder setBuffer:h_prev_buf offset:0 atIndex:1];
    [encoder setBuffer:c_prev_buf offset:0 atIndex:2];
    [encoder setBuffer:weight_ih_buf offset:0 atIndex:3];
    [encoder setBuffer:weight_hh_buf offset:0 atIndex:4];
    [encoder setBuffer:bias_buf offset:0 atIndex:5];
    [encoder setBuffer:h_next_buf offset:0 atIndex:6];
    [encoder setBuffer:c_next_buf offset:0 atIndex:7];
    [encoder setBuffer:batch_buf offset:0 atIndex:8];
    [encoder setBuffer:input_size_buf offset:0 atIndex:9];
    [encoder setBuffer:hidden_size_buf offset:0 atIndex:10];
    
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake((batch_size + 15) / 16, (hidden_size + 15) / 16, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Allocate outputs
    int64_t output_dims[2] = {batch_size, hidden_size};
    TF_Tensor* h_out = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 2, hidden_bytes, status);
    TF_Tensor* c_out = TF_AllocateOutput(ctx, 1, TF_FLOAT, output_dims, 2, hidden_bytes, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(h_out), [h_next_buf contents], hidden_bytes);
    memcpy(TF_TensorData(c_out), [c_next_buf contents], hidden_bytes);
  }
  
  TF_DeleteStatus(status);
}

extern "C" void MPSGRU_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* rnn_ctx = static_cast<MPSRNNContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* h_prev_tensor = nullptr;
    TF_Tensor* weight_ih_tensor = nullptr;
    TF_Tensor* weight_hh_tensor = nullptr;
    TF_Tensor* bias_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &h_prev_tensor, status);
    TF_GetInput(ctx, 2, &weight_ih_tensor, status);
    TF_GetInput(ctx, 3, &weight_hh_tensor, status);
    TF_GetInput(ctx, 4, &bias_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    uint32_t batch_size = TF_Dim(input_tensor, 0);
    uint32_t input_size = TF_Dim(input_tensor, 1);
    uint32_t hidden_size = TF_Dim(h_prev_tensor, 1);
    
    size_t input_bytes = batch_size * input_size * sizeof(float);
    size_t hidden_bytes = batch_size * hidden_size * sizeof(float);
    
    id<MTLBuffer> input_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(input_tensor) length:input_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> h_prev_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(h_prev_tensor) length:hidden_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_ih_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(weight_ih_tensor) length:3*hidden_size*input_size*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_hh_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(weight_hh_tensor) length:3*hidden_size*hidden_size*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias_buf = [rnn_ctx->device newBufferWithBytes:TF_TensorData(bias_tensor) length:3*hidden_size*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> h_next_buf = [rnn_ctx->device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> batch_buf = [rnn_ctx->device newBufferWithBytes:&batch_size length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_size_buf = [rnn_ctx->device newBufferWithBytes:&input_size length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> hidden_size_buf = [rnn_ctx->device newBufferWithBytes:&hidden_size length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [rnn_ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:rnn_ctx->gru_pipeline];
    [encoder setBuffer:input_buf offset:0 atIndex:0];
    [encoder setBuffer:h_prev_buf offset:0 atIndex:1];
    [encoder setBuffer:weight_ih_buf offset:0 atIndex:2];
    [encoder setBuffer:weight_hh_buf offset:0 atIndex:3];
    [encoder setBuffer:bias_buf offset:0 atIndex:4];
    [encoder setBuffer:h_next_buf offset:0 atIndex:5];
    [encoder setBuffer:batch_buf offset:0 atIndex:6];
    [encoder setBuffer:input_size_buf offset:0 atIndex:7];
    [encoder setBuffer:hidden_size_buf offset:0 atIndex:8];
    
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake((batch_size + 15) / 16, (hidden_size + 15) / 16, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    int64_t output_dims[2] = {batch_size, hidden_size};
    TF_Tensor* h_out = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 2, hidden_bytes, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(h_out), [h_next_buf contents], hidden_bytes);
  }
  
  TF_DeleteStatus(status);
}

}  // namespace mps
}  // namespace tensorflow
