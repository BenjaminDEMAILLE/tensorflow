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

// REAL IMPLEMENTATION: Activation functions with Metal compute shaders

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// Activation kernel context
struct MPSActivationContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
  id<MTLLibrary> library;
  
  // Pipeline states for different activations
  id<MTLComputePipelineState> relu_pipeline;
  id<MTLComputePipelineState> relu6_pipeline;
  id<MTLComputePipelineState> elu_pipeline;
  id<MTLComputePipelineState> selu_pipeline;
  id<MTLComputePipelineState> leaky_relu_pipeline;
  id<MTLComputePipelineState> sigmoid_pipeline;
  id<MTLComputePipelineState> tanh_pipeline;
  id<MTLComputePipelineState> softplus_pipeline;
  id<MTLComputePipelineState> softsign_pipeline;
  id<MTLComputePipelineState> swish_pipeline;
  id<MTLComputePipelineState> gelu_pipeline;
  
  float alpha; // for LeakyReLU, ELU
};

// Metal shaders for activation functions
static const char* kActivationShaders = R"(
#include <metal_stdlib>
using namespace metal;

// ReLU: max(0, x)
kernel void relu(device const float* input [[buffer(0)]],
                 device float* output [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    output[gid] = max(0.0f, input[gid]);
}

// ReLU6: min(max(0, x), 6)
kernel void relu6(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  uint gid [[thread_position_in_grid]])
{
    output[gid] = clamp(input[gid], 0.0f, 6.0f);
}

// ELU: x if x > 0, alpha * (exp(x) - 1) otherwise
kernel void elu(device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant float& alpha [[buffer(2)]],
                uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    output[gid] = x > 0.0f ? x : alpha * (exp(x) - 1.0f);
}

// SELU: scale * (x if x > 0, alpha * (exp(x) - 1) otherwise)
kernel void selu(device const float* input [[buffer(0)]],
                 device float* output [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    constant float scale = 1.0507009873554804934193349852946f;
    constant float alpha = 1.6732632423543772848170429916717f;
    float x = input[gid];
    output[gid] = scale * (x > 0.0f ? x : alpha * (exp(x) - 1.0f));
}

// Leaky ReLU: max(alpha * x, x)
kernel void leaky_relu(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant float& alpha [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    output[gid] = max(alpha * x, x);
}

// Sigmoid: 1 / (1 + exp(-x))
kernel void sigmoid(device const float* input [[buffer(0)]],
                    device float* output [[buffer(1)]],
                    uint gid [[thread_position_in_grid]])
{
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

// Tanh: (exp(2x) - 1) / (exp(2x) + 1)
kernel void tanh_activation(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            uint gid [[thread_position_in_grid]])
{
    output[gid] = tanh(input[gid]);
}

// Softplus: log(1 + exp(x))
kernel void softplus(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    // Numerically stable version
    output[gid] = x > 20.0f ? x : log(1.0f + exp(x));
}

// Softsign: x / (1 + |x|)
kernel void softsign(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    output[gid] = x / (1.0f + abs(x));
}

// Swish (SiLU): x * sigmoid(x)
kernel void swish(device const float* input [[buffer(0)]],
                  device float* output [[buffer(1)]],
                  uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}

// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
kernel void gelu(device const float* input [[buffer(0)]],
                 device float* output [[buffer(1)]],
                 uint gid [[thread_position_in_grid]])
{
    float x = input[gid];
    constant float sqrt_2_over_pi = 0.7978845608028654f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}
)";

static id<MTLComputePipelineState> CreatePipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  NSError* error = nil;
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  return pipeline;
}

extern "C" void* MPSActivation_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSActivationContext();
  TF_Status* status = TF_NewStatus();
  
  kernel_ctx->alpha = 0.2f; // Default for LeakyReLU
  TF_OpKernelConstruction_GetAttrFloat(ctx, "alpha", &kernel_ctx->alpha, status);
  
  // Initialize Metal
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  // Compile shaders
  NSError* error = nil;
  NSString* shaderSource = [NSString stringWithUTF8String:kActivationShaders];
  kernel_ctx->library = [kernel_ctx->device newLibraryWithSource:shaderSource options:nil error:&error];
  
  if (!error) {
    kernel_ctx->relu_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"relu");
    kernel_ctx->relu6_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"relu6");
    kernel_ctx->elu_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"elu");
    kernel_ctx->selu_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"selu");
    kernel_ctx->leaky_relu_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"leaky_relu");
    kernel_ctx->sigmoid_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"sigmoid");
    kernel_ctx->tanh_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"tanh_activation");
    kernel_ctx->softplus_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"softplus");
    kernel_ctx->softsign_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"softsign");
    kernel_ctx->swish_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"swish");
    kernel_ctx->gelu_pipeline = CreatePipeline(kernel_ctx->device, kernel_ctx->library, @"gelu");
  }
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSActivation_Delete(void* kernel) {
  auto* ctx = static_cast<MPSActivationContext*>(kernel);
  delete ctx;
}

// Generic compute function for activations
static void ComputeActivation(MPSActivationContext* ctx, TF_OpKernelContext* tf_ctx, 
                               id<MTLComputePipelineState> pipeline, bool needs_alpha = false) {
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    // Get input tensor
    TF_Tensor* input_tensor = nullptr;
    TF_GetInput(tf_ctx, 0, &input_tensor, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get number of elements
    int num_dims = TF_NumDims(input_tensor);
    size_t num_elements = 1;
    std::vector<int64_t> dims(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      dims[i] = TF_Dim(input_tensor, i);
      num_elements *= dims[i];
    }
    
    size_t buffer_size = num_elements * sizeof(float);
    
    // Create Metal buffers
    id<MTLBuffer> input_buffer = [ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                           length:buffer_size
                                                          options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> output_buffer = [ctx->device newBufferWithLength:buffer_size
                                                            options:MTLResourceStorageModeShared];
    
    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [ctx->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    
    if (needs_alpha) {
      id<MTLBuffer> alpha_buffer = [ctx->device newBufferWithBytes:&ctx->alpha 
                                                            length:sizeof(float) 
                                                           options:MTLResourceStorageModeShared];
      [encoder setBuffer:alpha_buffer offset:0 atIndex:2];
    }
    
    // Dispatch threads
    MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake((num_elements + 255) / 256, 1, 1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Allocate output tensor
    TF_Tensor* output_tf = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims.data(), num_dims, buffer_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Copy result back
    memcpy(TF_TensorData(output_tf), [output_buffer contents], buffer_size);
  }
  
  TF_DeleteStatus(status);
}

// Individual compute functions
extern "C" void MPSRelu_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->relu_pipeline);
}

extern "C" void MPSRelu6_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->relu6_pipeline);
}

extern "C" void MPSElu_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->elu_pipeline, true);
}

extern "C" void MPSSelu_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->selu_pipeline);
}

extern "C" void MPSLeakyRelu_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->leaky_relu_pipeline, true);
}

extern "C" void MPSSigmoid_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->sigmoid_pipeline);
}

extern "C" void MPSTanh_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->tanh_pipeline);
}

extern "C" void MPSSoftplus_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->softplus_pipeline);
}

extern "C" void MPSSoftsign_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->softsign_pipeline);
}

extern "C" void MPSSwish_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->swish_pipeline);
}

extern "C" void MPSGelu_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* activation_ctx = static_cast<MPSActivationContext*>(kernel);
  ComputeActivation(activation_ctx, ctx, activation_ctx->gelu_pipeline);
}

}  // namespace mps
}  // namespace tensorflow
