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

// MPS RNN/LSTM Operations Implementation
// Implements LSTM, GRU, and RNN operations using Metal Performance Shaders Graph

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/mps/utils/mps_common.h"

namespace tensorflow {
namespace mps {

// ============================================================================
// LSTM Cell Implementation
// ============================================================================

struct MPSLSTMCellAttrs {
  int hidden_size;
  bool use_peephole;
  float forget_bias;
};

void* MPSLSTMCell_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSLSTMCellAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_units", &attrs->hidden_size, s);
  if (TF_GetCode(s) != TF_OK) attrs->hidden_size = 128;
  
  TF_OpKernelConstruction_GetAttrBool(ctx, "use_peephole", &attrs->use_peephole, s);
  if (TF_GetCode(s) != TF_OK) attrs->use_peephole = false;
  
  TF_OpKernelConstruction_GetAttrFloat(ctx, "forget_bias", &attrs->forget_bias, s);
  if (TF_GetCode(s) != TF_OK) attrs->forget_bias = 1.0f;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSLSTMCell_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSLSTMCellAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  // Get input tensors
  TF_Tensor* input = nullptr;
  TF_Tensor* h_prev = nullptr;
  TF_Tensor* c_prev = nullptr;
  TF_Tensor* weights = nullptr;
  TF_Tensor* bias = nullptr;
  
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &h_prev, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 2, &c_prev, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 3, &weights, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 4, &bias, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Get dimensions
  int64_t batch_size = TF_Dim(input, 0);
  int64_t input_size = TF_Dim(input, 1);
  int64_t hidden_size = attrs->hidden_size;
  
  // Create MPSGraph for LSTM computation
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Create input placeholders
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch_size), @(input_size)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    MPSGraphTensor* hPrevTensor = [graph placeholderWithShape:@[@(batch_size), @(hidden_size)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"h_prev"];
    
    MPSGraphTensor* cPrevTensor = [graph placeholderWithShape:@[@(batch_size), @(hidden_size)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"c_prev"];
    
    MPSGraphTensor* weightsTensor = [graph placeholderWithShape:@[@(input_size + hidden_size), @(4 * hidden_size)]
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"weights"];
    
    MPSGraphTensor* biasTensor = [graph placeholderWithShape:@[@(4 * hidden_size)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:@"bias"];
    
    // LSTM forward pass: [input, h_prev] @ weights + bias
    MPSGraphTensor* concat = [graph concatTensor:inputTensor
                                      withTensor:hPrevTensor
                                       dimension:1
                                            name:@"concat"];
    
    MPSGraphTensor* matmul = [graph matrixMultiplicationWithPrimaryTensor:concat
                                                          secondaryTensor:weightsTensor
                                                                     name:@"matmul"];
    
    MPSGraphTensor* preact = [graph additionWithPrimaryTensor:matmul
                                              secondaryTensor:biasTensor
                                                         name:@"preact"];
    
    // Split into i, f, g, o gates
    NSArray<MPSGraphTensor*>* gates = [graph splitTensor:preact
                                              numSplits:4
                                                   axis:1
                                                   name:@"gates"];
    
    MPSGraphTensor* i_gate = gates[0];  // Input gate
    MPSGraphTensor* f_gate = gates[1];  // Forget gate
    MPSGraphTensor* g_gate = gates[2];  // Cell gate
    MPSGraphTensor* o_gate = gates[3];  // Output gate
    
    // Apply forget bias
    if (attrs->forget_bias != 0.0f) {
      MPSGraphTensor* forgetBias = [graph constantWithScalar:attrs->forget_bias
                                                       shape:@[@(batch_size), @(hidden_size)]
                                                    dataType:MPSDataTypeFloat32];
      f_gate = [graph additionWithPrimaryTensor:f_gate
                                secondaryTensor:forgetBias
                                           name:@"f_gate_bias"];
    }
    
    // Apply activations
    i_gate = [graph sigmoidWithTensor:i_gate name:@"i_sigmoid"];
    f_gate = [graph sigmoidWithTensor:f_gate name:@"f_sigmoid"];
    g_gate = [graph tanhWithTensor:g_gate name:@"g_tanh"];
    o_gate = [graph sigmoidWithTensor:o_gate name:@"o_sigmoid"];
    
    // Compute new cell state: c_new = f * c_prev + i * g
    MPSGraphTensor* f_times_c = [graph multiplicationWithPrimaryTensor:f_gate
                                                       secondaryTensor:cPrevTensor
                                                                  name:@"f_times_c"];
    MPSGraphTensor* i_times_g = [graph multiplicationWithPrimaryTensor:i_gate
                                                       secondaryTensor:g_gate
                                                                  name:@"i_times_g"];
    MPSGraphTensor* c_new = [graph additionWithPrimaryTensor:f_times_c
                                             secondaryTensor:i_times_g
                                                        name:@"c_new"];
    
    // Compute new hidden state: h_new = o * tanh(c_new)
    MPSGraphTensor* c_new_tanh = [graph tanhWithTensor:c_new name:@"c_new_tanh"];
    MPSGraphTensor* h_new = [graph multiplicationWithPrimaryTensor:o_gate
                                                   secondaryTensor:c_new_tanh
                                                              name:@"h_new"];
    
    // Get Metal device and command queue
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Create Metal buffers for input data
    float* input_data = static_cast<float*>(TF_TensorData(input));
    float* h_prev_data = static_cast<float*>(TF_TensorData(h_prev));
    float* c_prev_data = static_cast<float*>(TF_TensorData(c_prev));
    float* weights_data = static_cast<float*>(TF_TensorData(weights));
    float* bias_data = static_cast<float*>(TF_TensorData(bias));
    
    size_t input_bytes = batch_size * input_size * sizeof(float);
    size_t hidden_bytes = batch_size * hidden_size * sizeof(float);
    size_t weights_bytes = (input_size + hidden_size) * 4 * hidden_size * sizeof(float);
    size_t bias_bytes = 4 * hidden_size * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> hPrevBuffer = [device newBufferWithBytes:h_prev_data
                                                    length:hidden_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> cPrevBuffer = [device newBufferWithBytes:c_prev_data
                                                    length:hidden_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> weightsBuffer = [device newBufferWithBytes:weights_data
                                                      length:weights_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> biasBuffer = [device newBufferWithBytes:bias_data
                                                   length:bias_bytes
                                                  options:MTLResourceStorageModeShared];
    
    // Create output buffers
    id<MTLBuffer> hNewBuffer = [device newBufferWithLength:hidden_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> cNewBuffer = [device newBufferWithLength:hidden_bytes
                                                   options:MTLResourceStorageModeShared];
    
    // Create MPSGraphTensorData for execution
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch_size), @(input_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* hPrevData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:hPrevBuffer
                    shape:@[@(batch_size), @(hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* cPrevData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:cPrevBuffer
                    shape:@[@(batch_size), @(hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* weightsData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:weightsBuffer
                    shape:@[@(input_size + hidden_size), @(4 * hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* biasData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:biasBuffer
                    shape:@[@(4 * hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* hNewData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:hNewBuffer
                    shape:@[@(batch_size), @(hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* cNewData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:cNewBuffer
                    shape:@[@(batch_size), @(hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    // Execute graph
    NSDictionary* feeds = @{
      inputTensor: inputData,
      hPrevTensor: hPrevData,
      cPrevTensor: cPrevData,
      weightsTensor: weightsData,
      biasTensor: biasData
    };
    
    NSDictionary* targetTensors = @{
      h_new: hNewData,
      c_new: cNewData
    };
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    // Allocate output tensors
    int64_t h_dims[] = {batch_size, hidden_size};
    int64_t c_dims[] = {batch_size, hidden_size};
    
    TF_Tensor* h_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, h_dims, 2, hidden_bytes, s);
    TF_Tensor* c_output = TF_AllocateOutput(ctx, 1, TF_FLOAT, c_dims, 2, hidden_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    // Copy results
    float* h_out_data = static_cast<float*>(TF_TensorData(h_output));
    float* c_out_data = static_cast<float*>(TF_TensorData(c_output));
    
    memcpy(h_out_data, [hNewBuffer contents], hidden_bytes);
    memcpy(c_out_data, [cNewBuffer contents], hidden_bytes);
  }
  
  TF_DeleteStatus(s);
}

void MPSLSTMCell_Delete(void* kernel) {
  delete static_cast<MPSLSTMCellAttrs*>(kernel);
}

// ============================================================================
// GRU Cell Implementation
// ============================================================================

struct MPSGRUCellAttrs {
  int hidden_size;
};

void* MPSGRUCell_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSGRUCellAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_units", &attrs->hidden_size, s);
  if (TF_GetCode(s) != TF_OK) attrs->hidden_size = 128;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSGRUCell_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSGRUCellAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  // Get input tensors
  TF_Tensor* input = nullptr;
  TF_Tensor* h_prev = nullptr;
  TF_Tensor* weights = nullptr;
  TF_Tensor* bias = nullptr;
  
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &h_prev, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 2, &weights, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 3, &bias, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Get dimensions
  int64_t batch_size = TF_Dim(input, 0);
  int64_t input_size = TF_Dim(input, 1);
  int64_t hidden_size = attrs->hidden_size;
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Create input placeholders
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch_size), @(input_size)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    MPSGraphTensor* hPrevTensor = [graph placeholderWithShape:@[@(batch_size), @(hidden_size)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"h_prev"];
    
    MPSGraphTensor* weightsTensor = [graph placeholderWithShape:@[@(input_size + hidden_size), @(3 * hidden_size)]
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"weights"];
    
    MPSGraphTensor* biasTensor = [graph placeholderWithShape:@[@(3 * hidden_size)]
                                                    dataType:MPSDataTypeFloat32
                                                        name:@"bias"];
    
    // GRU forward pass: [input, h_prev] @ weights + bias
    MPSGraphTensor* concat = [graph concatTensor:inputTensor
                                      withTensor:hPrevTensor
                                       dimension:1
                                            name:@"concat"];
    
    MPSGraphTensor* matmul = [graph matrixMultiplicationWithPrimaryTensor:concat
                                                          secondaryTensor:weightsTensor
                                                                     name:@"matmul"];
    
    MPSGraphTensor* preact = [graph additionWithPrimaryTensor:matmul
                                              secondaryTensor:biasTensor
                                                         name:@"preact"];
    
    // Split into r, z, h gates
    NSArray<MPSGraphTensor*>* gates = [graph splitTensor:preact
                                              numSplits:3
                                                   axis:1
                                                   name:@"gates"];
    
    MPSGraphTensor* r_gate = gates[0];  // Reset gate
    MPSGraphTensor* z_gate = gates[1];  // Update gate
    MPSGraphTensor* h_gate = gates[2];  // Candidate gate
    
    // Apply activations
    r_gate = [graph sigmoidWithTensor:r_gate name:@"r_sigmoid"];
    z_gate = [graph sigmoidWithTensor:z_gate name:@"z_sigmoid"];
    h_gate = [graph tanhWithTensor:h_gate name:@"h_tanh"];
    
    // Compute new hidden state: h_new = (1 - z) * h_prev + z * h_candidate
    MPSGraphTensor* one = [graph constantWithScalar:1.0f
                                             shape:@[@(batch_size), @(hidden_size)]
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one_minus_z = [graph subtractionWithPrimaryTensor:one
                                                      secondaryTensor:z_gate
                                                                 name:@"one_minus_z"];
    
    MPSGraphTensor* term1 = [graph multiplicationWithPrimaryTensor:one_minus_z
                                                   secondaryTensor:hPrevTensor
                                                              name:@"term1"];
    MPSGraphTensor* term2 = [graph multiplicationWithPrimaryTensor:z_gate
                                                   secondaryTensor:h_gate
                                                              name:@"term2"];
    MPSGraphTensor* h_new = [graph additionWithPrimaryTensor:term1
                                             secondaryTensor:term2
                                                        name:@"h_new"];
    
    // Get Metal device and command queue
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Create Metal buffers
    float* input_data = static_cast<float*>(TF_TensorData(input));
    float* h_prev_data = static_cast<float*>(TF_TensorData(h_prev));
    float* weights_data = static_cast<float*>(TF_TensorData(weights));
    float* bias_data = static_cast<float*>(TF_TensorData(bias));
    
    size_t input_bytes = batch_size * input_size * sizeof(float);
    size_t hidden_bytes = batch_size * hidden_size * sizeof(float);
    size_t weights_bytes = (input_size + hidden_size) * 3 * hidden_size * sizeof(float);
    size_t bias_bytes = 3 * hidden_size * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> hPrevBuffer = [device newBufferWithBytes:h_prev_data
                                                    length:hidden_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> weightsBuffer = [device newBufferWithBytes:weights_data
                                                      length:weights_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> biasBuffer = [device newBufferWithBytes:bias_data
                                                   length:bias_bytes
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> hNewBuffer = [device newBufferWithLength:hidden_bytes
                                                   options:MTLResourceStorageModeShared];
    
    // Create MPSGraphTensorData
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch_size), @(input_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* hPrevData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:hPrevBuffer
                    shape:@[@(batch_size), @(hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* weightsData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:weightsBuffer
                    shape:@[@(input_size + hidden_size), @(3 * hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* biasData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:biasBuffer
                    shape:@[@(3 * hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* hNewData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:hNewBuffer
                    shape:@[@(batch_size), @(hidden_size)]
                 dataType:MPSDataTypeFloat32];
    
    // Execute graph
    NSDictionary* feeds = @{
      inputTensor: inputData,
      hPrevTensor: hPrevData,
      weightsTensor: weightsData,
      biasTensor: biasData
    };
    
    NSDictionary* targetTensors = @{h_new: hNewData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    // Allocate output tensor
    int64_t h_dims[] = {batch_size, hidden_size};
    TF_Tensor* h_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, h_dims, 2, hidden_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    // Copy result
    float* h_out_data = static_cast<float*>(TF_TensorData(h_output));
    memcpy(h_out_data, [hNewBuffer contents], hidden_bytes);
  }
  
  TF_DeleteStatus(s);
}

void MPSGRUCell_Delete(void* kernel) {
  delete static_cast<MPSGRUCellAttrs*>(kernel);
}

// ============================================================================
// Registration
// ============================================================================

void RegisterRNNOps(const char* platform_name, TF_Status* status) {
  // LSTM
  TF_KernelBuilder* lstm_kb = TF_NewKernelBuilder("LSTMCell", platform_name,
                                                   &MPSLSTMCell_Create,
                                                   &MPSLSTMCell_Compute,
                                                   &MPSLSTMCell_Delete);
  TF_KernelBuilder_TypeConstraint(lstm_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSLSTMCellFloat", lstm_kb, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  // GRU
  TF_KernelBuilder* gru_kb = TF_NewKernelBuilder("GRUCell", platform_name,
                                                  &MPSGRUCell_Create,
                                                  &MPSGRUCell_Compute,
                                                  &MPSGRUCell_Delete);
  TF_KernelBuilder_TypeConstraint(gru_kb, "T", TF_FLOAT, status);
  TF_RegisterKernelBuilder("MPSGRUCellFloat", gru_kb, status);
  if (TF_GetCode(status) != TF_OK) return;
  
  // TODO: Ajouter LSTMBlockCell, GRUBlockCell, CudnnLSTM, CudnnGRU, etc.
}

}  // namespace mps
}  // namespace tensorflow
