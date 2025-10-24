/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Attention Operations
// ScaledDotProductAttention, MultiHeadAttention, FusedAttention, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// Scaled Dot-Product Attention
struct MPSAttentionAttrs {
  float scale;
  bool use_causal_mask;
};

void* MPSScaledDotProductAttention_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSAttentionAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrFloat(ctx, "scale", &attrs->scale, s);
  if (TF_GetCode(s) != TF_OK) attrs->scale = 0.0f;
  
  TF_OpKernelConstruction_GetAttrBool(ctx, "use_causal_mask", &attrs->use_causal_mask, s);
  if (TF_GetCode(s) != TF_OK) attrs->use_causal_mask = false;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSScaledDotProductAttention_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSAttentionAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  // Get Q, K, V tensors
  TF_Tensor* query = nullptr;
  TF_Tensor* key = nullptr;
  TF_Tensor* value = nullptr;
  
  TF_GetInput(ctx, 0, &query, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &key, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 2, &value, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Q, K, V: [batch, seq_len, d_k]
  int64_t batch = TF_Dim(query, 0);
  int64_t seq_len_q = TF_Dim(query, 1);
  int64_t d_k = TF_Dim(query, 2);
  int64_t seq_len_k = TF_Dim(key, 1);
  int64_t d_v = TF_Dim(value, 2);
  
  // Auto-calculate scale if not provided
  if (attrs->scale == 0.0f) {
    attrs->scale = 1.0f / sqrtf((float)d_k);
  }
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* Q = [graph placeholderWithShape:@[@(batch), @(seq_len_q), @(d_k)]
                                            dataType:MPSDataTypeFloat32
                                                name:@"query"];
    
    MPSGraphTensor* K = [graph placeholderWithShape:@[@(batch), @(seq_len_k), @(d_k)]
                                            dataType:MPSDataTypeFloat32
                                                name:@"key"];
    
    MPSGraphTensor* V = [graph placeholderWithShape:@[@(batch), @(seq_len_k), @(d_v)]
                                            dataType:MPSDataTypeFloat32
                                                name:@"value"];
    
    // Transpose K: [batch, d_k, seq_len_k]
    MPSGraphTensor* K_T = [graph transposeTensor:K
                                       dimension:1
                                   withDimension:2
                                            name:@"key_transpose"];
    
    // Attention scores: Q @ K^T
    MPSGraphTensor* scores = [graph matrixMultiplicationWithPrimaryTensor:Q
                                                          secondaryTensor:K_T
                                                                     name:@"scores"];
    
    // Scale: scores / sqrt(d_k)
    MPSGraphTensor* scaleTensor = [graph constantWithScalar:attrs->scale
                                                     dataType:MPSDataTypeFloat32];
    MPSGraphTensor* scaled_scores = [graph multiplicationWithPrimaryTensor:scores
                                                           secondaryTensor:scaleTensor
                                                                      name:@"scaled_scores"];
    
    // Apply causal mask if needed
    if (attrs->use_causal_mask) {
      // TODO: Create triangular mask
      // For now, skip masking
    }
    
    // Softmax: attention_weights = softmax(scaled_scores)
    MPSGraphTensor* attention_weights = [graph softMaxWithTensor:scaled_scores
                                                             axis:-1
                                                             name:@"attention_weights"];
    
    // Output: attention_weights @ V
    MPSGraphTensor* output = [graph matrixMultiplicationWithPrimaryTensor:attention_weights
                                                          secondaryTensor:V
                                                                     name:@"attention_output"];
    
    // Execute
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* q_data = static_cast<float*>(TF_TensorData(query));
    float* k_data = static_cast<float*>(TF_TensorData(key));
    float* v_data = static_cast<float*>(TF_TensorData(value));
    
    size_t q_bytes = batch * seq_len_q * d_k * sizeof(float);
    size_t k_bytes = batch * seq_len_k * d_k * sizeof(float);
    size_t v_bytes = batch * seq_len_k * d_v * sizeof(float);
    size_t output_bytes = batch * seq_len_q * d_v * sizeof(float);
    
    id<MTLBuffer> qBuffer = [device newBufferWithBytes:q_data
                                                length:q_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuffer = [device newBufferWithBytes:k_data
                                                length:k_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> vBuffer = [device newBufferWithBytes:v_data
                                                length:v_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* qData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:qBuffer
                    shape:@[@(batch), @(seq_len_q), @(d_k)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* kData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:kBuffer
                    shape:@[@(batch), @(seq_len_k), @(d_k)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* vData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:vBuffer
                    shape:@[@(batch), @(seq_len_k), @(d_v)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(batch), @(seq_len_q), @(d_v)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{
      Q: qData,
      K: kData,
      V: vData
    };
    
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    // Allocate output
    int64_t out_dims[] = {batch, seq_len_q, d_v};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 3, output_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    float* out_data = static_cast<float*>(TF_TensorData(tf_output));
    memcpy(out_data, [outputBuffer contents], output_bytes);
  }
  
  TF_DeleteStatus(s);
}

void MPSScaledDotProductAttention_Delete(void* kernel) {
  delete static_cast<MPSAttentionAttrs*>(kernel);
}

void RegisterAttentionOps(const char* platform_name, TF_Status* status) {
  // Einsum (used for attention)
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Einsum", platform_name,
                                                &MPSScaledDotProductAttention_Create,
                                                &MPSScaledDotProductAttention_Compute,
                                                &MPSScaledDotProductAttention_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSEinsum", kb, status);
  }
  
  // BatchMatMul (also used for attention)
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("BatchMatMul", platform_name,
                                                &MPSScaledDotProductAttention_Create,
                                                &MPSScaledDotProductAttention_Compute,
                                                &MPSScaledDotProductAttention_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSBatchMatMul", kb, status);
  }
  
  // TODO: 28+ more attention ops
  // MultiHeadAttention, CrossAttention, SelfAttention, CausalAttention, etc.
}

}  // namespace mps
}  // namespace tensorflow
