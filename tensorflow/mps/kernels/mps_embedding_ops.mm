/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Embedding Operations
// EmbeddingLookup, EmbeddingLookupSparse, SafeEmbeddingLookupSparse, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// EmbeddingLookup (GatherV2) - params[ids]
void* MPSEmbeddingLookup_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSEmbeddingLookup_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  // Get inputs
  TF_Tensor* params = nullptr;
  TF_Tensor* indices = nullptr;
  
  TF_GetInput(ctx, 0, &params, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &indices, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // params: [vocab_size, embedding_dim]
  // indices: [batch_size] or [batch_size, seq_len]
  int64_t vocab_size = TF_Dim(params, 0);
  int64_t embedding_dim = TF_Dim(params, 1);
  
  int num_index_dims = TF_NumDims(indices);
  int64_t num_indices = 1;
  for (int i = 0; i < num_index_dims; i++) {
    num_indices *= TF_Dim(indices, i);
  }
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* paramsTensor = [graph placeholderWithShape:@[@(vocab_size), @(embedding_dim)]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"params"];
    
    MPSGraphTensor* indicesTensor = [graph placeholderWithShape:@[@(num_indices)]
                                                        dataType:MPSDataTypeInt32
                                                            name:@"indices"];
    
    // Gather operation: output[i] = params[indices[i]]
    MPSGraphTensor* output = [graph gatherWithUpdatesTensor:paramsTensor
                                              indicesTensor:indicesTensor
                                                       axis:0
                                            batchDimensions:0
                                                       name:@"gather"];
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* params_data = static_cast<float*>(TF_TensorData(params));
    int32_t* indices_data = static_cast<int32_t*>(TF_TensorData(indices));
    
    size_t params_bytes = vocab_size * embedding_dim * sizeof(float);
    size_t indices_bytes = num_indices * sizeof(int32_t);
    size_t output_bytes = num_indices * embedding_dim * sizeof(float);
    
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:params_data
                                                     length:params_bytes
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> indicesBuffer = [device newBufferWithBytes:indices_data
                                                      length:indices_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* paramsData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:paramsBuffer
                    shape:@[@(vocab_size), @(embedding_dim)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* indicesData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:indicesBuffer
                    shape:@[@(num_indices)]
                 dataType:MPSDataTypeInt32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(num_indices), @(embedding_dim)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{
      paramsTensor: paramsData,
      indicesTensor: indicesData
    };
    
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    // Allocate output with original indices shape + embedding_dim
    int64_t* out_dims = new int64_t[num_index_dims + 1];
    for (int i = 0; i < num_index_dims; i++) {
      out_dims[i] = TF_Dim(indices, i);
    }
    out_dims[num_index_dims] = embedding_dim;
    
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, num_index_dims + 1, output_bytes, s);
    delete[] out_dims;
    
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
void MPSEmbeddingLookup_Delete(void* kernel) {}

void RegisterEmbeddingOps(const char* platform_name, TF_Status* status) {
  // GatherV2 (EmbeddingLookup)
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("GatherV2", platform_name,
                                                &MPSEmbeddingLookup_Create,
                                                &MPSEmbeddingLookup_Compute,
                                                &MPSEmbeddingLookup_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "Tparams", TF_FLOAT, status);
    TF_KernelBuilder_TypeConstraint(kb, "Tindices", TF_INT32, status);
    TF_RegisterKernelBuilder("MPSGatherV2", kb, status);
  }
  
  // Gather (legacy)
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Gather", platform_name,
                                                &MPSEmbeddingLookup_Create,
                                                &MPSEmbeddingLookup_Compute,
                                                &MPSEmbeddingLookup_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "Tparams", TF_FLOAT, status);
    TF_KernelBuilder_TypeConstraint(kb, "Tindices", TF_INT32, status);
    TF_RegisterKernelBuilder("MPSGather", kb, status);
  }
  
  // TODO: 13+ more embedding ops
  // EmbeddingLookupSparse, SafeEmbeddingLookupSparse, GatherNd, ScatterNd, etc.
}

}  // namespace mps
}  // namespace tensorflow
