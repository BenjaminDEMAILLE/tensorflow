/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Sparse Tensor Operations
// SparseToDense, SparseSoftmax, SparseMatMul, SparseTensorDenseMatMul, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// SparseToDense
void* MPSSparseToDense_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSSparseToDense_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  // Get inputs: indices, output_shape, values, default_value
  TF_Tensor* indices = nullptr;
  TF_Tensor* output_shape = nullptr;
  TF_Tensor* values = nullptr;
  TF_Tensor* default_value = nullptr;
  
  TF_GetInput(ctx, 0, &indices, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &output_shape, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 2, &values, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 3, &default_value, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t num_indices = TF_Dim(indices, 0);
  int64_t ndims = TF_Dim(indices, 1);
  
  int64_t* shape_data = static_cast<int64_t*>(TF_TensorData(output_shape));
  int32_t* indices_data = static_cast<int32_t*>(TF_TensorData(indices));
  float* values_data = static_cast<float*>(TF_TensorData(values));
  float default_val = *static_cast<float*>(TF_TensorData(default_value));
  
  // Calculate total size
  int64_t total_size = 1;
  for (int64_t i = 0; i < ndims; i++) {
    total_size *= shape_data[i];
  }
  
  // Allocate output
  size_t output_bytes = total_size * sizeof(float);
  TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, shape_data, ndims, output_bytes, s);
  
  if (TF_GetCode(s) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  // Fill with default value
  float* out_data = static_cast<float*>(TF_TensorData(tf_output));
  for (int64_t i = 0; i < total_size; i++) {
    out_data[i] = default_val;
  }
  
  // Fill sparse values
  // Simplified: assumes 2D indices for now
  if (ndims == 2) {
    int64_t cols = shape_data[1];
    for (int64_t i = 0; i < num_indices; i++) {
      int32_t row = indices_data[i * 2];
      int32_t col = indices_data[i * 2 + 1];
      out_data[row * cols + col] = values_data[i];
    }
  }
  
  TF_DeleteStatus(s);
}
void MPSSparseToDense_Delete(void* kernel) {}

// SparseMatMul
void* MPSSparseMatMul_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSSparseMatMul_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* a = nullptr;
  TF_Tensor* b = nullptr;
  
  TF_GetInput(ctx, 0, &a, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &b, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t m = TF_Dim(a, 0);
  int64_t k = TF_Dim(a, 1);
  int64_t n = TF_Dim(b, 1);
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* aTensor = [graph placeholderWithShape:@[@(m), @(k)]
                                                  dataType:MPSDataTypeFloat32
                                                      name:@"a"];
    
    MPSGraphTensor* bTensor = [graph placeholderWithShape:@[@(k), @(n)]
                                                  dataType:MPSDataTypeFloat32
                                                      name:@"b"];
    
    // Matrix multiplication
    MPSGraphTensor* output = [graph matrixMultiplicationWithPrimaryTensor:aTensor
                                                          secondaryTensor:bTensor
                                                                     name:@"sparse_matmul"];
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* a_data = static_cast<float*>(TF_TensorData(a));
    float* b_data = static_cast<float*>(TF_TensorData(b));
    
    size_t a_bytes = m * k * sizeof(float);
    size_t b_bytes = k * n * sizeof(float);
    size_t output_bytes = m * n * sizeof(float);
    
    id<MTLBuffer> aBuffer = [device newBufferWithBytes:a_data
                                                length:a_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuffer = [device newBufferWithBytes:b_data
                                                length:b_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* aData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:aBuffer
                    shape:@[@(m), @(k)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* bData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bBuffer
                    shape:@[@(k), @(n)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(m), @(n)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{
      aTensor: aData,
      bTensor: bData
    };
    
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    int64_t out_dims[] = {m, n};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 2, output_bytes, s);
    
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
void MPSSparseMatMul_Delete(void* kernel) {}

void RegisterSparseOps(const char* platform_name, TF_Status* status) {
  // SparseToDense
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("SparseToDense", platform_name,
                                                &MPSSparseToDense_Create,
                                                &MPSSparseToDense_Compute,
                                                &MPSSparseToDense_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSparseToDense", kb, status);
  }
  
  // SparseMatMul
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("SparseMatMul", platform_name,
                                                &MPSSparseMatMul_Create,
                                                &MPSSparseMatMul_Compute,
                                                &MPSSparseMatMul_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "Ta", TF_FLOAT, status);
    TF_KernelBuilder_TypeConstraint(kb, "Tb", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSparseMatMul", kb, status);
  }
  
  // TODO: 98+ more sparse ops
  // SparseSoftmax, SparseTensorDenseMatMul, SparseTensorDenseAdd
  // SparseAdd, SparseCross, SparseReorder, SparseSlice, etc.
}

}  // namespace mps
}  // namespace tensorflow
