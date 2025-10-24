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

// Extended reduction operations for MPS backend

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace {
id<MTLDevice> GetMetalDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

NSArray* GetShapeArray(TF_Tensor* tensor) {
  int nd = TF_NumDims(tensor);
  NSMutableArray* shape = [NSMutableArray array];
  for (int i = 0; i < nd; i++) {
    [shape addObject:@(TF_Dim(tensor, i))];
  }
  return shape;
}
}

// Common reduction kernel structure
struct MPSReductionCtx {
  bool keep_dims;
};

// ===== Sum =====
extern "C" void* MPSSum_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSReductionCtx();
  TF_Status* status = TF_NewStatus();
  TF_Bool keep_dims = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "keep_dims", &keep_dims, status);
  kernel_ctx->keep_dims = (keep_dims != 0);
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSSum_Delete(void* kernel) {
  delete reinterpret_cast<MPSReductionCtx*>(kernel);
}

extern "C" void MPSSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axes = nullptr;
  TF_GetInput(ctx, 1, &axes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int num_axes = TF_NumElements(axes);
  int32_t* axes_data = (int32_t*)TF_TensorData(axes);
  NSMutableArray* axesArray = [NSMutableArray array];
  for (int i = 0; i < num_axes; i++) {
    [axesArray addObject:@(axes_data[i])];
  }
  
  MPSGraphTensor* result = [graph reductionSumWithTensor:inputTensor axes:axesArray name:@"sum"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(float));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== Mean =====
extern "C" void* MPSMean_Create(TF_OpKernelConstruction* ctx) {
  return MPSSum_Create(ctx);
}

extern "C" void MPSMean_Delete(void* kernel) {
  MPSSum_Delete(kernel);
}

extern "C" void MPSMean_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axes = nullptr;
  TF_GetInput(ctx, 1, &axes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int num_axes = TF_NumElements(axes);
  int32_t* axes_data = (int32_t*)TF_TensorData(axes);
  NSMutableArray* axesArray = [NSMutableArray array];
  for (int i = 0; i < num_axes; i++) {
    [axesArray addObject:@(axes_data[i])];
  }
  
  MPSGraphTensor* result = [graph reductionMeanWithTensor:inputTensor axes:axesArray name:@"mean"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(float));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== Max =====
extern "C" void* MPSMax_Create(TF_OpKernelConstruction* ctx) {
  return MPSSum_Create(ctx);
}

extern "C" void MPSMax_Delete(void* kernel) {
  MPSSum_Delete(kernel);
}

extern "C" void MPSMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axes = nullptr;
  TF_GetInput(ctx, 1, &axes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int num_axes = TF_NumElements(axes);
  int32_t* axes_data = (int32_t*)TF_TensorData(axes);
  NSMutableArray* axesArray = [NSMutableArray array];
  for (int i = 0; i < num_axes; i++) {
    [axesArray addObject:@(axes_data[i])];
  }
  
  MPSGraphTensor* result = [graph reductionMaximumWithTensor:inputTensor axes:axesArray name:@"max"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(float));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== Min =====
extern "C" void* MPSMin_Create(TF_OpKernelConstruction* ctx) {
  return MPSSum_Create(ctx);
}

extern "C" void MPSMin_Delete(void* kernel) {
  MPSSum_Delete(kernel);
}

extern "C" void MPSMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axes = nullptr;
  TF_GetInput(ctx, 1, &axes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int num_axes = TF_NumElements(axes);
  int32_t* axes_data = (int32_t*)TF_TensorData(axes);
  NSMutableArray* axesArray = [NSMutableArray array];
  for (int i = 0; i < num_axes; i++) {
    [axesArray addObject:@(axes_data[i])];
  }
  
  MPSGraphTensor* result = [graph reductionMinimumWithTensor:inputTensor axes:axesArray name:@"min"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(float));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== Prod =====
extern "C" void* MPSProd_Create(TF_OpKernelConstruction* ctx) {
  return MPSSum_Create(ctx);
}

extern "C" void MPSProd_Delete(void* kernel) {
  MPSSum_Delete(kernel);
}

extern "C" void MPSProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axes = nullptr;
  TF_GetInput(ctx, 1, &axes, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int num_axes = TF_NumElements(axes);
  int32_t* axes_data = (int32_t*)TF_TensorData(axes);
  NSMutableArray* axesArray = [NSMutableArray array];
  for (int i = 0; i < num_axes; i++) {
    [axesArray addObject:@(axes_data[i])];
  }
  
  MPSGraphTensor* result = [graph reductionProductWithTensor:inputTensor axes:axesArray name:@"prod"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(float));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== ArgMax =====
extern "C" void* MPSArgMax_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSReductionCtx();
  kernel_ctx->keep_dims = false;
  return kernel_ctx;
}

extern "C" void MPSArgMax_Delete(void* kernel) {
  delete reinterpret_cast<MPSReductionCtx*>(kernel);
}

extern "C" void MPSArgMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axis_tensor = nullptr;
  TF_GetInput(ctx, 1, &axis_tensor, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int32_t axis = *(int32_t*)TF_TensorData(axis_tensor);
  MPSGraphTensor* result = [graph reductionArgMaximumWithTensor:inputTensor axis:axis name:@"argmax"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_INT32, dims, nd, nelems * sizeof(int32_t), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(int32_t));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== ArgMin =====
extern "C" void* MPSArgMin_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSReductionCtx();
  kernel_ctx->keep_dims = false;
  return kernel_ctx;
}

extern "C" void MPSArgMin_Delete(void* kernel) {
  delete reinterpret_cast<MPSReductionCtx*>(kernel);
}

extern "C" void MPSArgMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axis_tensor = nullptr;
  TF_GetInput(ctx, 1, &axis_tensor, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int32_t axis = *(int32_t*)TF_TensorData(axis_tensor);
  MPSGraphTensor* result = [graph reductionArgMinimumWithTensor:inputTensor axis:axis name:@"argmin"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_INT32, dims, nd, nelems * sizeof(int32_t), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(int32_t));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== CumulativeSum =====
extern "C" void* MPSCumulativeSum_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSReductionCtx();
  kernel_ctx->keep_dims = false;
  return kernel_ctx;
}

extern "C" void MPSCumulativeSum_Delete(void* kernel) {
  delete reinterpret_cast<MPSReductionCtx*>(kernel);
}

extern "C" void MPSCumulativeSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axis_tensor = nullptr;
  TF_GetInput(ctx, 1, &axis_tensor, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int32_t axis = *(int32_t*)TF_TensorData(axis_tensor);
  MPSGraphTensor* result = [graph cumulativeSumWithTensor:inputTensor axis:axis name:@"cumsum"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(float));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}

// ===== CumulativeProd =====
extern "C" void* MPSCumulativeProd_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSReductionCtx();
  kernel_ctx->keep_dims = false;
  return kernel_ctx;
}

extern "C" void MPSCumulativeProd_Delete(void* kernel) {
  delete reinterpret_cast<MPSReductionCtx*>(kernel);
}

extern "C" void MPSCumulativeProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* kernel_ctx = static_cast<MPSReductionCtx*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  TF_Tensor* axis_tensor = nullptr;
  TF_GetInput(ctx, 1, &axis_tensor, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  MPSGraph* graph = [[MPSGraph alloc] init];
  MPSGraphTensor* inputTensor = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:@"input"];
  
  int32_t axis = *(int32_t*)TF_TensorData(axis_tensor);
  MPSGraphTensor* result = [graph cumulativeProductWithTensor:inputTensor axis:axis name:@"cumprod"];
  
  MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:[GetMetalDevice() newBufferWithBytes:TF_TensorData(input) length:TF_TensorByteSize(input) options:MTLResourceStorageModeShared] shape:GetShapeArray(input) dataType:MPSDataTypeFloat32];
  
  MPSGraphTensorData* outputData = [graph runWithFeeds:@{inputTensor: inputData} targetTensors:@[result] targetOperations:nil executionDescriptor:nil];
  
  int nd = (int)[outputData.shape count];
  int64_t dims[8], nelems = 1;
  for (int i = 0; i < nd; ++i) { dims[i] = [outputData.shape[i] longLongValue]; nelems *= dims[i]; }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  memcpy(TF_TensorData(output), [outputData mpsndarray].data.contents, nelems * sizeof(float));
  
  [inputData release];
  [graph release];
  }
  
  TF_DeleteStatus(status);
}
