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

// REAL IMPLEMENTATION: Tensor manipulation operations

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <vector>

namespace tensorflow {
namespace mps {

struct MPSTensorManipContext {
  std::vector<int64_t> perm;  // For transpose
  int64_t axis;               // For concat/split
  
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
};

extern "C" void* MPSTensorManip_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSTensorManipContext();
  TF_Status* status = TF_NewStatus();
  
  // Try to get perm for transpose
  int64_t* perm = nullptr;
  int perm_len = 0;
  TF_OpKernelConstruction_GetAttrInt64List(ctx, "perm", &perm, &perm_len, status);
  if (TF_GetCode(status) == TF_OK && perm_len > 0) {
    kernel_ctx->perm.assign(perm, perm + perm_len);
  }
  
  // Try to get axis for concat/split
  kernel_ctx->axis = 0;
  TF_OpKernelConstruction_GetAttrInt64(ctx, "axis", &kernel_ctx->axis, status);
  
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  
  TF_DeleteStatus(status);
  return kernel_ctx;
}

extern "C" void MPSTensorManip_Delete(void* kernel) {
  auto* ctx = static_cast<MPSTensorManipContext*>(kernel);
  delete ctx;
}

// Reshape
extern "C" void MPSReshape_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* manip_ctx = static_cast<MPSTensorManipContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* shape_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &shape_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get input shape
    int num_dims = TF_NumDims(input_tensor);
    size_t total_size = 1;
    for (int i = 0; i < num_dims; ++i) {
      total_size *= TF_Dim(input_tensor, i);
    }
    
    // Get new shape
    int new_num_dims = TF_Dim(shape_tensor, 0);
    int64_t* new_shape_data = static_cast<int64_t*>(TF_TensorData(shape_tensor));
    std::vector<int64_t> new_shape(new_shape_data, new_shape_data + new_num_dims);
    
    // Reshape is just a metadata operation in TensorFlow
    // Copy data as-is with new shape
    size_t data_size = total_size * sizeof(float);
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, new_shape.data(), 
                                             new_num_dims, data_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), TF_TensorData(input_tensor), data_size);
  }
  
  TF_DeleteStatus(status);
}

// Transpose
extern "C" void MPSTranspose_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* manip_ctx = static_cast<MPSTensorManipContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_GetInput(ctx, 0, &input_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(input_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(input_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Convert perm to NSArray
    NSMutableArray* perm_array = [NSMutableArray array];
    for (int64_t p : manip_ctx->perm) {
      [perm_array addObject:@(p)];
    }
    
    MPSGraphTensor* outputTensor = [graph transposeTensor:inputTensor
                                            permutation:perm_array
                                                   name:@"transpose"];
    
    size_t data_size = total_size * sizeof(float);
    
    id<MTLBuffer> input_buffer = [manip_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                                 length:data_size
                                                                options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                              shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                           dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: input_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:manip_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    // Calculate output shape after transpose
    std::vector<int64_t> output_shape(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      output_shape[i] = TF_Dim(input_tensor, manip_ctx->perm[i]);
    }
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_shape.data(), 
                                             num_dims, data_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, data_size);
  }
  
  TF_DeleteStatus(status);
}

// Concat
extern "C" void MPSConcat_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* manip_ctx = static_cast<MPSTensorManipContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    // Get number of inputs
    int num_inputs = TF_NumInputs(ctx);
    
    std::vector<TF_Tensor*> input_tensors;
    std::vector<MPSGraphTensor*> mps_tensors;
    std::vector<MPSGraphTensorData*> mps_data;
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    // Get all input tensors
    for (int i = 0; i < num_inputs; ++i) {
      TF_Tensor* input_tensor = nullptr;
      TF_GetInput(ctx, i, &input_tensor, status);
      
      if (TF_GetCode(status) != TF_OK) {
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteStatus(status);
        return;
      }
      
      input_tensors.push_back(input_tensor);
      
      // Create MPS tensor
      int num_dims = TF_NumDims(input_tensor);
      std::vector<NSNumber*> shape_vec;
      size_t total_size = 1;
      
      for (int j = 0; j < num_dims; ++j) {
        int64_t dim = TF_Dim(input_tensor, j);
        shape_vec.push_back(@(dim));
        total_size *= dim;
      }
      
      MPSGraphTensor* mps_tensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                        dataType:MPSDataTypeFloat32
                                                            name:[NSString stringWithFormat:@"input%d", i]];
      mps_tensors.push_back(mps_tensor);
      
      size_t data_size = total_size * sizeof(float);
      id<MTLBuffer> buffer = [manip_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                             length:data_size
                                                            options:MTLResourceStorageModeShared];
      
      MPSGraphTensorData* data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                          shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                       dataType:MPSDataTypeFloat32];
      mps_data.push_back(data);
    }
    
    // Concatenate
    MPSGraphTensor* outputTensor = [graph concatTensors:[NSArray arrayWithObjects:mps_tensors.data() count:mps_tensors.size()]
                                              dimension:manip_ctx->axis
                                                   name:@"concat"];
    
    // Create feeds dictionary
    NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
    for (size_t i = 0; i < mps_tensors.size(); ++i) {
      feeds[mps_tensors[i]] = mps_data[i];
    }
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:manip_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    // Calculate output shape
    int num_dims = TF_NumDims(input_tensors[0]);
    std::vector<int64_t> output_shape;
    
    for (int i = 0; i < num_dims; ++i) {
      if (i == manip_ctx->axis) {
        int64_t concat_dim = 0;
        for (auto* tensor : input_tensors) {
          concat_dim += TF_Dim(tensor, i);
        }
        output_shape.push_back(concat_dim);
      } else {
        output_shape.push_back(TF_Dim(input_tensors[0], i));
      }
    }
    
    size_t output_size = sizeof(float);
    for (int64_t dim : output_shape) {
      output_size *= dim;
    }
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_shape.data(), 
                                             num_dims, output_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, output_size);
  }
  
  TF_DeleteStatus(status);
}

// Slice
extern "C" void MPSSlice_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* manip_ctx = static_cast<MPSTensorManipContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* begin_tensor = nullptr;
    TF_Tensor* size_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &begin_tensor, status);
    TF_GetInput(ctx, 2, &size_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(input_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(input_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    // Get begin and size arrays
    int64_t* begin_data = static_cast<int64_t*>(TF_TensorData(begin_tensor));
    int64_t* size_data = static_cast<int64_t*>(TF_TensorData(size_tensor));
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Convert to NSArray
    NSMutableArray* starts = [NSMutableArray array];
    NSMutableArray* ends = [NSMutableArray array];
    NSMutableArray* strides = [NSMutableArray array];
    
    std::vector<int64_t> output_shape;
    
    for (int i = 0; i < num_dims; ++i) {
      [starts addObject:@(begin_data[i])];
      int64_t end = (size_data[i] == -1) ? TF_Dim(input_tensor, i) : (begin_data[i] + size_data[i]);
      [ends addObject:@(end)];
      [strides addObject:@(1)];
      
      output_shape.push_back(end - begin_data[i]);
    }
    
    MPSGraphTensor* outputTensor = [graph sliceTensor:inputTensor
                                            starts:starts
                                              ends:ends
                                           strides:strides
                                              name:@"slice"];
    
    size_t data_size = total_size * sizeof(float);
    
    id<MTLBuffer> input_buffer = [manip_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                                 length:data_size
                                                                options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                              shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                           dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: input_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:manip_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    size_t output_size = sizeof(float);
    for (int64_t dim : output_shape) {
      output_size *= dim;
    }
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_shape.data(), 
                                             num_dims, output_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, output_size);
  }
  
  TF_DeleteStatus(status);
}

// Pad
extern "C" void MPSPad_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* manip_ctx = static_cast<MPSTensorManipContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* paddings_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &paddings_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(input_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(input_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    // Get paddings [n, 2] array
    int64_t* paddings_data = static_cast<int64_t*>(TF_TensorData(paddings_tensor));
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Convert paddings to NSArray
    NSMutableArray* left_padding = [NSMutableArray array];
    NSMutableArray* right_padding = [NSMutableArray array];
    
    std::vector<int64_t> output_shape;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t left = paddings_data[i * 2];
      int64_t right = paddings_data[i * 2 + 1];
      
      [left_padding addObject:@(left)];
      [right_padding addObject:@(right)];
      
      output_shape.push_back(TF_Dim(input_tensor, i) + left + right);
    }
    
    MPSGraphTensor* outputTensor = [graph padTensor:inputTensor
                                  withPaddingMode:MPSGraphPaddingModeConstant
                                      leftPadding:left_padding
                                     rightPadding:right_padding
                                    constantValue:0.0
                                             name:@"pad"];
    
    size_t input_size = total_size * sizeof(float);
    
    id<MTLBuffer> input_buffer = [manip_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                                 length:input_size
                                                                options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                              shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                           dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: input_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:manip_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    size_t output_size = sizeof(float);
    for (int64_t dim : output_shape) {
      output_size *= dim;
    }
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_shape.data(), 
                                             num_dims, output_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, output_size);
  }
  
  TF_DeleteStatus(status);
}

// Tile
extern "C" void MPSTile_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* manip_ctx = static_cast<MPSTensorManipContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* multiples_tensor = nullptr;
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &multiples_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(input_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(input_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    // Get multiples
    int64_t* multiples_data = static_cast<int64_t*>(TF_TensorData(multiples_tensor));
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Tile using repeat/broadcast
    MPSGraphTensor* outputTensor = inputTensor;
    std::vector<int64_t> output_shape;
    
    for (int i = 0; i < num_dims; ++i) {
      if (multiples_data[i] > 1) {
        outputTensor = [graph tileTensor:outputTensor
                          withMultiplier:multiples_data[i]
                                    axis:i
                                    name:[NSString stringWithFormat:@"tile_%d", i]];
      }
      output_shape.push_back(TF_Dim(input_tensor, i) * multiples_data[i]);
    }
    
    size_t input_size = total_size * sizeof(float);
    
    id<MTLBuffer> input_buffer = [manip_ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                                 length:input_size
                                                                options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                              shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                           dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: input_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:manip_ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    size_t output_size = sizeof(float);
    for (int64_t dim : output_shape) {
      output_size *= dim;
    }
    
    TF_Tensor* output_tf = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_shape.data(), 
                                             num_dims, output_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, output_size);
  }
  
  TF_DeleteStatus(status);
}

}  // namespace mps
}  // namespace tensorflow
