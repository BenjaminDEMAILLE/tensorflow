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

// REAL IMPLEMENTATION: Extended math operations with Metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

struct MPSMathContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> command_queue;
};

extern "C" void* MPSMath_Create(TF_OpKernelConstruction* ctx) {
  auto* kernel_ctx = new MPSMathContext();
  kernel_ctx->device = MTLCreateSystemDefaultDevice();
  kernel_ctx->command_queue = [kernel_ctx->device newCommandQueue];
  return kernel_ctx;
}

extern "C" void MPSMath_Delete(void* kernel) {
  auto* ctx = static_cast<MPSMathContext*>(kernel);
  delete ctx;
}

// Generic unary operation using MPSGraph
static void ComputeUnaryOp(MPSMathContext* ctx, TF_OpKernelContext* tf_ctx,
                            MPSGraphTensor* (^opBlock)(MPSGraph*, MPSGraphTensor*)) {
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_GetInput(tf_ctx, 0, &input_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
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
    
    MPSGraphTensor* outputTensor = opBlock(graph, inputTensor);
    
    size_t data_size = total_size * sizeof(float);
    
    id<MTLBuffer> input_buffer = [ctx->device newBufferWithBytes:TF_TensorData(input_tensor)
                                                           length:data_size
                                                          options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                              shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                           dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: input_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    TF_Tensor* output_tf = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, shape_vec.data(), num_dims, data_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, data_size);
  }
  
  TF_DeleteStatus(status);
}

// Generic binary operation using MPSGraph
static void ComputeBinaryOp(MPSMathContext* ctx, TF_OpKernelContext* tf_ctx,
                             MPSGraphTensor* (^opBlock)(MPSGraph*, MPSGraphTensor*, MPSGraphTensor*)) {
  TF_Status* status = TF_NewStatus();
  
  @autoreleasepool {
    TF_Tensor* input1_tensor = nullptr;
    TF_Tensor* input2_tensor = nullptr;
    
    TF_GetInput(tf_ctx, 0, &input1_tensor, status);
    TF_GetInput(tf_ctx, 1, &input2_tensor, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int num_dims = TF_NumDims(input1_tensor);
    std::vector<NSNumber*> shape_vec;
    size_t total_size = 1;
    
    for (int i = 0; i < num_dims; ++i) {
      int64_t dim = TF_Dim(input1_tensor, i);
      shape_vec.push_back(@(dim));
      total_size *= dim;
    }
    
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* input1Tensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"input1"];
    
    MPSGraphTensor* input2Tensor = [graph placeholderWithShape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                       dataType:MPSDataTypeFloat32
                                                           name:@"input2"];
    
    MPSGraphTensor* outputTensor = opBlock(graph, input1Tensor, input2Tensor);
    
    size_t data_size = total_size * sizeof(float);
    
    id<MTLBuffer> input1_buffer = [ctx->device newBufferWithBytes:TF_TensorData(input1_tensor)
                                                            length:data_size
                                                           options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> input2_buffer = [ctx->device newBufferWithBytes:TF_TensorData(input2_tensor)
                                                            length:data_size
                                                           options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* input1_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input1_buffer
                                                                               shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                            dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* input2_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input2_buffer
                                                                               shape:[NSArray arrayWithObjects:shape_vec.data() count:shape_vec.size()]
                                                                            dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{input1Tensor: input1_data, input2Tensor: input2_data};
    
    MPSGraphTensorData* result = [graph runWithMTLCommandQueue:ctx->command_queue
                                                         feeds:feeds
                                                targetTensors:@[outputTensor]
                                         targetOperations:nil][outputTensor];
    
    TF_Tensor* output_tf = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, shape_vec.data(), num_dims, data_size, status);
    
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(tf_ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    memcpy(TF_TensorData(output_tf), [[result mpsndarray] mpsBuffer].contents, data_size);
  }
  
  TF_DeleteStatus(status);
}

// Unary operations
extern "C" void MPSAbs_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g absoluteWithTensor:t name:@"abs"];
  });
}

extern "C" void MPSSqrt_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g squareRootWithTensor:t name:@"sqrt"];
  });
}

extern "C" void MPSRsqrt_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g reverseSquareRootWithTensor:t name:@"rsqrt"];
  });
}

extern "C" void MPSExp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g exponentWithTensor:t name:@"exp"];
  });
}

extern "C" void MPSLog_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g logarithmWithTensor:t name:@"log"];
  });
}

extern "C" void MPSSin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g sinWithTensor:t name:@"sin"];
  });
}

extern "C" void MPSCos_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g cosWithTensor:t name:@"cos"];
  });
}

extern "C" void MPSTan_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g tanWithTensor:t name:@"tan"];
  });
}

extern "C" void MPSAsin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g asinWithTensor:t name:@"asin"];
  });
}

extern "C" void MPSAcos_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g acosWithTensor:t name:@"acos"];
  });
}

extern "C" void MPSAtan_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g atanWithTensor:t name:@"atan"];
  });
}

extern "C" void MPSSinh_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g sinhWithTensor:t name:@"sinh"];
  });
}

extern "C" void MPSCosh_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g coshWithTensor:t name:@"cosh"];
  });
}

extern "C" void MPSTanh_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g tanhWithTensor:t name:@"tanh"];
  });
}

extern "C" void MPSCeil_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g ceilWithTensor:t name:@"ceil"];
  });
}

extern "C" void MPSFloor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g floorWithTensor:t name:@"floor"];
  });
}

extern "C" void MPSRound_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g roundWithTensor:t name:@"round"];
  });
}

extern "C" void MPSSquare_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g squareWithTensor:t name:@"square"];
  });
}

extern "C" void MPSNegate_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g negativeWithTensor:t name:@"negate"];
  });
}

extern "C" void MPSReciprocal_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g reciprocalWithTensor:t name:@"reciprocal"];
  });
}

extern "C" void MPSSign_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeUnaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t) {
    return [g signWithTensor:t name:@"sign"];
  });
}

// Binary operations
extern "C" void MPSAdd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g additionWithPrimaryTensor:t1 secondaryTensor:t2 name:@"add"];
  });
}

extern "C" void MPSSubtract_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g subtractionWithPrimaryTensor:t1 secondaryTensor:t2 name:@"subtract"];
  });
}

extern "C" void MPSMultiply_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g multiplicationWithPrimaryTensor:t1 secondaryTensor:t2 name:@"multiply"];
  });
}

extern "C" void MPSDivide_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g divisionWithPrimaryTensor:t1 secondaryTensor:t2 name:@"divide"];
  });
}

extern "C" void MPSPow_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g powerWithPrimaryTensor:t1 secondaryTensor:t2 name:@"pow"];
  });
}

extern "C" void MPSMinimum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g minimumWithPrimaryTensor:t1 secondaryTensor:t2 name:@"minimum"];
  });
}

extern "C" void MPSMaximum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g maximumWithPrimaryTensor:t1 secondaryTensor:t2 name:@"maximum"];
  });
}

extern "C" void MPSFloorMod_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* math_ctx = static_cast<MPSMathContext*>(kernel);
  ComputeBinaryOp(math_ctx, ctx, ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* t1, MPSGraphTensor* t2) {
    return [g floorModuloWithPrimaryTensor:t1 secondaryTensor:t2 name:@"floormod"];
  });
}

}  // namespace mps
}  // namespace tensorflow
