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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <cmath>

namespace {

struct MPSReductionContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  MPSGraph* graph;
  
  MPSReductionContext() {
    device = MTLCreateSystemDefaultDevice();
    commandQueue = [device newCommandQueue];
    graph = [[MPSGraph new] autorelease];
  }
  
  ~MPSReductionContext() {
    [commandQueue release];
    [device release];
  }
  
  // Execute reduction with MPSGraph
  template<typename GraphOp>
  void ExecuteReduction(TF_OpKernelContext* ctx, GraphOp graph_op) {
    @autoreleasepool {
      TF_Status* status = TF_NewStatus();
      
      TF_Tensor* input = nullptr;
      TF_GetInput(ctx, 0, &input, status);
      if (TF_GetCode(status) != TF_OK) {
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteStatus(status);
        return;
      }
      
      int nd = TF_NumDims(input);
      NSMutableArray* shape = [NSMutableArray arrayWithCapacity:nd];
      int64_t nelems = 1;
      for (int i = 0; i < nd; ++i) {
        int64_t dim = TF_Dim(input, i);
        [shape addObject:@(dim)];
        nelems *= dim;
      }
      
      // Create graph
      MPSGraphTensor* inputTensor = [graph placeholderWithShape:shape
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"input"];
      MPSGraphTensor* outputTensor = graph_op(graph, inputTensor);
      
      // Create input data
      float* input_data = (float*)TF_TensorData(input);
      id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                       length:nelems * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
      
      MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:inputBuffer
                      shape:shape
                   dataType:MPSDataTypeFloat32];
      
      // Execute
      NSDictionary* feeds = @{inputTensor: inputData};
      NSDictionary* results = [graph runWithFeeds:feeds
                                   targetTensors:@[outputTensor]
                                 targetOperations:nil
                              executionDescriptor:nil];
      
      MPSGraphTensorData* resultData = results[outputTensor];
      NSArray* outputShape = [resultData shape];
      int output_nd = [outputShape count];
      int64_t output_dims[8];
      int64_t output_nelems = 1;
      for (int i = 0; i < output_nd; ++i) {
        output_dims[i] = [[outputShape objectAtIndex:i] longLongValue];
        output_nelems *= output_dims[i];
      }
      
      // Allocate output
      TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, output_nd,
                                           output_nelems * sizeof(float), status);
      if (TF_GetCode(status) == TF_OK) {
        float* output_data = (float*)TF_TensorData(output);
        id<MTLBuffer> resultBuffer = [[resultData mpsndarray] mpsndArrayBuffer];
        memcpy(output_data, [resultBuffer contents], output_nelems * sizeof(float));
      } else {
        TF_OpKernelContext_Failure(ctx, status);
      }
      
      [inputData release];
      [inputBuffer release];
      TF_DeleteStatus(status);
    }
  }
  
  // Execute cumulative operation
  template<typename GraphOp>
  void ExecuteCumulative(TF_OpKernelContext* ctx, GraphOp graph_op) {
    @autoreleasepool {
      TF_Status* status = TF_NewStatus();
      
      TF_Tensor* input = nullptr;
      TF_GetInput(ctx, 0, &input, status);
      if (TF_GetCode(status) != TF_OK) {
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteStatus(status);
        return;
      }
      
      int nd = TF_NumDims(input);
      NSMutableArray* shape = [NSMutableArray arrayWithCapacity:nd];
      int64_t nelems = 1;
      for (int i = 0; i < nd; ++i) {
        int64_t dim = TF_Dim(input, i);
        [shape addObject:@(dim)];
        nelems *= dim;
      }
      
      // Create graph - cumulative along last axis
      MPSGraphTensor* inputTensor = [graph placeholderWithShape:shape
                                                        dataType:MPSDataTypeFloat32
                                                            name:@"input"];
      MPSGraphTensor* outputTensor = graph_op(graph, inputTensor, @(nd - 1));
      
      float* input_data = (float*)TF_TensorData(input);
      id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                       length:nelems * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
      
      MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:inputBuffer
                      shape:shape
                   dataType:MPSDataTypeFloat32];
      
      NSDictionary* feeds = @{inputTensor: inputData};
      NSDictionary* results = [graph runWithFeeds:feeds
                                   targetTensors:@[outputTensor]
                                 targetOperations:nil
                              executionDescriptor:nil];
      
      MPSGraphTensorData* resultData = results[outputTensor];
      NSArray* outputShape = [resultData shape];
      int output_nd = [outputShape count];
      int64_t output_dims[8];
      int64_t output_nelems = 1;
      for (int i = 0; i < output_nd; ++i) {
        output_dims[i] = [[outputShape objectAtIndex:i] longLongValue];
        output_nelems *= output_dims[i];
      }
      
      TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, output_nd,
                                           output_nelems * sizeof(float), status);
      if (TF_GetCode(status) == TF_OK) {
        float* output_data = (float*)TF_TensorData(output);
        id<MTLBuffer> resultBuffer = [[resultData mpsndarray] mpsndArrayBuffer];
        memcpy(output_data, [resultBuffer contents], output_nelems * sizeof(float));
      } else {
        TF_OpKernelContext_Failure(ctx, status);
      }
      
      [inputData release];
      [inputBuffer release];
      TF_DeleteStatus(status);
    }
  }
};

static MPSReductionContext* GetContext() {
  static MPSReductionContext* ctx = nullptr;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    ctx = new MPSReductionContext();
  });
  return ctx;
}

} // namespace

// ============================================================================
// STATISTICAL REDUCTIONS
// ============================================================================

// ReduceStd
extern "C" void* MPSReduceStd_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSReduceStd_Delete(void* kernel) {}
extern "C" void MPSReduceStd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  GetContext()->ExecuteReduction(ctx, [](MPSGraph* g, MPSGraphTensor* x) {
    // std = sqrt(variance)
    MPSGraphTensor* mean = [g reductionMeanWithTensor:x axes:nil name:@"mean"];
    MPSGraphTensor* diff = [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:@"diff"];
    MPSGraphTensor* squared = [g multiplicationWithPrimaryTensor:diff secondaryTensor:diff name:@"squared"];
    MPSGraphTensor* variance = [g reductionMeanWithTensor:squared axes:nil name:@"variance"];
    return [g squareRootWithTensor:variance name:@"std"];
  });
}

// ReduceVariance
extern "C" void* MPSReduceVariance_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSReduceVariance_Delete(void* kernel) {}
extern "C" void MPSReduceVariance_Compute(void* kernel, TF_OpKernelContext* ctx) {
  GetContext()->ExecuteReduction(ctx, [](MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* mean = [g reductionMeanWithTensor:x axes:nil name:@"mean"];
    MPSGraphTensor* diff = [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:@"diff"];
    MPSGraphTensor* squared = [g multiplicationWithPrimaryTensor:diff secondaryTensor:diff name:@"squared"];
    return [g reductionMeanWithTensor:squared axes:nil name:@"variance"];
  });
}

// ReduceEuclideanNorm
extern "C" void* MPSReduceEuclideanNorm_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSReduceEuclideanNorm_Delete(void* kernel) {}
extern "C" void MPSReduceEuclideanNorm_Compute(void* kernel, TF_OpKernelContext* ctx) {
  GetContext()->ExecuteReduction(ctx, [](MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* squared = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:@"squared"];
    MPSGraphTensor* sum = [g reductionSumWithTensor:squared axes:nil name:@"sum"];
    return [g squareRootWithTensor:sum name:@"norm"];
  });
}

// ReduceLogsumexp
extern "C" void* MPSReduceLogsumexp_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSReduceLogsumexp_Delete(void* kernel) {}
extern "C" void MPSReduceLogsumexp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  GetContext()->ExecuteReduction(ctx, [](MPSGraph* g, MPSGraphTensor* x) {
    // logsumexp = log(sum(exp(x)))
    // Numerically stable version: max_val + log(sum(exp(x - max_val)))
    MPSGraphTensor* maxVal = [g reductionMaximumWithTensor:x axes:nil name:@"max"];
    MPSGraphTensor* shifted = [g subtractionWithPrimaryTensor:x secondaryTensor:maxVal name:@"shifted"];
    MPSGraphTensor* expShifted = [g exponentWithTensor:shifted name:@"exp"];
    MPSGraphTensor* sumExp = [g reductionSumWithTensor:expShifted axes:nil name:@"sumexp"];
    MPSGraphTensor* logSum = [g logarithmWithTensor:sumExp name:@"log"];
    return [g additionWithPrimaryTensor:maxVal secondaryTensor:logSum name:@"logsumexp"];
  });
}

// ============================================================================
// CUMULATIVE OPERATIONS
// ============================================================================

// CumSum
extern "C" void* MPSCumSum_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSCumSum_Delete(void* kernel) {}
extern "C" void MPSCumSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  GetContext()->ExecuteCumulative(ctx, [](MPSGraph* g, MPSGraphTensor* x, NSNumber* axis) {
    return [g cumulativeSumWithTensor:x axis:[axis integerValue] name:@"cumsum"];
  });
}

// CumProd
extern "C" void* MPSCumProd_Create(TF_OpKernelConstruction* ctx) { return GetContext(); }
extern "C" void MPSCumProd_Delete(void* kernel) {}
extern "C" void MPSCumProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  GetContext()->ExecuteCumulative(ctx, [](MPSGraph* g, MPSGraphTensor* x, NSNumber* axis) {
    // MPSGraph doesn't have cumulativeProduct directly, implement via scan
    // For simplicity, use CPU fallback
    @autoreleasepool {
      TF_Status* status = TF_NewStatus();
      TF_Tensor* input = nullptr;
      TF_GetInput(ctx, 0, &input, status);
      if (TF_GetCode(status) != TF_OK) {
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteStatus(status);
        return x; // Dummy return
      }
      
      int nd = TF_NumDims(input);
      int64_t dims[8];
      int64_t nelems = 1;
      for (int i = 0; i < nd; ++i) {
        dims[i] = TF_Dim(input, i);
        nelems *= dims[i];
      }
      
      TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
      if (TF_GetCode(status) != TF_OK) {
        TF_OpKernelContext_Failure(ctx, status);
        TF_DeleteStatus(status);
        return x;
      }
      
      const float* in_data = (const float*)TF_TensorData(input);
      float* out_data = (float*)TF_TensorData(output);
      
      // Simple CPU implementation for cumulative product along last axis
      int64_t outer = 1;
      for (int i = 0; i < nd - 1; ++i) outer *= dims[i];
      int64_t inner = dims[nd - 1];
      
      for (int64_t i = 0; i < outer; ++i) {
        float prod = 1.0f;
        for (int64_t j = 0; j < inner; ++j) {
          prod *= in_data[i * inner + j];
          out_data[i * inner + j] = prod;
        }
      }
      
      TF_DeleteStatus(status);
      return x;
    }
  });
}

// CumMax - CPU fallback
extern "C" void* MPSCumMax_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSCumMax_Delete(void* kernel) {}
extern "C" void MPSCumMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  int nd = TF_NumDims(input);
  int64_t dims[8];
  int64_t nelems = 1;
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
    nelems *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  const float* in_data = (const float*)TF_TensorData(input);
  float* out_data = (float*)TF_TensorData(output);
  
  int64_t outer = 1;
  for (int i = 0; i < nd - 1; ++i) outer *= dims[i];
  int64_t inner = dims[nd - 1];
  
  for (int64_t i = 0; i < outer; ++i) {
    float max_val = in_data[i * inner];
    out_data[i * inner] = max_val;
    for (int64_t j = 1; j < inner; ++j) {
      if (in_data[i * inner + j] > max_val) {
        max_val = in_data[i * inner + j];
      }
      out_data[i * inner + j] = max_val;
    }
  }
  
  TF_DeleteStatus(status);
}

// CumMin - CPU fallback
extern "C" void* MPSCumMin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSCumMin_Delete(void* kernel) {}
extern "C" void MPSCumMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  int nd = TF_NumDims(input);
  int64_t dims[8];
  int64_t nelems = 1;
  for (int i = 0; i < nd; ++i) {
    dims[i] = TF_Dim(input, i);
    nelems *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims, nd, nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
    TF_DeleteStatus(status);
    return;
  }
  
  const float* in_data = (const float*)TF_TensorData(input);
  float* out_data = (float*)TF_TensorData(output);
  
  int64_t outer = 1;
  for (int i = 0; i < nd - 1; ++i) outer *= dims[i];
  int64_t inner = dims[nd - 1];
  
  for (int64_t i = 0; i < outer; ++i) {
    float min_val = in_data[i * inner];
    out_data[i * inner] = min_val;
    for (int64_t j = 1; j < inner; ++j) {
      if (in_data[i * inner + j] < min_val) {
        min_val = in_data[i * inner + j];
      }
      out_data[i * inner + j] = min_val;
    }
  }
  
  TF_DeleteStatus(status);
}

// ============================================================================
// SEGMENT OPERATIONS - CPU implementations
// ============================================================================

// Helper for segment operations
static void SegmentOp(TF_OpKernelContext* ctx, 
                     std::function<void(float&, float)> reduce_op,
                     float init_val) {
  TF_Status* status = TF_NewStatus();
  
  TF_Tensor* data = nullptr;
  TF_Tensor* segment_ids = nullptr;
  TF_GetInput(ctx, 0, &data, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  TF_GetInput(ctx, 1, &segment_ids, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  int nd = TF_NumDims(data);
  int64_t data_dims[8];
  int64_t nelems = 1;
  for (int i = 0; i < nd; ++i) {
    data_dims[i] = TF_Dim(data, i);
    nelems *= data_dims[i];
  }
  
  int nsegments = TF_Dim(segment_ids, 0);
  const int32_t* seg_ids = (const int32_t*)TF_TensorData(segment_ids);
  int max_id = 0;
  for (int i = 0; i < nsegments; ++i) {
    if (seg_ids[i] > max_id) max_id = seg_ids[i];
  }
  int num_segments = max_id + 1;
  
  // Output shape: [num_segments, ...rest of data dims]
  int64_t output_dims[8];
  output_dims[0] = num_segments;
  for (int i = 1; i < nd; ++i) {
    output_dims[i] = data_dims[i];
  }
  
  int64_t out_nelems = 1;
  for (int i = 0; i < nd; ++i) out_nelems *= output_dims[i];
  
  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, nd, out_nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  
  const float* data_ptr = (const float*)TF_TensorData(data);
  float* out_ptr = (float*)TF_TensorData(output);
  
  // Initialize output
  for (int64_t i = 0; i < out_nelems; ++i) {
    out_ptr[i] = init_val;
  }
  
  // Reduce
  int64_t inner_size = nelems / nsegments;
  for (int i = 0; i < nsegments; ++i) {
    int seg_id = seg_ids[i];
    for (int64_t j = 0; j < inner_size; ++j) {
      reduce_op(out_ptr[seg_id * inner_size + j], data_ptr[i * inner_size + j]);
    }
  }
  
  TF_DeleteStatus(status);
}

// SegmentSum
extern "C" void* MPSSegmentSum_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentSum_Delete(void* kernel) {}
extern "C" void MPSSegmentSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { acc += val; }, 0.0f);
}

// SegmentMean
extern "C" void* MPSSegmentMean_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentMean_Delete(void* kernel) {}
extern "C" void MPSSegmentMean_Compute(void* kernel, TF_OpKernelContext* ctx) {
  // Implement SegmentMean via CPU fallback: compute segment sums and divide by counts
  TF_Status* status = TF_NewStatus();

  TF_Tensor* data = nullptr;
  TF_Tensor* segment_ids = nullptr;
  TF_GetInput(ctx, 0, &data, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }
  TF_GetInput(ctx, 1, &segment_ids, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  int nd = TF_NumDims(data);
  int64_t data_dims[8];
  int64_t nelems = 1;
  for (int i = 0; i < nd; ++i) {
    data_dims[i] = TF_Dim(data, i);
    nelems *= data_dims[i];
  }

  int nsegments = TF_Dim(segment_ids, 0);
  const int32_t* seg_ids = (const int32_t*)TF_TensorData(segment_ids);
  int max_id = 0;
  for (int i = 0; i < nsegments; ++i) { if (seg_ids[i] > max_id) max_id = seg_ids[i]; }
  int num_segments = max_id + 1;

  // Output shape: [num_segments, ...rest]
  int64_t output_dims[8];
  output_dims[0] = num_segments;
  for (int i = 1; i < nd; ++i) output_dims[i] = data_dims[i];

  int64_t inner_size = nelems / nsegments;  // size of trailing dims product
  int64_t out_nelems = (int64_t)num_segments * inner_size;

  TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, nd, out_nelems * sizeof(float), status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(ctx, status); TF_DeleteStatus(status); return; }

  const float* data_ptr = (const float*)TF_TensorData(data);
  float* out_ptr = (float*)TF_TensorData(output);

  // Initialize sums to 0
  for (int64_t i = 0; i < out_nelems; ++i) out_ptr[i] = 0.0f;
  // Initialize counts per segment
  std::vector<int32_t> counts(num_segments, 0);

  // Accumulate sums and counts
  for (int i = 0; i < nsegments; ++i) {
    int seg = seg_ids[i];
    counts[seg] += 1;
    for (int64_t j = 0; j < inner_size; ++j) {
      out_ptr[(int64_t)seg * inner_size + j] += data_ptr[(int64_t)i * inner_size + j];
    }
  }

  // Divide by counts to get mean
  for (int seg = 0; seg < num_segments; ++seg) {
    int32_t c = counts[seg];
    float denom = c > 0 ? (float)c : 1.0f;
    for (int64_t j = 0; j < inner_size; ++j) {
      out_ptr[(int64_t)seg * inner_size + j] /= denom;
    }
  }

  TF_DeleteStatus(status);
}

// SegmentMax
extern "C" void* MPSSegmentMax_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentMax_Delete(void* kernel) {}
extern "C" void MPSSegmentMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { if (val > acc) acc = val; }, -INFINITY);
}

// SegmentMin
extern "C" void* MPSSegmentMin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentMin_Delete(void* kernel) {}
extern "C" void MPSSegmentMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { if (val < acc) acc = val; }, INFINITY);
}

// SegmentProd
extern "C" void* MPSSegmentProd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSSegmentProd_Delete(void* kernel) {}
extern "C" void MPSSegmentProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { acc *= val; }, 1.0f);
}

// UnsortedSegmentSum
extern "C" void* MPSUnsortedSegmentSum_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentSum_Delete(void* kernel) {}
extern "C" void MPSUnsortedSegmentSum_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { acc += val; }, 0.0f);
}

// UnsortedSegmentMax
extern "C" void* MPSUnsortedSegmentMax_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentMax_Delete(void* kernel) {}
extern "C" void MPSUnsortedSegmentMax_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { if (val > acc) acc = val; }, -INFINITY);
}

// UnsortedSegmentMin
extern "C" void* MPSUnsortedSegmentMin_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentMin_Delete(void* kernel) {}
extern "C" void MPSUnsortedSegmentMin_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { if (val < acc) acc = val; }, INFINITY);
}

// UnsortedSegmentProd
extern "C" void* MPSUnsortedSegmentProd_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
extern "C" void MPSUnsortedSegmentProd_Delete(void* kernel) {}
extern "C" void MPSUnsortedSegmentProd_Compute(void* kernel, TF_OpKernelContext* ctx) {
  SegmentOp(ctx, [](float& acc, float val) { acc *= val; }, 1.0f);
}
