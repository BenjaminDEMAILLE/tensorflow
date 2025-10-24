// Random number generation and quantization operations with Metal
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <Metal/Metal.h>
#include <random>

namespace {
id<MTLDevice> GetMetalDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

id<MTLCommandQueue> GetCommandQueue() {
  static id<MTLCommandQueue> queue = [GetMetalDevice() newCommandQueue];
  return queue;
}
}

// RandomUniform - Generate uniform random numbers
extern "C" {

typedef struct {
  int64_t seed;
  int64_t seed2;
  std::mt19937* generator;
  id<MTLCommandQueue> queue;
} MPSRandomUniformContext;

void* MPSRandomUniform_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSRandomUniformContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed", &context->seed, status);
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed2", &context->seed2, status);
  TF_DeleteStatus(status);
  
  int64_t actual_seed = context->seed != 0 ? context->seed : context->seed2;
  if (actual_seed == 0) {
    std::random_device rd;
    actual_seed = rd();
  }
  
  context->generator = new std::mt19937(actual_seed);
  
  return context;
}

void MPSRandomUniform_Delete(void* kernel) {
  auto* context = static_cast<MPSRandomUniformContext*>(kernel);
  delete context->generator;
  delete context;
}

void MPSRandomUniform_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSRandomUniformContext*>(kernel);
    
    TF_Tensor* shape_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &shape_tensor, status);
    
    int num_dims = TF_NumElements(shape_tensor);
    int32_t* shape_data = static_cast<int32_t*>(TF_TensorData(shape_tensor));
    
    int64_t* output_dims = new int64_t[num_dims];
    int64_t total_size = 1;
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = shape_data[i];
      total_size *= shape_data[i];
    }
    
    size_t bytes = total_size * sizeof(float);
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < total_size; i++) {
      output_data[i] = dist(*context->generator);
    }
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// RandomNormal - Generate normal random numbers
extern "C" {

typedef struct {
  int64_t seed;
  int64_t seed2;
  std::mt19937* generator;
  id<MTLCommandQueue> queue;
} MPSRandomNormalContext;

void* MPSRandomNormal_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSRandomNormalContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed", &context->seed, status);
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed2", &context->seed2, status);
  TF_DeleteStatus(status);
  
  int64_t actual_seed = context->seed != 0 ? context->seed : context->seed2;
  if (actual_seed == 0) {
    std::random_device rd;
    actual_seed = rd();
  }
  
  context->generator = new std::mt19937(actual_seed);
  
  return context;
}

void MPSRandomNormal_Delete(void* kernel) {
  auto* context = static_cast<MPSRandomNormalContext*>(kernel);
  delete context->generator;
  delete context;
}

void MPSRandomNormal_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSRandomNormalContext*>(kernel);
    
    TF_Tensor* shape_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &shape_tensor, status);
    
    int num_dims = TF_NumElements(shape_tensor);
    int32_t* shape_data = static_cast<int32_t*>(TF_TensorData(shape_tensor));
    
    int64_t* output_dims = new int64_t[num_dims];
    int64_t total_size = 1;
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = shape_data[i];
      total_size *= shape_data[i];
    }
    
    size_t bytes = total_size * sizeof(float);
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < total_size; i++) {
      output_data[i] = dist(*context->generator);
    }
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// Dropout - Randomly drop elements
extern "C" {

typedef struct {
  float rate;
  int64_t seed;
  std::mt19937* generator;
  id<MTLCommandQueue> queue;
} MPSDropoutContext;

void* MPSDropout_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSDropoutContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrFloat(ctx, "rate", &context->rate, status);
  TF_OpKernelConstruction_GetAttrInt64(ctx, "seed", &context->seed, status);
  TF_DeleteStatus(status);
  
  int64_t actual_seed = context->seed != 0 ? context->seed : 0;
  if (actual_seed == 0) {
    std::random_device rd;
    actual_seed = rd();
  }
  
  context->generator = new std::mt19937(actual_seed);
  
  return context;
}

void MPSDropout_Delete(void* kernel) {
  auto* context = static_cast<MPSDropoutContext*>(kernel);
  delete context->generator;
  delete context;
}

void MPSDropout_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSDropoutContext*>(kernel);
    
    TF_Tensor* input_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    
    size_t bytes = size * sizeof(float);
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float keep_prob = 1.0f - context->rate;
    float scale = 1.0f / keep_prob;
    
    for (int64_t i = 0; i < size; i++) {
      if (dist(*context->generator) < keep_prob) {
        output_data[i] = input_data[i] * scale;
      } else {
        output_data[i] = 0.0f;
      }
    }
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// QuantizeV2 - Quantize float to int8
extern "C" {

typedef struct {
  const char* mode;
  const char* round_mode;
  id<MTLCommandQueue> queue;
} MPSQuantizeV2Context;

void* MPSQuantizeV2_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSQuantizeV2Context();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  
  const char* mode = nullptr;
  size_t mode_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "mode", &mode, &mode_len, status);
  context->mode = mode;
  
  const char* round_mode = nullptr;
  size_t round_mode_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "round_mode", &round_mode, &round_mode_len, status);
  context->round_mode = round_mode;
  
  TF_DeleteStatus(status);
  return context;
}

void MPSQuantizeV2_Delete(void* kernel) {
  delete static_cast<MPSQuantizeV2Context*>(kernel);
}

void MPSQuantizeV2_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* min_range_tensor = nullptr;
    TF_Tensor* max_range_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &min_range_tensor, status);
    TF_GetInput(ctx, 2, &max_range_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
    float min_range = *static_cast<float*>(TF_TensorData(min_range_tensor));
    float max_range = *static_cast<float*>(TF_TensorData(max_range_tensor));
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    
    size_t bytes = size * sizeof(int8_t);
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_QINT8, output_dims, num_dims, bytes, status);
    int8_t* output_data = static_cast<int8_t*>(TF_TensorData(output_tensor));
    
    // Quantize: map [min_range, max_range] to [-128, 127]
    float scale = 255.0f / (max_range - min_range);
    for (int64_t i = 0; i < size; i++) {
      float val = (input_data[i] - min_range) * scale - 128.0f;
      output_data[i] = (int8_t)std::max(-128.0f, std::min(127.0f, roundf(val)));
    }
    
    // Output min/max
    int64_t scalar_dims[] = {};
    TF_Tensor* out_min = TF_AllocateOutput(ctx, 1, TF_FLOAT, scalar_dims, 0, sizeof(float), status);
    TF_Tensor* out_max = TF_AllocateOutput(ctx, 2, TF_FLOAT, scalar_dims, 0, sizeof(float), status);
    *static_cast<float*>(TF_TensorData(out_min)) = min_range;
    *static_cast<float*>(TF_TensorData(out_max)) = max_range;
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// Dequantize - Convert int8 to float
extern "C" {

typedef struct {
  const char* mode;
  id<MTLCommandQueue> queue;
} MPSDequantizeContext;

void* MPSDequantize_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSDequantizeContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  const char* mode = nullptr;
  size_t mode_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "mode", &mode, &mode_len, status);
  context->mode = mode;
  TF_DeleteStatus(status);
  
  return context;
}

void MPSDequantize_Delete(void* kernel) {
  delete static_cast<MPSDequantizeContext*>(kernel);
}

void MPSDequantize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* min_range_tensor = nullptr;
    TF_Tensor* max_range_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &min_range_tensor, status);
    TF_GetInput(ctx, 2, &max_range_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    int8_t* input_data = static_cast<int8_t*>(TF_TensorData(input_tensor));
    float min_range = *static_cast<float*>(TF_TensorData(min_range_tensor));
    float max_range = *static_cast<float*>(TF_TensorData(max_range_tensor));
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    
    size_t bytes = size * sizeof(float);
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    
    // Dequantize: map [-128, 127] to [min_range, max_range]
    float scale = (max_range - min_range) / 255.0f;
    for (int64_t i = 0; i < size; i++) {
      output_data[i] = ((float)input_data[i] + 128.0f) * scale + min_range;
    }
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// FakeQuantWithMinMaxVars - Simulate quantization for training
extern "C" {

typedef struct {
  int num_bits;
  bool narrow_range;
  id<MTLCommandQueue> queue;
} MPSFakeQuantContext;

void* MPSFakeQuant_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSFakeQuantContext();
  context->queue = GetCommandQueue();
  
  TF_Status* status = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrInt32(ctx, "num_bits", &context->num_bits, status);
  TF_OpKernelConstruction_GetAttrBool(ctx, "narrow_range", &context->narrow_range, status);
  TF_DeleteStatus(status);
  
  return context;
}

void MPSFakeQuant_Delete(void* kernel) {
  delete static_cast<MPSFakeQuantContext*>(kernel);
}

void MPSFakeQuant_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSFakeQuantContext*>(kernel);
    
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* min_tensor = nullptr;
    TF_Tensor* max_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &min_tensor, status);
    TF_GetInput(ctx, 2, &max_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
    float min_val = *static_cast<float*>(TF_TensorData(min_tensor));
    float max_val = *static_cast<float*>(TF_TensorData(max_tensor));
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    
    size_t bytes = size * sizeof(float);
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    
    int num_levels = (1 << context->num_bits) - (context->narrow_range ? 1 : 0);
    float scale = (max_val - min_val) / num_levels;
    
    for (int64_t i = 0; i < size; i++) {
      float clamped = std::max(min_val, std::min(max_val, input_data[i]));
      float quantized = roundf((clamped - min_val) / scale);
      output_data[i] = quantized * scale + min_val;
    }
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"

// ClipByValue - Clip values to range
extern "C" {

const char kClipByValueKernel[] = R"(
#include <metal_stdlib>
using namespace metal;

kernel void ClipByValueCompute(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& min_val [[buffer(2)]],
    constant float& max_val [[buffer(3)]],
    constant int& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = clamp(input[gid], min_val, max_val);
}
)";

typedef struct {
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
} MPSClipByValueContext;

void* MPSClipByValue_Create(TF_OpKernelConstruction* ctx) {
  auto* context = new MPSClipByValueContext();
  context->queue = GetCommandQueue();
  
  id<MTLDevice> device = GetMetalDevice();
  NSError* error = nil;
  
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kClipByValueKernel]
                                                options:nil
                                                  error:&error];
  
  id<MTLFunction> function = [library newFunctionWithName:@"ClipByValueCompute"];
  context->pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  
  return context;
}

void MPSClipByValue_Delete(void* kernel) {
  delete static_cast<MPSClipByValueContext*>(kernel);
}

void MPSClipByValue_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    auto* context = static_cast<MPSClipByValueContext*>(kernel);
    
    TF_Tensor* input_tensor = nullptr;
    TF_Tensor* min_tensor = nullptr;
    TF_Tensor* max_tensor = nullptr;
    TF_Status* status = TF_NewStatus();
    
    TF_GetInput(ctx, 0, &input_tensor, status);
    TF_GetInput(ctx, 1, &min_tensor, status);
    TF_GetInput(ctx, 2, &max_tensor, status);
    
    int64_t size = TF_NumElements(input_tensor);
    float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
    float min_val = *static_cast<float*>(TF_TensorData(min_tensor));
    float max_val = *static_cast<float*>(TF_TensorData(max_tensor));
    
    size_t bytes = size * sizeof(float);
    
    id<MTLDevice> device = GetMetalDevice();
    id<MTLBuffer> input_buf = [device newBufferWithBytes:input_data length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> commandBuffer = [context->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:context->pipeline];
    [encoder setBuffer:input_buf offset:0 atIndex:0];
    [encoder setBuffer:output_buf offset:0 atIndex:1];
    [encoder setBytes:&min_val length:sizeof(float) atIndex:2];
    [encoder setBytes:&max_val length:sizeof(float) atIndex:3];
    
    int sz = (int)size;
    [encoder setBytes:&sz length:sizeof(int) atIndex:4];
    
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    int num_dims = TF_NumDims(input_tensor);
    int64_t* output_dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; i++) {
      output_dims[i] = TF_Dim(input_tensor, i);
    }
    
    TF_Tensor* output_tensor = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, num_dims, bytes, status);
    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    memcpy(output_data, [output_buf contents], bytes);
    
    delete[] output_dims;
    TF_DeleteStatus(status);
  }
}

} // extern "C"
