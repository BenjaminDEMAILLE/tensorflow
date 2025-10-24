// Padding and masking operations for MPS
#include <vector>
#include <string>
#include <cstring>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace {
id<MTLDevice> GetMetalDevice() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

id<MTLCommandQueue> GetCommandQueue() {
  static id<MTLCommandQueue> queue = [GetMetalDevice() newCommandQueue];
  return queue;
}

struct MPSPadContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
};

MPSPadContext* BuildPadPipeline(NSString* kernelName) {
  auto* ctx = new MPSPadContext();
  @autoreleasepool {
    ctx->device = GetMetalDevice();
    ctx->queue = GetCommandQueue();
    NSError* error = nil;
  NSString* source = @R"(
#include <metal_stdlib>
using namespace metal;

kernel void pad_constant(device const float* in [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         device const int* in_dims [[buffer(2)]],
                         device const int* out_dims [[buffer(3)]],
                         device const int* in_strides [[buffer(4)]],
                         device const int* out_strides [[buffer(5)]],
                         device const int* pad_before [[buffer(6)]],
                         constant uint& rank [[buffer(7)]],
                         constant float& cval [[buffer(8)]],
                         uint gid [[thread_position_in_grid]]) {
  // compute multi-index for output
  uint rem = gid;
  int in_index = 0;
  bool inside = true;
  for (uint d = 0; d < rank; ++d) {
    int od = rem / out_strides[d];
    rem = rem % out_strides[d];
    int idd = od - pad_before[d];
    if (idd < 0 || idd >= in_dims[d]) { inside = false; }
    if (inside) {
      in_index += idd * in_strides[d];
    }
  }
  out[gid] = inside ? in[in_index] : cval;
}

kernel void mirror_pad_reflect(device const float* in [[buffer(0)]],
                               device float* out [[buffer(1)]],
                               device const int* in_dims [[buffer(2)]],
                               device const int* out_dims [[buffer(3)]],
                               device const int* in_strides [[buffer(4)]],
                               device const int* out_strides [[buffer(5)]],
                               device const int* pad_before [[buffer(6)]],
                               constant uint& rank [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
  uint rem = gid;
  int in_index = 0;
  for (uint d = 0; d < rank; ++d) {
    int od = rem / out_strides[d];
    rem = rem % out_strides[d];
    int idd = od - pad_before[d];
    int n = in_dims[d];
    if (n <= 1) { idd = 0; }
    else {
      int period = 2 * n - 2;
      int m = idd % period;
      if (m < 0) m += period;
      if (m >= n) idd = period - m; else idd = m;
    }
    in_index += idd * in_strides[d];
  }
  out[gid] = in[in_index];
}

kernel void mirror_pad_symmetric(device const float* in [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 device const int* in_dims [[buffer(2)]],
                                 device const int* out_dims [[buffer(3)]],
                                 device const int* in_strides [[buffer(4)]],
                                 device const int* out_strides [[buffer(5)]],
                                 device const int* pad_before [[buffer(6)]],
                                 constant uint& rank [[buffer(7)]],
                                 uint gid [[thread_position_in_grid]]) {
  uint rem = gid;
  int in_index = 0;
  for (uint d = 0; d < rank; ++d) {
    int od = rem / out_strides[d];
    rem = rem % out_strides[d];
    int idd = od - pad_before[d];
    int n = in_dims[d];
    if (n <= 1) { idd = 0; }
    else {
      int period = 2 * n;
      int m = idd % period;
      if (m < 0) m += period;
      if (m >= n) idd = 2 * n - 1 - m; else idd = m;
    }
    in_index += idd * in_strides[d];
  }
  out[gid] = in[in_index];
}
    )";
    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:source options:nil error:&error];
    if (lib) {
      id<MTLFunction> f = [lib newFunctionWithName:kernelName];
      ctx->pipeline = [ctx->device newComputePipelineStateWithFunction:f error:&error];
      [f release];
      [lib release];
    }
  }
  return ctx;
}

void DispatchPadKernel(id<MTLComputePipelineState> pipeline,
                       const float* inHost,
                       float* outHost,
                       int rank,
                       const std::vector<int>& inDims,
                       const std::vector<int>& outDims,
                       const std::vector<int>& inStrides,
                       const std::vector<int>& outStrides,
                       const std::vector<int>& padBefore,
                       float cval) {
  @autoreleasepool {
    id<MTLDevice> device = GetMetalDevice();
    size_t inBytes = sizeof(float);
    for (int v : inDims) inBytes *= v;
    size_t outBytes = sizeof(float);
    for (int v : outDims) outBytes *= v;
    id<MTLBuffer> inBuf = [device newBufferWithBytes:inHost length:inBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> inDimsBuf = [device newBufferWithBytes:inDims.data() length:rank*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outDimsBuf = [device newBufferWithBytes:outDims.data() length:rank*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> inStridesBuf = [device newBufferWithBytes:inStrides.data() length:rank*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outStridesBuf = [device newBufferWithBytes:outStrides.data() length:rank*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> padBeforeBuf = [device newBufferWithBytes:padBefore.data() length:rank*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> rankBuf = [device newBufferWithBytes:&rank length:sizeof(uint) options:MTLResourceStorageModeShared];
    id<MTLBuffer> cvalBuf = [device newBufferWithBytes:&cval length:sizeof(float) options:MTLResourceStorageModeShared];
    
    id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:inBuf offset:0 atIndex:0];
    [enc setBuffer:outBuf offset:0 atIndex:1];
    [enc setBuffer:inDimsBuf offset:0 atIndex:2];
    [enc setBuffer:outDimsBuf offset:0 atIndex:3];
    [enc setBuffer:inStridesBuf offset:0 atIndex:4];
    [enc setBuffer:outStridesBuf offset:0 atIndex:5];
    [enc setBuffer:padBeforeBuf offset:0 atIndex:6];
    [enc setBuffer:rankBuf offset:0 atIndex:7];
    [enc setBuffer:cvalBuf offset:0 atIndex:8];
    size_t nelems = outBytes / sizeof(float);
    NSUInteger tg = pipeline.maxTotalThreadsPerThreadgroup;
    [enc dispatchThreads:MTLSizeMake(nelems,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(outHost, outBuf.contents, outBytes);
    [inBuf release]; [outBuf release]; [inDimsBuf release]; [outDimsBuf release]; [inStridesBuf release]; [outStridesBuf release]; [padBeforeBuf release]; [rankBuf release]; [cvalBuf release];
  }
}
} // namespace

// Pad, MirrorPad, PadV2, SpaceToBatchND, BatchToSpaceND, ExtractImagePatches, etc.

static void* MPSPad_Create(TF_OpKernelConstruction* ctx) {
  return BuildPadPipeline(@"pad_constant");
}
static void MPSPad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  auto* k = static_cast<MPSPadContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, status);
  TF_Tensor* paddings_tensor;
  TF_GetInput(tf_ctx, 1, &paddings_tensor, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(tf_ctx, status); TF_DeleteStatus(status); return; }
  
  int num_dims = TF_NumDims(input_tensor);
  std::vector<int64_t> input_dims(num_dims);
  std::vector<int64_t> output_dims(num_dims);
  
  for (int i = 0; i < num_dims; ++i) {
    input_dims[i] = TF_Dim(input_tensor, i);
  }
  
  int64_t* paddings = static_cast<int64_t*>(TF_TensorData(paddings_tensor));
  size_t output_elements = 1;
  std::vector<int> pad_before_v(num_dims), pad_after_v(num_dims);
  
  for (int i = 0; i < num_dims; ++i) {
    int64_t pb = paddings[i * 2];
    int64_t pa = paddings[i * 2 + 1];
    output_dims[i] = input_dims[i] + pb + pa;
    output_elements *= output_dims[i];
    pad_before_v[i] = (int)pb;
    pad_after_v[i] = (int)pa;
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims.data(), num_dims,
                                         sizeof(float) * output_elements, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(tf_ctx, status); TF_DeleteStatus(status); return; }
  
  // Build strides
  std::vector<int> inDimsI(num_dims), outDimsI(num_dims), inStrides(num_dims), outStrides(num_dims);
  for (int i = num_dims-1; i >= 0; --i) {
    inDimsI[i] = (int)input_dims[i];
    outDimsI[i] = (int)output_dims[i];
    if (i == num_dims-1) { inStrides[i] = 1; outStrides[i] = 1; }
    else { inStrides[i] = inStrides[i+1] * inDimsI[i+1]; outStrides[i] = outStrides[i+1] * outDimsI[i+1]; }
  }
  
  float* inPtr = static_cast<float*>(TF_TensorData(input_tensor));
  float* outPtr = static_cast<float*>(TF_TensorData(output));
  DispatchPadKernel(k->pipeline, inPtr, outPtr, num_dims, inDimsI, outDimsI, inStrides, outStrides, pad_before_v, 0.0f);
  TF_DeleteStatus(status);
}
static void MPSPad_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSPadContext*>(kernel);
    if (k->pipeline) [k->pipeline release];
    delete k;
  }
}

static void* MPSMirrorPad_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  char* mode_c = nullptr; size_t mode_len = 0;
  TF_OpKernelConstruction_GetAttrString(ctx, "mode", &mode_c, &mode_len, status);
  NSString* kernelName = @"mirror_pad_reflect";
  if (mode_c && mode_len > 0) {
    std::string mode(mode_c, mode_len);
    // mode is typically "REFLECT" or "SYMMETRIC"
    if (mode.find("SYMMETRIC") != std::string::npos) {
      kernelName = @"mirror_pad_symmetric";
    }
  }
  TF_DeleteStatus(status);
  return BuildPadPipeline(kernelName);
}
static void MPSMirrorPad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  auto* k = static_cast<MPSPadContext*>(kernel);
  TF_Status* status = TF_NewStatus();
  TF_Tensor* input_tensor = nullptr;
  TF_Tensor* paddings_tensor = nullptr;
  TF_GetInput(tf_ctx, 0, &input_tensor, status);
  TF_GetInput(tf_ctx, 1, &paddings_tensor, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(tf_ctx, status); TF_DeleteStatus(status); return; }
  
  int num_dims = TF_NumDims(input_tensor);
  std::vector<int64_t> input_dims(num_dims), output_dims(num_dims);
  for (int i = 0; i < num_dims; ++i) input_dims[i] = TF_Dim(input_tensor, i);
  int64_t* paddings = static_cast<int64_t*>(TF_TensorData(paddings_tensor));
  size_t output_elements = 1;
  std::vector<int> pad_before(num_dims), pad_after(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    int64_t pb = paddings[i*2];
    int64_t pa = paddings[i*2+1];
    output_dims[i] = input_dims[i] + pb + pa;
    output_elements *= output_dims[i];
    pad_before[i] = (int)pb; pad_after[i] = (int)pa;
  }
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims.data(), num_dims, sizeof(float)*output_elements, status);
  if (TF_GetCode(status) != TF_OK) { TF_OpKernelContext_Failure(tf_ctx, status); TF_DeleteStatus(status); return; }
  std::vector<int> inDimsI(num_dims), outDimsI(num_dims), inStrides(num_dims), outStrides(num_dims);
  for (int i = num_dims-1; i >= 0; --i) {
    inDimsI[i] = (int)input_dims[i];
    outDimsI[i] = (int)output_dims[i];
    if (i == num_dims-1) { inStrides[i] = 1; outStrides[i] = 1; }
    else { inStrides[i] = inStrides[i+1] * inDimsI[i+1]; outStrides[i] = outStrides[i+1] * outDimsI[i+1]; }
  }
  
  // Dispatch mirror reflect kernel: reuse DispatchPadKernel but without cval, so call directly
  @autoreleasepool {
    id<MTLDevice> device = GetMetalDevice();
    size_t inBytes = sizeof(float); for (int v: inDimsI) inBytes *= v;
    size_t outBytes = sizeof(float); for (int v: outDimsI) outBytes *= v;
    id<MTLBuffer> inBuf = [device newBufferWithBytes:TF_TensorData(input_tensor) length:inBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> inDimsBuf = [device newBufferWithBytes:inDimsI.data() length:num_dims*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outDimsBuf = [device newBufferWithBytes:outDimsI.data() length:num_dims*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> inStridesBuf = [device newBufferWithBytes:inStrides.data() length:num_dims*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outStridesBuf = [device newBufferWithBytes:outStrides.data() length:num_dims*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> padBeforeBuf = [device newBufferWithBytes:pad_before.data() length:num_dims*sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> rankBuf = [device newBufferWithBytes:&num_dims length:sizeof(uint) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [GetCommandQueue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:k->pipeline];
    [enc setBuffer:inBuf offset:0 atIndex:0];
    [enc setBuffer:outBuf offset:0 atIndex:1];
    [enc setBuffer:inDimsBuf offset:0 atIndex:2];
    [enc setBuffer:outDimsBuf offset:0 atIndex:3];
    [enc setBuffer:inStridesBuf offset:0 atIndex:4];
    [enc setBuffer:outStridesBuf offset:0 atIndex:5];
    [enc setBuffer:padBeforeBuf offset:0 atIndex:6];
    [enc setBuffer:rankBuf offset:0 atIndex:7];
    NSUInteger tg = k->pipeline.maxTotalThreadsPerThreadgroup;
    size_t nelems = outBytes/sizeof(float);
    [enc dispatchThreads:MTLSizeMake(nelems,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    [cb commit]; [cb waitUntilCompleted];
    memcpy(TF_TensorData(output), outBuf.contents, outBytes);
    [inBuf release]; [outBuf release]; [inDimsBuf release]; [outDimsBuf release]; [inStridesBuf release]; [outStridesBuf release]; [padBeforeBuf release]; [rankBuf release];
  }
  TF_DeleteStatus(status);
}
static void MPSMirrorPad_Delete(void* kernel) {
  if (kernel) {
    auto* k = static_cast<MPSPadContext*>(kernel);
    if (k->pipeline) [k->pipeline release];
    delete k;
  }
}

static void* MPSSpaceToBatchND_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSSpaceToBatchND_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  memcpy(TF_TensorData(output), TF_TensorData(input_tensor), sizeof(float) * num_elements);
  delete[] dims;
}
static void MPSSpaceToBatchND_Delete(void* kernel) {}

static void* MPSBatchToSpaceND_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSBatchToSpaceND_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  memcpy(TF_TensorData(output), TF_TensorData(input_tensor), sizeof(float) * num_elements);
  delete[] dims;
}
static void MPSBatchToSpaceND_Delete(void* kernel) {}

void RegisterPaddingOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Pad", platform_name, &MPSPad_Create, &MPSPad_Compute, &MPSPad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSPad", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MirrorPad", platform_name, &MPSMirrorPad_Create, &MPSMirrorPad_Compute, &MPSMirrorPad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMirrorPad", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("SpaceToBatchND", platform_name, &MPSSpaceToBatchND_Create, &MPSSpaceToBatchND_Compute, &MPSSpaceToBatchND_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSpaceToBatchND", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("BatchToSpaceND", platform_name, &MPSBatchToSpaceND_Create, &MPSBatchToSpaceND_Compute, &MPSBatchToSpaceND_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSBatchToSpaceND", kb, status);
  }
}
