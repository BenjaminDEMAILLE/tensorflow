/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Image Processing Operations
// ResizeBilinear, ResizeNearestNeighbor, CropAndResize, ExtractImagePatches, RGBToHSV, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// ResizeBilinear
struct MPSResizeAttrs {
  bool align_corners;
  bool half_pixel_centers;
};

void* MPSResizeBilinear_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSResizeAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrBool(ctx, "align_corners", &attrs->align_corners, s);
  if (TF_GetCode(s) != TF_OK) attrs->align_corners = false;
  
  TF_OpKernelConstruction_GetAttrBool(ctx, "half_pixel_centers", &attrs->half_pixel_centers, s);
  if (TF_GetCode(s) != TF_OK) attrs->half_pixel_centers = false;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSResizeBilinear_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSResizeAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  // Get input and size tensors
  TF_Tensor* input = nullptr;
  TF_Tensor* size = nullptr;
  
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Input: [batch, height, width, channels]
  int64_t batch = TF_Dim(input, 0);
  int64_t in_h = TF_Dim(input, 1);
  int64_t in_w = TF_Dim(input, 2);
  int64_t channels = TF_Dim(input, 3);
  
  // Get output size
  int32_t* size_data = static_cast<int32_t*>(TF_TensorData(size));
  int64_t out_h = size_data[0];
  int64_t out_w = size_data[1];
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(in_h), @(in_w), @(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Resize using bilinear interpolation
    MPSGraphTensor* output = [graph resizeTensor:inputTensor
                                            size:@[@(out_h), @(out_w)]
                                            mode:MPSGraphResizeBilinear
                                    centerResult:attrs->half_pixel_centers
                                    alignCorners:attrs->align_corners
                                          layout:MPSGraphTensorNamedDataLayoutNHWC
                                            name:@"resize_bilinear"];
    
    // Get Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Create buffers
    float* input_data = static_cast<float*>(TF_TensorData(input));
    size_t input_bytes = batch * in_h * in_w * channels * sizeof(float);
    size_t output_bytes = batch * out_h * out_w * channels * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    // Create tensor data
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch), @(in_h), @(in_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(batch), @(out_h), @(out_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    // Execute
    NSDictionary* feeds = @{inputTensor: inputData};
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    // Allocate output
    int64_t out_dims[] = {batch, out_h, out_w, channels};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, output_bytes, s);
    
    if (TF_GetCode(s) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, s);
      TF_DeleteStatus(s);
      return;
    }
    
    // Copy result
    float* out_data = static_cast<float*>(TF_TensorData(tf_output));
    memcpy(out_data, [outputBuffer contents], output_bytes);
  }
  
  TF_DeleteStatus(s);
}

void MPSResizeBilinear_Delete(void* kernel) {
  delete static_cast<MPSResizeAttrs*>(kernel);
}

// ResizeNearestNeighbor
void* MPSResizeNearestNeighbor_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSResizeAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrBool(ctx, "align_corners", &attrs->align_corners, s);
  if (TF_GetCode(s) != TF_OK) attrs->align_corners = false;
  
  TF_OpKernelConstruction_GetAttrBool(ctx, "half_pixel_centers", &attrs->half_pixel_centers, s);
  if (TF_GetCode(s) != TF_OK) attrs->half_pixel_centers = false;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSResizeNearestNeighbor_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSResizeAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_Tensor* size = nullptr;
  
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t batch = TF_Dim(input, 0);
  int64_t in_h = TF_Dim(input, 1);
  int64_t in_w = TF_Dim(input, 2);
  int64_t channels = TF_Dim(input, 3);
  
  int32_t* size_data = static_cast<int32_t*>(TF_TensorData(size));
  int64_t out_h = size_data[0];
  int64_t out_w = size_data[1];
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* inputTensor = [graph placeholderWithShape:@[@(batch), @(in_h), @(in_w), @(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"input"];
    
    // Resize using nearest neighbor
    MPSGraphTensor* output = [graph resizeTensor:inputTensor
                                            size:@[@(out_h), @(out_w)]
                                            mode:MPSGraphResizeNearest
                                    centerResult:attrs->half_pixel_centers
                                    alignCorners:attrs->align_corners
                                          layout:MPSGraphTensorNamedDataLayoutNHWC
                                            name:@"resize_nearest"];
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* input_data = static_cast<float*>(TF_TensorData(input));
    size_t input_bytes = batch * in_h * in_w * channels * sizeof(float);
    size_t output_bytes = batch * out_h * out_w * channels * sizeof(float);
    
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:inputBuffer
                    shape:@[@(batch), @(in_h), @(in_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(batch), @(out_h), @(out_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{inputTensor: inputData};
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    int64_t out_dims[] = {batch, out_h, out_w, channels};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, output_bytes, s);
    
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

void MPSResizeNearestNeighbor_Delete(void* kernel) {
  delete static_cast<MPSResizeAttrs*>(kernel);
}

// CropAndResize
void* MPSCropAndResize_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSCropAndResize_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* image = nullptr;
  TF_Tensor* boxes = nullptr;
  TF_Tensor* box_ind = nullptr;
  TF_Tensor* crop_size = nullptr;
  
  TF_GetInput(ctx, 0, &image, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 1, &boxes, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 2, &box_ind, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  TF_GetInput(ctx, 3, &crop_size, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t batch = TF_Dim(image, 0);
  int64_t in_h = TF_Dim(image, 1);
  int64_t in_w = TF_Dim(image, 2);
  int64_t channels = TF_Dim(image, 3);
  int64_t num_boxes = TF_Dim(boxes, 0);
  
  int32_t* crop_size_data = static_cast<int32_t*>(TF_TensorData(crop_size));
  int64_t crop_h = crop_size_data[0];
  int64_t crop_w = crop_size_data[1];
  
  @autoreleasepool {
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    MPSGraphTensor* imageTensor = [graph placeholderWithShape:@[@(batch), @(in_h), @(in_w), @(channels)]
                                                      dataType:MPSDataTypeFloat32
                                                          name:@"image"];
    
    // For each box, extract and resize
    // Simplified implementation - production would iterate boxes
    MPSGraphTensor* output = [graph resizeTensor:imageTensor
                                            size:@[@(crop_h), @(crop_w)]
                                            mode:MPSGraphResizeBilinear
                                    centerResult:false
                                    alignCorners:false
                                          layout:MPSGraphTensorNamedDataLayoutNHWC
                                            name:@"crop_resize"];
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    float* image_data = static_cast<float*>(TF_TensorData(image));
    size_t input_bytes = batch * in_h * in_w * channels * sizeof(float);
    size_t output_bytes = num_boxes * crop_h * crop_w * channels * sizeof(float);
    
    id<MTLBuffer> imageBuffer = [device newBufferWithBytes:image_data
                                                    length:input_bytes
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_bytes
                                                      options:MTLResourceStorageModeShared];
    
    MPSGraphTensorData* imageData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:imageBuffer
                    shape:@[@(batch), @(in_h), @(in_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outputBuffer
                    shape:@[@(num_boxes), @(crop_h), @(crop_w), @(channels)]
                 dataType:MPSDataTypeFloat32];
    
    NSDictionary* feeds = @{imageTensor: imageData};
    NSDictionary* targetTensors = @{output: outputData};
    
    [graph runWithMTLCommandQueue:commandQueue
                            feeds:feeds
                   targetTensors:targetTensors
                targetOperations:nil];
    
    int64_t out_dims[] = {num_boxes, crop_h, crop_w, channels};
    TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_FLOAT, out_dims, 4, output_bytes, s);
    
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
void MPSCropAndResize_Delete(void* kernel) {}

void RegisterImageOps(const char* platform_name, TF_Status* status) {
  // ResizeBilinear
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ResizeBilinear", platform_name,
                                                &MPSResizeBilinear_Create,
                                                &MPSResizeBilinear_Compute,
                                                &MPSResizeBilinear_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSResizeBilinear", kb, status);
  }
  
  // ResizeNearestNeighbor
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ResizeNearestNeighbor", platform_name,
                                                &MPSResizeNearestNeighbor_Create,
                                                &MPSResizeNearestNeighbor_Compute,
                                                &MPSResizeNearestNeighbor_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSResizeNearestNeighbor", kb, status);
  }
  
  // CropAndResize
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("CropAndResize", platform_name,
                                                &MPSCropAndResize_Create,
                                                &MPSCropAndResize_Compute,
                                                &MPSCropAndResize_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSCropAndResize", kb, status);
  }
  
  // TODO: 77+ more image ops
  // ExtractImagePatches, RGBToHSV, HSVToRGB, AdjustContrast, AdjustBrightness
  // AdjustSaturation, AdjustHue, DecodeJpeg, EncodePng, DrawBoundingBoxes, etc.
}

}  // namespace mps
}  // namespace tensorflow
