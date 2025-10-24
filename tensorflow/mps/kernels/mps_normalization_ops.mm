// Normalization operations for MPS
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// ============================================================
// LayerNorm - Layer normalization
// ============================================================
static void* MPSLayerNorm_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSLayerNorm_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  TF_Tensor* scale_tensor;
  TF_GetInput(tf_ctx, 1, &scale_tensor, TF_NewStatus());
  TF_Tensor* offset_tensor;
  TF_GetInput(tf_ctx, 2, &offset_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* scale_data = static_cast<float*>(TF_TensorData(scale_tensor));
  float* offset_data = static_cast<float*>(TF_TensorData(offset_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  int64_t norm_size = dims[num_dims - 1];
  int64_t batch_size = num_elements / norm_size;
  
  float epsilon = 1e-5f;
  
  for (int64_t b = 0; b < batch_size; ++b) {
    float* batch_input = input_data + b * norm_size;
    float* batch_output = output_data + b * norm_size;
    
    // Compute mean
    float mean = 0.0f;
    for (int64_t i = 0; i < norm_size; ++i) {
      mean += batch_input[i];
    }
    mean /= norm_size;
    
    // Compute variance
    float variance = 0.0f;
    for (int64_t i = 0; i < norm_size; ++i) {
      float diff = batch_input[i] - mean;
      variance += diff * diff;
    }
    variance /= norm_size;
    
    // Normalize
    float inv_std = 1.0f / sqrtf(variance + epsilon);
    for (int64_t i = 0; i < norm_size; ++i) {
      batch_output[i] = (batch_input[i] - mean) * inv_std * scale_data[i] + offset_data[i];
    }
  }
  
  delete[] dims;
}

static void MPSLayerNorm_Delete(void* kernel) {}

// ============================================================
// GroupNorm - Group normalization
// ============================================================
static void* MPSGroupNorm_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSGroupNorm_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  TF_Tensor* scale_tensor;
  TF_GetInput(tf_ctx, 1, &scale_tensor, TF_NewStatus());
  TF_Tensor* offset_tensor;
  TF_GetInput(tf_ctx, 2, &offset_tensor, TF_NewStatus());
  
  // TODO: Get num_groups from attributes
  int num_groups = 32;
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* scale_data = static_cast<float*>(TF_TensorData(scale_tensor));
  float* offset_data = static_cast<float*>(TF_TensorData(offset_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Simplified group normalization (TODO: full implementation)
  int64_t batch_size = dims[0];
  int64_t num_channels = dims[num_dims - 1];
  int64_t spatial_size = num_elements / (batch_size * num_channels);
  int64_t channels_per_group = num_channels / num_groups;
  
  float epsilon = 1e-5f;
  
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int g = 0; g < num_groups; ++g) {
      int64_t group_start = g * channels_per_group;
      int64_t group_end = (g + 1) * channels_per_group;
      int64_t group_size = channels_per_group * spatial_size;
      
      // Compute mean and variance for group
      float mean = 0.0f;
      float variance = 0.0f;
      
      // Simplified: just copy input to output for now
      // TODO: Implement full group normalization
    }
  }
  
  memcpy(output_data, input_data, sizeof(float) * num_elements);
  
  delete[] dims;
}

static void MPSGroupNorm_Delete(void* kernel) {}

// ============================================================
// InstanceNorm - Instance normalization
// ============================================================
static void* MPSInstanceNorm_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSInstanceNorm_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  TF_Tensor* scale_tensor;
  TF_GetInput(tf_ctx, 1, &scale_tensor, TF_NewStatus());
  TF_Tensor* offset_tensor;
  TF_GetInput(tf_ctx, 2, &offset_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* scale_data = static_cast<float*>(TF_TensorData(scale_tensor));
  float* offset_data = static_cast<float*>(TF_TensorData(offset_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Instance norm: normalize each sample-channel independently
  // Input shape: [batch, height, width, channels]
  int64_t batch_size = dims[0];
  int64_t num_channels = dims[num_dims - 1];
  int64_t spatial_size = num_elements / (batch_size * num_channels);
  
  float epsilon = 1e-5f;
  
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t c = 0; c < num_channels; ++c) {
      // Compute mean for this instance-channel
      float mean = 0.0f;
      for (int64_t s = 0; s < spatial_size; ++s) {
        int64_t idx = b * num_channels * spatial_size + s * num_channels + c;
        mean += input_data[idx];
      }
      mean /= spatial_size;
      
      // Compute variance
      float variance = 0.0f;
      for (int64_t s = 0; s < spatial_size; ++s) {
        int64_t idx = b * num_channels * spatial_size + s * num_channels + c;
        float diff = input_data[idx] - mean;
        variance += diff * diff;
      }
      variance /= spatial_size;
      
      // Normalize
      float inv_std = 1.0f / sqrtf(variance + epsilon);
      for (int64_t s = 0; s < spatial_size; ++s) {
        int64_t idx = b * num_channels * spatial_size + s * num_channels + c;
        output_data[idx] = (input_data[idx] - mean) * inv_std * scale_data[c] + offset_data[c];
      }
    }
  }
  
  delete[] dims;
}

static void MPSInstanceNorm_Delete(void* kernel) {}

// ============================================================
// LocalResponseNormalization - LRN across channels
// ============================================================
static void* MPSLocalResponseNorm_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSLocalResponseNorm_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // TODO: Get parameters from attributes
  int depth_radius = 5;
  float bias = 1.0f;
  float alpha = 1.0f;
  float beta = 0.5f;
  
  int64_t batch_size = dims[0];
  int64_t height = dims[1];
  int64_t width = dims[2];
  int64_t num_channels = dims[3];
  
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t h = 0; h < height; ++h) {
      for (int64_t w = 0; w < width; ++w) {
        for (int64_t c = 0; c < num_channels; ++c) {
          // Compute sum of squares in local neighborhood
          float sum_squares = 0.0f;
          int64_t c_start = (c - depth_radius < 0) ? 0 : (c - depth_radius);
          int64_t c_end = (c + depth_radius >= num_channels) ? num_channels : (c + depth_radius + 1);
          
          for (int64_t cc = c_start; cc < c_end; ++cc) {
            int64_t idx = b * height * width * num_channels + h * width * num_channels + w * num_channels + cc;
            float val = input_data[idx];
            sum_squares += val * val;
          }
          
          int64_t idx = b * height * width * num_channels + h * width * num_channels + w * num_channels + c;
          float scale = powf(bias + alpha * sum_squares / (2 * depth_radius + 1), beta);
          output_data[idx] = input_data[idx] / scale;
        }
      }
    }
  }
  
  delete[] dims;
}

static void MPSLocalResponseNorm_Delete(void* kernel) {}

// ============================================================
// Registration
// ============================================================
void RegisterNormalizationOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("LayerNorm", platform_name,
                                                &MPSLayerNorm_Create,
                                                &MPSLayerNorm_Compute,
                                                &MPSLayerNorm_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSLayerNorm", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("GroupNorm", platform_name,
                                                &MPSGroupNorm_Create,
                                                &MPSGroupNorm_Compute,
                                                &MPSGroupNorm_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSGroupNorm", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("InstanceNorm", platform_name,
                                                &MPSInstanceNorm_Create,
                                                &MPSInstanceNorm_Compute,
                                                &MPSInstanceNorm_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSInstanceNorm", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("LRN", platform_name,
                                                &MPSLocalResponseNorm_Create,
                                                &MPSLocalResponseNorm_Compute,
                                                &MPSLocalResponseNorm_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSLocalResponseNorm", kb, status);
  }
}
