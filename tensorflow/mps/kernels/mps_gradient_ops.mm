// Gradient/Backward operations for training on MPS
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// ============================================================
// ReluGrad - Gradient for ReLU
// ============================================================
static void* MPSReluGrad_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSReluGrad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* gradients_tensor;
  TF_GetInput(tf_ctx, 0, &gradients_tensor, TF_NewStatus());
  TF_Tensor* features_tensor;
  TF_GetInput(tf_ctx, 1, &features_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(gradients_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(gradients_tensor, i);
  }
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* gradients_data = static_cast<float*>(TF_TensorData(gradients_tensor));
  float* features_data = static_cast<float*>(TF_TensorData(features_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // ReLU gradient: gradient * (features > 0)
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = (features_data[i] > 0.0f) ? gradients_data[i] : 0.0f;
  }
  
  delete[] dims;
}

static void MPSReluGrad_Delete(void* kernel) {}

// ============================================================
// Relu6Grad - Gradient for ReLU6
// ============================================================
static void* MPSRelu6Grad_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSRelu6Grad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* gradients_tensor;
  TF_GetInput(tf_ctx, 0, &gradients_tensor, TF_NewStatus());
  TF_Tensor* features_tensor;
  TF_GetInput(tf_ctx, 1, &features_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(gradients_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(gradients_tensor, i);
  }
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* gradients_data = static_cast<float*>(TF_TensorData(gradients_tensor));
  float* features_data = static_cast<float*>(TF_TensorData(features_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // ReLU6 gradient: gradient * (0 < features < 6)
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = (features_data[i] > 0.0f && features_data[i] < 6.0f) ? gradients_data[i] : 0.0f;
  }
  
  delete[] dims;
}

static void MPSRelu6Grad_Delete(void* kernel) {}

// ============================================================
// SoftmaxCrossEntropyWithLogits - Combined softmax + cross entropy
// ============================================================
static void* MPSSoftmaxCrossEntropyWithLogits_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSSoftmaxCrossEntropyWithLogits_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* features_tensor;
  TF_GetInput(tf_ctx, 0, &features_tensor, TF_NewStatus());
  TF_Tensor* labels_tensor;
  TF_GetInput(tf_ctx, 1, &labels_tensor, TF_NewStatus());
  
  int batch_size = TF_Dim(features_tensor, 0);
  int num_classes = TF_Dim(features_tensor, 1);
  
  // Output 0: loss (batch_size,)
  int64_t loss_dims[] = {static_cast<int64_t>(batch_size)};
  TF_Tensor* loss_output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, loss_dims, 1,
                                              sizeof(float) * batch_size, TF_NewStatus());
  
  // Output 1: backprop (batch_size, num_classes)
  int64_t backprop_dims[] = {static_cast<int64_t>(batch_size), static_cast<int64_t>(num_classes)};
  TF_Tensor* backprop_output = TF_AllocateOutput(tf_ctx, 1, TF_FLOAT, backprop_dims, 2,
                                                  sizeof(float) * batch_size * num_classes, TF_NewStatus());
  
  float* features_data = static_cast<float*>(TF_TensorData(features_tensor));
  float* labels_data = static_cast<float*>(TF_TensorData(labels_tensor));
  float* loss_data = static_cast<float*>(TF_TensorData(loss_output));
  float* backprop_data = static_cast<float*>(TF_TensorData(backprop_output));
  
  for (int b = 0; b < batch_size; ++b) {
    float* batch_features = features_data + b * num_classes;
    float* batch_labels = labels_data + b * num_classes;
    
    // Compute softmax
    float max_val = batch_features[0];
    for (int c = 1; c < num_classes; ++c) {
      if (batch_features[c] > max_val) max_val = batch_features[c];
    }
    
    float sum_exp = 0.0f;
    float* softmax = new float[num_classes];
    for (int c = 0; c < num_classes; ++c) {
      softmax[c] = expf(batch_features[c] - max_val);
      sum_exp += softmax[c];
    }
    for (int c = 0; c < num_classes; ++c) {
      softmax[c] /= sum_exp;
    }
    
    // Compute cross entropy loss
    float loss = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      if (batch_labels[c] > 0.0f) {
        loss -= batch_labels[c] * logf(softmax[c] + 1e-10f);
      }
    }
    loss_data[b] = loss;
    
    // Compute gradient: softmax - labels
    for (int c = 0; c < num_classes; ++c) {
      backprop_data[b * num_classes + c] = softmax[c] - batch_labels[c];
    }
    
    delete[] softmax;
  }
}

static void MPSSoftmaxCrossEntropyWithLogits_Delete(void* kernel) {}

// ============================================================
// Conv2DBackpropInput - Gradient for Conv2D w.r.t. input
// ============================================================
static void* MPSConv2DBackpropInput_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSConv2DBackpropInput_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_sizes_tensor;
  TF_GetInput(tf_ctx, 0, &input_sizes_tensor, TF_NewStatus());
  TF_Tensor* filter_tensor;
  TF_GetInput(tf_ctx, 1, &filter_tensor, TF_NewStatus());
  TF_Tensor* out_backprop_tensor;
  TF_GetInput(tf_ctx, 2, &out_backprop_tensor, TF_NewStatus());
  
  int64_t* input_sizes = static_cast<int64_t*>(TF_TensorData(input_sizes_tensor));
  int num_dims = TF_Dim(input_sizes_tensor, 0);
  
  size_t output_size = sizeof(float);
  for (int i = 0; i < num_dims; ++i) {
    output_size *= input_sizes[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, input_sizes, num_dims, output_size, TF_NewStatus());
  
  // TODO: Implement Conv2D backprop using MPSGraph or Metal
  // For now, return zeros
  memset(TF_TensorData(output), 0, output_size);
}

static void MPSConv2DBackpropInput_Delete(void* kernel) {}

// ============================================================
// Conv2DBackpropFilter - Gradient for Conv2D w.r.t. filter
// ============================================================
static void* MPSConv2DBackpropFilter_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSConv2DBackpropFilter_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  TF_Tensor* filter_sizes_tensor;
  TF_GetInput(tf_ctx, 1, &filter_sizes_tensor, TF_NewStatus());
  TF_Tensor* out_backprop_tensor;
  TF_GetInput(tf_ctx, 2, &out_backprop_tensor, TF_NewStatus());
  
  int64_t* filter_sizes = static_cast<int64_t*>(TF_TensorData(filter_sizes_tensor));
  int num_dims = TF_Dim(filter_sizes_tensor, 0);
  
  size_t output_size = sizeof(float);
  for (int i = 0; i < num_dims; ++i) {
    output_size *= filter_sizes[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, filter_sizes, num_dims, output_size, TF_NewStatus());
  
  // TODO: Implement Conv2D filter gradient using MPSGraph
  // For now, return zeros
  memset(TF_TensorData(output), 0, output_size);
}

static void MPSConv2DBackpropFilter_Delete(void* kernel) {}

// ============================================================
// MaxPoolGrad - Gradient for MaxPool
// ============================================================
static void* MPSMaxPoolGrad_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSMaxPoolGrad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* orig_input_tensor;
  TF_GetInput(tf_ctx, 0, &orig_input_tensor, TF_NewStatus());
  TF_Tensor* orig_output_tensor;
  TF_GetInput(tf_ctx, 1, &orig_output_tensor, TF_NewStatus());
  TF_Tensor* grad_tensor;
  TF_GetInput(tf_ctx, 2, &grad_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(orig_input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(orig_input_tensor, i);
  }
  
  size_t output_size = sizeof(float);
  for (int i = 0; i < num_dims; ++i) {
    output_size *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims, output_size, TF_NewStatus());
  
  // TODO: Implement MaxPool gradient - route gradients to max locations
  // For now, return zeros
  memset(TF_TensorData(output), 0, output_size);
  
  delete[] dims;
}

static void MPSMaxPoolGrad_Delete(void* kernel) {}

// ============================================================
// BiasAddGrad - Gradient for BiasAdd
// ============================================================
static void* MPSBiasAddGrad_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSBiasAddGrad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* out_backprop_tensor;
  TF_GetInput(tf_ctx, 0, &out_backprop_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(out_backprop_tensor);
  int64_t num_channels = TF_Dim(out_backprop_tensor, num_dims - 1);
  
  int64_t output_dims[] = {num_channels};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 1,
                                         sizeof(float) * num_channels, TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(out_backprop_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Sum gradients across all dimensions except the last one
  size_t total_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    total_elements *= TF_Dim(out_backprop_tensor, i);
  }
  
  size_t batch_size = total_elements / num_channels;
  
  for (int64_t c = 0; c < num_channels; ++c) {
    float sum = 0.0f;
    for (size_t b = 0; b < batch_size; ++b) {
      sum += input_data[b * num_channels + c];
    }
    output_data[c] = sum;
  }
}

static void MPSBiasAddGrad_Delete(void* kernel) {}

// ============================================================
// BatchNormGrad - Gradient for batch normalization
// ============================================================
static void* MPSBatchNormGrad_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSBatchNormGrad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  // TODO: Implement full batch norm gradient computation
  // Outputs: dx, dscale, doffset
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t output_size = sizeof(float);
  for (int i = 0; i < num_dims; ++i) {
    output_size *= dims[i];
  }
  
  // Output 0: dx (same shape as input)
  TF_Tensor* dx = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims, output_size, TF_NewStatus());
  memset(TF_TensorData(dx), 0, output_size);
  
  // Output 1: dscale (num_channels,)
  int64_t num_channels = dims[num_dims - 1];
  int64_t param_dims[] = {num_channels};
  TF_Tensor* dscale = TF_AllocateOutput(tf_ctx, 1, TF_FLOAT, param_dims, 1, sizeof(float) * num_channels, TF_NewStatus());
  memset(TF_TensorData(dscale), 0, sizeof(float) * num_channels);
  
  // Output 2: doffset (num_channels,)
  TF_Tensor* doffset = TF_AllocateOutput(tf_ctx, 2, TF_FLOAT, param_dims, 1, sizeof(float) * num_channels, TF_NewStatus());
  memset(TF_TensorData(doffset), 0, sizeof(float) * num_channels);
  
  delete[] dims;
}

static void MPSBatchNormGrad_Delete(void* kernel) {}

// ============================================================
// Registration
// ============================================================
void RegisterGradientOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ReluGrad", platform_name,
                                                &MPSReluGrad_Create,
                                                &MPSReluGrad_Compute,
                                                &MPSReluGrad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSReluGrad", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Relu6Grad", platform_name,
                                                &MPSRelu6Grad_Create,
                                                &MPSRelu6Grad_Compute,
                                                &MPSRelu6Grad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSRelu6Grad", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("SoftmaxCrossEntropyWithLogits", platform_name,
                                                &MPSSoftmaxCrossEntropyWithLogits_Create,
                                                &MPSSoftmaxCrossEntropyWithLogits_Compute,
                                                &MPSSoftmaxCrossEntropyWithLogits_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSoftmaxCrossEntropyWithLogits", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Conv2DBackpropInput", platform_name,
                                                &MPSConv2DBackpropInput_Create,
                                                &MPSConv2DBackpropInput_Compute,
                                                &MPSConv2DBackpropInput_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSConv2DBackpropInput", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Conv2DBackpropFilter", platform_name,
                                                &MPSConv2DBackpropFilter_Create,
                                                &MPSConv2DBackpropFilter_Compute,
                                                &MPSConv2DBackpropFilter_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSConv2DBackpropFilter", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MaxPoolGrad", platform_name,
                                                &MPSMaxPoolGrad_Create,
                                                &MPSMaxPoolGrad_Compute,
                                                &MPSMaxPoolGrad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMaxPoolGrad", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("BiasAddGrad", platform_name,
                                                &MPSBiasAddGrad_Create,
                                                &MPSBiasAddGrad_Compute,
                                                &MPSBiasAddGrad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSBiasAddGrad", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("FusedBatchNormGrad", platform_name,
                                                &MPSBatchNormGrad_Create,
                                                &MPSBatchNormGrad_Compute,
                                                &MPSBatchNormGrad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSBatchNormGrad", kb, status);
  }
}
