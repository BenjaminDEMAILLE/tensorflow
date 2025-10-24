// Loss functions for MPS
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#import <Foundation/Foundation.h>
#include <cmath>

// ============================================================
// SparseSoftmaxCrossEntropyWithLogits - Sparse cross entropy
// ============================================================
static void* MPSSparseSoftmaxCrossEntropy_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSSparseSoftmaxCrossEntropy_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
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
  int64_t* labels_data = static_cast<int64_t*>(TF_TensorData(labels_tensor));
  float* loss_data = static_cast<float*>(TF_TensorData(loss_output));
  float* backprop_data = static_cast<float*>(TF_TensorData(backprop_output));
  
  for (int b = 0; b < batch_size; ++b) {
    float* batch_features = features_data + b * num_classes;
    int64_t label = labels_data[b];
    
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
    loss_data[b] = -logf(softmax[label] + 1e-10f);
    
    // Compute gradient: softmax - one_hot(label)
    for (int c = 0; c < num_classes; ++c) {
      backprop_data[b * num_classes + c] = softmax[c];
      if (c == label) {
        backprop_data[b * num_classes + c] -= 1.0f;
      }
    }
    
    delete[] softmax;
  }
}

static void MPSSparseSoftmaxCrossEntropy_Delete(void* kernel) {}

// ============================================================
// SigmoidCrossEntropyWithLogits - Binary cross entropy
// ============================================================
static void* MPSSigmoidCrossEntropy_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSSigmoidCrossEntropy_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* logits_tensor;
  TF_GetInput(tf_ctx, 0, &logits_tensor, TF_NewStatus());
  TF_Tensor* labels_tensor;
  TF_GetInput(tf_ctx, 1, &labels_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(logits_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(logits_tensor, i);
  }
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* logits_data = static_cast<float*>(TF_TensorData(logits_tensor));
  float* labels_data = static_cast<float*>(TF_TensorData(labels_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Binary cross entropy: max(x, 0) - x * z + log(1 + exp(-abs(x)))
  // where x = logits, z = labels
  for (size_t i = 0; i < num_elements; ++i) {
    float x = logits_data[i];
    float z = labels_data[i];
    float max_val = (x > 0.0f) ? x : 0.0f;
    output_data[i] = max_val - x * z + logf(1.0f + expf(-fabsf(x)));
  }
  
  delete[] dims;
}

static void MPSSigmoidCrossEntropy_Delete(void* kernel) {}

// ============================================================
// L2Loss - L2 loss (sum of squares / 2)
// ============================================================
static void* MPSL2Loss_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSL2Loss_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= TF_Dim(input_tensor, i);
  }
  
  // Output is scalar
  int64_t output_dims[] = {};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 0,
                                         sizeof(float), TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // L2 loss: sum(x^2) / 2
  float sum_squares = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    sum_squares += input_data[i] * input_data[i];
  }
  *output_data = sum_squares / 2.0f;
}

static void MPSL2Loss_Delete(void* kernel) {}

// ============================================================
// MeanSquaredError - MSE loss
// ============================================================
static void* MPSMeanSquaredError_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSMeanSquaredError_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* predictions_tensor;
  TF_GetInput(tf_ctx, 0, &predictions_tensor, TF_NewStatus());
  TF_Tensor* labels_tensor;
  TF_GetInput(tf_ctx, 1, &labels_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(predictions_tensor);
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= TF_Dim(predictions_tensor, i);
  }
  
  // Output is scalar
  int64_t output_dims[] = {};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 0,
                                         sizeof(float), TF_NewStatus());
  
  float* predictions_data = static_cast<float*>(TF_TensorData(predictions_tensor));
  float* labels_data = static_cast<float*>(TF_TensorData(labels_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // MSE: mean((predictions - labels)^2)
  float sum_squares = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    float diff = predictions_data[i] - labels_data[i];
    sum_squares += diff * diff;
  }
  *output_data = sum_squares / num_elements;
}

static void MPSMeanSquaredError_Delete(void* kernel) {}

// ============================================================
// MeanAbsoluteError - MAE loss
// ============================================================
static void* MPSMeanAbsoluteError_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSMeanAbsoluteError_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* predictions_tensor;
  TF_GetInput(tf_ctx, 0, &predictions_tensor, TF_NewStatus());
  TF_Tensor* labels_tensor;
  TF_GetInput(tf_ctx, 1, &labels_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(predictions_tensor);
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= TF_Dim(predictions_tensor, i);
  }
  
  // Output is scalar
  int64_t output_dims[] = {};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 0,
                                         sizeof(float), TF_NewStatus());
  
  float* predictions_data = static_cast<float*>(TF_TensorData(predictions_tensor));
  float* labels_data = static_cast<float*>(TF_TensorData(labels_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // MAE: mean(abs(predictions - labels))
  float sum_abs = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    sum_abs += fabsf(predictions_data[i] - labels_data[i]);
  }
  *output_data = sum_abs / num_elements;
}

static void MPSMeanAbsoluteError_Delete(void* kernel) {}

// ============================================================
// HuberLoss - Huber loss (smooth L1)
// ============================================================
static void* MPSHuberLoss_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSHuberLoss_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* predictions_tensor;
  TF_GetInput(tf_ctx, 0, &predictions_tensor, TF_NewStatus());
  TF_Tensor* labels_tensor;
  TF_GetInput(tf_ctx, 1, &labels_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(predictions_tensor);
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= TF_Dim(predictions_tensor, i);
  }
  
  // Output is scalar
  int64_t output_dims[] = {};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 0,
                                         sizeof(float), TF_NewStatus());
  
  float* predictions_data = static_cast<float*>(TF_TensorData(predictions_tensor));
  float* labels_data = static_cast<float*>(TF_TensorData(labels_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Huber loss: 0.5 * x^2 if |x| <= delta, delta * (|x| - 0.5 * delta) otherwise
  float delta = 1.0f;  // TODO: Get from attributes
  float sum_loss = 0.0f;
  
  for (size_t i = 0; i < num_elements; ++i) {
    float diff = predictions_data[i] - labels_data[i];
    float abs_diff = fabsf(diff);
    
    if (abs_diff <= delta) {
      sum_loss += 0.5f * diff * diff;
    } else {
      sum_loss += delta * (abs_diff - 0.5f * delta);
    }
  }
  
  *output_data = sum_loss / num_elements;
}

static void MPSHuberLoss_Delete(void* kernel) {}

// ============================================================
// Registration
// ============================================================
void RegisterLossOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("SparseSoftmaxCrossEntropyWithLogits", platform_name,
                                                &MPSSparseSoftmaxCrossEntropy_Create,
                                                &MPSSparseSoftmaxCrossEntropy_Compute,
                                                &MPSSparseSoftmaxCrossEntropy_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_KernelBuilder_TypeConstraint(kb, "Tlabels", TF_INT64, status);
    TF_RegisterKernelBuilder("MPSSparseSoftmaxCrossEntropy", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("SigmoidCrossEntropyWithLogits", platform_name,
                                                &MPSSigmoidCrossEntropy_Create,
                                                &MPSSigmoidCrossEntropy_Compute,
                                                &MPSSigmoidCrossEntropy_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSigmoidCrossEntropy", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("L2Loss", platform_name,
                                                &MPSL2Loss_Create,
                                                &MPSL2Loss_Compute,
                                                &MPSL2Loss_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSL2Loss", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MeanSquaredError", platform_name,
                                                &MPSMeanSquaredError_Create,
                                                &MPSMeanSquaredError_Compute,
                                                &MPSMeanSquaredError_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMeanSquaredError", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("MeanAbsoluteError", platform_name,
                                                &MPSMeanAbsoluteError_Create,
                                                &MPSMeanAbsoluteError_Compute,
                                                &MPSMeanAbsoluteError_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMeanAbsoluteError", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("HuberLoss", platform_name,
                                                &MPSHuberLoss_Create,
                                                &MPSHuberLoss_Compute,
                                                &MPSHuberLoss_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSHuberLoss", kb, status);
  }
}
