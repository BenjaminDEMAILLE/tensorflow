// Optimizer operations for MPS (SGD, Adam, RMSprop, etc.)
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#import <Foundation/Foundation.h>
#include <cmath>

// ApplyGradientDescent, ApplyAdam, ApplyRMSprop, ApplyMomentum, etc.

static void* MPSApplyGradientDescent_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSApplyGradientDescent_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* var_tensor;
  TF_GetInput(tf_ctx, 0, &var_tensor, TF_NewStatus());
  TF_Tensor* alpha_tensor;
  TF_GetInput(tf_ctx, 1, &alpha_tensor, TF_NewStatus());
  TF_Tensor* delta_tensor;
  TF_GetInput(tf_ctx, 2, &delta_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(var_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(var_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* var_data = static_cast<float*>(TF_TensorData(var_tensor));
  float alpha = *static_cast<float*>(TF_TensorData(alpha_tensor));
  float* delta_data = static_cast<float*>(TF_TensorData(delta_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // SGD: var = var - alpha * delta
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = var_data[i] - alpha * delta_data[i];
  }
  
  delete[] dims;
}
static void MPSApplyGradientDescent_Delete(void* kernel) {}

static void* MPSApplyMomentum_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSApplyMomentum_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* var_tensor;
  TF_GetInput(tf_ctx, 0, &var_tensor, TF_NewStatus());
  TF_Tensor* accum_tensor;
  TF_GetInput(tf_ctx, 1, &accum_tensor, TF_NewStatus());
  TF_Tensor* lr_tensor;
  TF_GetInput(tf_ctx, 2, &lr_tensor, TF_NewStatus());
  TF_Tensor* grad_tensor;
  TF_GetInput(tf_ctx, 3, &grad_tensor, TF_NewStatus());
  TF_Tensor* momentum_tensor;
  TF_GetInput(tf_ctx, 4, &momentum_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(var_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(var_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* var_data = static_cast<float*>(TF_TensorData(var_tensor));
  float* accum_data = static_cast<float*>(TF_TensorData(accum_tensor));
  float lr = *static_cast<float*>(TF_TensorData(lr_tensor));
  float* grad_data = static_cast<float*>(TF_TensorData(grad_tensor));
  float momentum = *static_cast<float*>(TF_TensorData(momentum_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Momentum: accum = momentum * accum + grad; var = var - lr * accum
  for (size_t i = 0; i < num_elements; ++i) {
    accum_data[i] = momentum * accum_data[i] + grad_data[i];
    output_data[i] = var_data[i] - lr * accum_data[i];
  }
  
  delete[] dims;
}
static void MPSApplyMomentum_Delete(void* kernel) {}

static void* MPSApplyAdam_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSApplyAdam_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* var_tensor;
  TF_GetInput(tf_ctx, 0, &var_tensor, TF_NewStatus());
  TF_Tensor* m_tensor;
  TF_GetInput(tf_ctx, 1, &m_tensor, TF_NewStatus());
  TF_Tensor* v_tensor;
  TF_GetInput(tf_ctx, 2, &v_tensor, TF_NewStatus());
  TF_Tensor* beta1_power_tensor;
  TF_GetInput(tf_ctx, 3, &beta1_power_tensor, TF_NewStatus());
  TF_Tensor* beta2_power_tensor;
  TF_GetInput(tf_ctx, 4, &beta2_power_tensor, TF_NewStatus());
  TF_Tensor* lr_tensor;
  TF_GetInput(tf_ctx, 5, &lr_tensor, TF_NewStatus());
  TF_Tensor* beta1_tensor;
  TF_GetInput(tf_ctx, 6, &beta1_tensor, TF_NewStatus());
  TF_Tensor* beta2_tensor;
  TF_GetInput(tf_ctx, 7, &beta2_tensor, TF_NewStatus());
  TF_Tensor* epsilon_tensor;
  TF_GetInput(tf_ctx, 8, &epsilon_tensor, TF_NewStatus());
  TF_Tensor* grad_tensor;
  TF_GetInput(tf_ctx, 9, &grad_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(var_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(var_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* var_data = static_cast<float*>(TF_TensorData(var_tensor));
  float* m_data = static_cast<float*>(TF_TensorData(m_tensor));
  float* v_data = static_cast<float*>(TF_TensorData(v_tensor));
  float beta1_power = *static_cast<float*>(TF_TensorData(beta1_power_tensor));
  float beta2_power = *static_cast<float*>(TF_TensorData(beta2_power_tensor));
  float lr = *static_cast<float*>(TF_TensorData(lr_tensor));
  float beta1 = *static_cast<float*>(TF_TensorData(beta1_tensor));
  float beta2 = *static_cast<float*>(TF_TensorData(beta2_tensor));
  float epsilon = *static_cast<float*>(TF_TensorData(epsilon_tensor));
  float* grad_data = static_cast<float*>(TF_TensorData(grad_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  float alpha = lr * sqrtf(1.0f - beta2_power) / (1.0f - beta1_power);
  
  // Adam: m = beta1 * m + (1-beta1) * grad
  //       v = beta2 * v + (1-beta2) * grad^2
  //       var = var - alpha * m / (sqrt(v) + epsilon)
  for (size_t i = 0; i < num_elements; ++i) {
    m_data[i] = beta1 * m_data[i] + (1.0f - beta1) * grad_data[i];
    v_data[i] = beta2 * v_data[i] + (1.0f - beta2) * grad_data[i] * grad_data[i];
    output_data[i] = var_data[i] - alpha * m_data[i] / (sqrtf(v_data[i]) + epsilon);
  }
  
  delete[] dims;
}
static void MPSApplyAdam_Delete(void* kernel) {}

static void* MPSApplyRMSprop_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSApplyRMSprop_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* var_tensor;
  TF_GetInput(tf_ctx, 0, &var_tensor, TF_NewStatus());
  TF_Tensor* ms_tensor;
  TF_GetInput(tf_ctx, 1, &ms_tensor, TF_NewStatus());
  TF_Tensor* mom_tensor;
  TF_GetInput(tf_ctx, 2, &mom_tensor, TF_NewStatus());
  TF_Tensor* lr_tensor;
  TF_GetInput(tf_ctx, 3, &lr_tensor, TF_NewStatus());
  TF_Tensor* rho_tensor;
  TF_GetInput(tf_ctx, 4, &rho_tensor, TF_NewStatus());
  TF_Tensor* momentum_tensor;
  TF_GetInput(tf_ctx, 5, &momentum_tensor, TF_NewStatus());
  TF_Tensor* epsilon_tensor;
  TF_GetInput(tf_ctx, 6, &epsilon_tensor, TF_NewStatus());
  TF_Tensor* grad_tensor;
  TF_GetInput(tf_ctx, 7, &grad_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(var_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(var_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* var_data = static_cast<float*>(TF_TensorData(var_tensor));
  float* ms_data = static_cast<float*>(TF_TensorData(ms_tensor));
  float* mom_data = static_cast<float*>(TF_TensorData(mom_tensor));
  float lr = *static_cast<float*>(TF_TensorData(lr_tensor));
  float rho = *static_cast<float*>(TF_TensorData(rho_tensor));
  float momentum = *static_cast<float*>(TF_TensorData(momentum_tensor));
  float epsilon = *static_cast<float*>(TF_TensorData(epsilon_tensor));
  float* grad_data = static_cast<float*>(TF_TensorData(grad_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // RMSprop: ms = rho * ms + (1-rho) * grad^2
  //          mom = momentum * mom + lr * grad / sqrt(ms + epsilon)
  //          var = var - mom
  for (size_t i = 0; i < num_elements; ++i) {
    ms_data[i] = rho * ms_data[i] + (1.0f - rho) * grad_data[i] * grad_data[i];
    mom_data[i] = momentum * mom_data[i] + lr * grad_data[i] / sqrtf(ms_data[i] + epsilon);
    output_data[i] = var_data[i] - mom_data[i];
  }
  
  delete[] dims;
}
static void MPSApplyRMSprop_Delete(void* kernel) {}

static void* MPSApplyAdagrad_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSApplyAdagrad_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* var_tensor;
  TF_GetInput(tf_ctx, 0, &var_tensor, TF_NewStatus());
  TF_Tensor* accum_tensor;
  TF_GetInput(tf_ctx, 1, &accum_tensor, TF_NewStatus());
  TF_Tensor* lr_tensor;
  TF_GetInput(tf_ctx, 2, &lr_tensor, TF_NewStatus());
  TF_Tensor* grad_tensor;
  TF_GetInput(tf_ctx, 3, &grad_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(var_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(var_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* var_data = static_cast<float*>(TF_TensorData(var_tensor));
  float* accum_data = static_cast<float*>(TF_TensorData(accum_tensor));
  float lr = *static_cast<float*>(TF_TensorData(lr_tensor));
  float* grad_data = static_cast<float*>(TF_TensorData(grad_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  float epsilon = 1e-7f;
  
  // Adagrad: accum = accum + grad^2
  //          var = var - lr * grad / sqrt(accum + epsilon)
  for (size_t i = 0; i < num_elements; ++i) {
    accum_data[i] += grad_data[i] * grad_data[i];
    output_data[i] = var_data[i] - lr * grad_data[i] / sqrtf(accum_data[i] + epsilon);
  }
  
  delete[] dims;
}
static void MPSApplyAdagrad_Delete(void* kernel) {}

void RegisterOptimizerOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ApplyGradientDescent", platform_name,
                                                &MPSApplyGradientDescent_Create,
                                                &MPSApplyGradientDescent_Compute,
                                                &MPSApplyGradientDescent_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSApplyGradientDescent", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ApplyMomentum", platform_name,
                                                &MPSApplyMomentum_Create,
                                                &MPSApplyMomentum_Compute,
                                                &MPSApplyMomentum_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSApplyMomentum", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ApplyAdam", platform_name,
                                                &MPSApplyAdam_Create,
                                                &MPSApplyAdam_Compute,
                                                &MPSApplyAdam_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSApplyAdam", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ApplyRMSprop", platform_name,
                                                &MPSApplyRMSprop_Create,
                                                &MPSApplyRMSprop_Compute,
                                                &MPSApplyRMSprop_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSApplyRMSprop", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ApplyAdagrad", platform_name,
                                                &MPSApplyAdagrad_Create,
                                                &MPSApplyAdagrad_Compute,
                                                &MPSApplyAdagrad_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSApplyAdagrad", kb, status);
  }
}
