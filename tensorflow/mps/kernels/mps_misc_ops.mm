// Miscellaneous operations for MPS
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#import <Foundation/Foundation.h>
#include <cmath>

// Cast, Identity, Print, Assert, CheckNumerics, etc.

static void* MPSCast_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSCast_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
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
  
  // Simple memcpy (TODO: actual type casting)
  memcpy(TF_TensorData(output), TF_TensorData(input_tensor), sizeof(float) * num_elements);
  
  delete[] dims;
}
static void MPSCast_Delete(void* kernel) {}

static void* MPSIdentity_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSIdentity_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
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
static void MPSIdentity_Delete(void* kernel) {}

static void* MPSCheckNumerics_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSCheckNumerics_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
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
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Check for NaN/Inf and copy
  for (size_t i = 0; i < num_elements; ++i) {
    if (std::isnan(input_data[i]) || std::isinf(input_data[i])) {
      // TODO: Set error status
      output_data[i] = 0.0f;
    } else {
      output_data[i] = input_data[i];
    }
  }
  
  delete[] dims;
}
static void MPSCheckNumerics_Delete(void* kernel) {}

static void* MPSClipByValue_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSClipByValue_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* t_tensor;
  TF_GetInput(tf_ctx, 0, &t_tensor, TF_NewStatus());
  TF_Tensor* clip_min_tensor;
  TF_GetInput(tf_ctx, 1, &clip_min_tensor, TF_NewStatus());
  TF_Tensor* clip_max_tensor;
  TF_GetInput(tf_ctx, 2, &clip_max_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(t_tensor);
  int64_t* dims = new int64_t[num_dims];
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(t_tensor, i);
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float* t_data = static_cast<float*>(TF_TensorData(t_tensor));
  float clip_min = *static_cast<float*>(TF_TensorData(clip_min_tensor));
  float clip_max = *static_cast<float*>(TF_TensorData(clip_max_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  for (size_t i = 0; i < num_elements; ++i) {
    float val = t_data[i];
    output_data[i] = (val < clip_min) ? clip_min : ((val > clip_max) ? clip_max : val);
  }
  
  delete[] dims;
}
static void MPSClipByValue_Delete(void* kernel) {}

static void* MPSFill_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSFill_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* dims_tensor;
  TF_GetInput(tf_ctx, 0, &dims_tensor, TF_NewStatus());
  TF_Tensor* value_tensor;
  TF_GetInput(tf_ctx, 1, &value_tensor, TF_NewStatus());
  
  int num_dims = TF_Dim(dims_tensor, 0);
  int64_t* dims = static_cast<int64_t*>(TF_TensorData(dims_tensor));
  
  size_t num_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_elements *= dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims,
                                         sizeof(float) * num_elements, TF_NewStatus());
  
  float value = *static_cast<float*>(TF_TensorData(value_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = value;
  }
}
static void MPSFill_Delete(void* kernel) {}

void RegisterMiscOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Cast", platform_name, &MPSCast_Create, &MPSCast_Compute, &MPSCast_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "SrcT", TF_FLOAT, status);
    TF_KernelBuilder_TypeConstraint(kb, "DstT", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSCast", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Identity", platform_name, &MPSIdentity_Create, &MPSIdentity_Compute, &MPSIdentity_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSIdentity", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("CheckNumerics", platform_name, &MPSCheckNumerics_Create, &MPSCheckNumerics_Compute, &MPSCheckNumerics_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSCheckNumerics", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ClipByValue", platform_name, &MPSClipByValue_Create, &MPSClipByValue_Compute, &MPSClipByValue_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSClipByValue", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Fill", platform_name, &MPSFill_Create, &MPSFill_Compute, &MPSFill_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSFill", kb, status);
  }
}
