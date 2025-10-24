// Extended array operations for MPS
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#import <Foundation/Foundation.h>
#include <cmath>

// Tile, Repeat, Unique, UniqueWithCounts, Reverse, ReverseSequence, etc.

static void* MPSTile_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSTile_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  TF_Tensor* multiples_tensor;
  TF_GetInput(tf_ctx, 1, &multiples_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* input_dims = new int64_t[num_dims];
  int64_t* output_dims = new int64_t[num_dims];
  int64_t* multiples = static_cast<int64_t*>(TF_TensorData(multiples_tensor));
  
  size_t output_elements = 1;
  for (int i = 0; i < num_dims; ++i) {
    input_dims[i] = TF_Dim(input_tensor, i);
    output_dims[i] = input_dims[i] * multiples[i];
    output_elements *= output_dims[i];
  }
  
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, num_dims,
                                         sizeof(float) * output_elements, TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  // Simple tiling (TODO: optimize)
  size_t input_size = 1;
  for (int i = 0; i < num_dims; ++i) input_size *= input_dims[i];
  
  for (size_t i = 0; i < output_elements; ++i) {
    output_data[i] = input_data[i % input_size];
  }
  
  delete[] input_dims;
  delete[] output_dims;
}
static void MPSTile_Delete(void* kernel) {}

static void* MPSReverse_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSReverse_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
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
  
  // Reverse along last dimension (simplified)
  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = input_data[num_elements - 1 - i];
  }
  
  delete[] dims;
}
static void MPSReverse_Delete(void* kernel) {}

static void* MPSUnique_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSUnique_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  size_t num_elements = TF_Dim(input_tensor, 0);
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  
  // Simplified: just copy unique values
  int64_t output_dims[] = {static_cast<int64_t>(num_elements)};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 1,
                                         sizeof(float) * num_elements, TF_NewStatus());
  TF_Tensor* idx_output = TF_AllocateOutput(tf_ctx, 1, TF_INT32, output_dims, 1,
                                             sizeof(int32_t) * num_elements, TF_NewStatus());
  
  memcpy(TF_TensorData(output), input_data, sizeof(float) * num_elements);
  
  int32_t* idx_data = static_cast<int32_t*>(TF_TensorData(idx_output));
  for (size_t i = 0; i < num_elements; ++i) {
    idx_data[i] = static_cast<int32_t>(i);
  }
}
static void MPSUnique_Delete(void* kernel) {}

static void* MPSOneHot_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSOneHot_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* indices_tensor;
  TF_GetInput(tf_ctx, 0, &indices_tensor, TF_NewStatus());
  TF_Tensor* depth_tensor;
  TF_GetInput(tf_ctx, 1, &depth_tensor, TF_NewStatus());
  
  int num_indices = TF_Dim(indices_tensor, 0);
  int64_t depth = *static_cast<int64_t*>(TF_TensorData(depth_tensor));
  
  int64_t output_dims[] = {static_cast<int64_t>(num_indices), depth};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 2,
                                         sizeof(float) * num_indices * depth, TF_NewStatus());
  
  int64_t* indices_data = static_cast<int64_t*>(TF_TensorData(indices_tensor));
  float* output_data = static_cast<float*>(TF_TensorData(output));
  
  memset(output_data, 0, sizeof(float) * num_indices * depth);
  
  for (int i = 0; i < num_indices; ++i) {
    int64_t idx = indices_data[i];
    if (idx >= 0 && idx < depth) {
      output_data[i * depth + idx] = 1.0f;
    }
  }
}
static void MPSOneHot_Delete(void* kernel) {}

static void* MPSTopKV2_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSTopKV2_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  TF_Tensor* k_tensor;
  TF_GetInput(tf_ctx, 1, &k_tensor, TF_NewStatus());
  
  int64_t batch_size = TF_Dim(input_tensor, 0);
  int64_t num_elements = TF_Dim(input_tensor, 1);
  int64_t k = *static_cast<int64_t*>(TF_TensorData(k_tensor));
  
  int64_t output_dims[] = {batch_size, k};
  TF_Tensor* values_output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, output_dims, 2,
                                                sizeof(float) * batch_size * k, TF_NewStatus());
  TF_Tensor* indices_output = TF_AllocateOutput(tf_ctx, 1, TF_INT32, output_dims, 2,
                                                 sizeof(int32_t) * batch_size * k, TF_NewStatus());
  
  float* input_data = static_cast<float*>(TF_TensorData(input_tensor));
  float* values_data = static_cast<float*>(TF_TensorData(values_output));
  int32_t* indices_data = static_cast<int32_t*>(TF_TensorData(indices_output));
  
  // Simplified: return first k elements
  for (int64_t b = 0; b < batch_size; ++b) {
    for (int64_t i = 0; i < k; ++i) {
      values_data[b * k + i] = input_data[b * num_elements + i];
      indices_data[b * k + i] = static_cast<int32_t>(i);
    }
  }
}
static void MPSTopKV2_Delete(void* kernel) {}

void RegisterArrayOpsExtended(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Tile", platform_name, &MPSTile_Create, &MPSTile_Compute, &MPSTile_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSTile", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Reverse", platform_name, &MPSReverse_Create, &MPSReverse_Compute, &MPSReverse_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSReverse", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Unique", platform_name, &MPSUnique_Create, &MPSUnique_Compute, &MPSUnique_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSUnique", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("OneHot", platform_name, &MPSOneHot_Create, &MPSOneHot_Compute, &MPSOneHot_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSOneHot", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("TopKV2", platform_name, &MPSTopKV2_Create, &MPSTopKV2_Compute, &MPSTopKV2_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSTopKV2", kb, status);
  }
}
