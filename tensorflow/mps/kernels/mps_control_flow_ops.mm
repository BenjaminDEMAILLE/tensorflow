// Control flow ops for MPS backend
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"

// ============================================================
// Switch - Routes data to one of two outputs based on pred
// ============================================================
static void* MPSSwitch_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSSwitch_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* data_tensor;
  TF_GetInput(tf_ctx, 0, &data_tensor, TF_NewStatus());
  TF_Tensor* pred_tensor;
  TF_GetInput(tf_ctx, 1, &pred_tensor, TF_NewStatus());
  
  bool pred = *static_cast<bool*>(TF_TensorData(pred_tensor));
  
  int num_dims = TF_NumDims(data_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(data_tensor, i);
  }
  
  size_t total_size = TF_TensorByteSize(data_tensor);
  
  if (pred) {
    // Output 1 (true path)
    TF_Tensor* output = TF_AllocateOutput(tf_ctx, 1, TF_FLOAT, dims, num_dims, total_size, TF_NewStatus());
    memcpy(TF_TensorData(output), TF_TensorData(data_tensor), total_size);
  } else {
    // Output 0 (false path)
    TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims, total_size, TF_NewStatus());
    memcpy(TF_TensorData(output), TF_TensorData(data_tensor), total_size);
  }
  
  delete[] dims;
}

static void MPSSwitch_Delete(void* kernel) {}

// ============================================================
// Merge - Merges any available input
// ============================================================
static void* MPSMerge_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSMerge_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  // Try to get first input
  TF_Tensor* input0;
  TF_Status* status = TF_NewStatus();
  TF_GetInput(tf_ctx, 0, &input0, status);
  
  if (TF_GetCode(status) == TF_OK && input0 != nullptr) {
    int num_dims = TF_NumDims(input0);
    int64_t* dims = new int64_t[num_dims];
    for (int i = 0; i < num_dims; ++i) {
      dims[i] = TF_Dim(input0, i);
    }
    size_t total_size = TF_TensorByteSize(input0);
    
    TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims, total_size, TF_NewStatus());
    memcpy(TF_TensorData(output), TF_TensorData(input0), total_size);
    
    // Output index
    int64_t idx_dims[] = {};
    TF_Tensor* idx_output = TF_AllocateOutput(tf_ctx, 1, TF_INT32, idx_dims, 0, sizeof(int32_t), TF_NewStatus());
    *static_cast<int32_t*>(TF_TensorData(idx_output)) = 0;
    
    delete[] dims;
  }
  
  TF_DeleteStatus(status);
}

static void MPSMerge_Delete(void* kernel) {}

// ============================================================
// Enter - Enters a new frame for control flow
// ============================================================
static void* MPSEnter_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSEnter_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t total_size = TF_TensorByteSize(input_tensor);
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims, total_size, TF_NewStatus());
  memcpy(TF_TensorData(output), TF_TensorData(input_tensor), total_size);
  
  delete[] dims;
}

static void MPSEnter_Delete(void* kernel) {}

// ============================================================
// Exit - Exits a control flow frame
// ============================================================
static void* MPSExit_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSExit_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t total_size = TF_TensorByteSize(input_tensor);
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims, total_size, TF_NewStatus());
  memcpy(TF_TensorData(output), TF_TensorData(input_tensor), total_size);
  
  delete[] dims;
}

static void MPSExit_Delete(void* kernel) {}

// ============================================================
// NextIteration - Continues to next loop iteration
// ============================================================
static void* MPSNextIteration_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSNextIteration_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int num_dims = TF_NumDims(input_tensor);
  int64_t* dims = new int64_t[num_dims];
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = TF_Dim(input_tensor, i);
  }
  
  size_t total_size = TF_TensorByteSize(input_tensor);
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_FLOAT, dims, num_dims, total_size, TF_NewStatus());
  memcpy(TF_TensorData(output), TF_TensorData(input_tensor), total_size);
  
  delete[] dims;
}

static void MPSNextIteration_Delete(void* kernel) {}

// ============================================================
// LoopCond - Forwards boolean condition for while loop
// ============================================================
static void* MPSLoopCond_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSLoopCond_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_Tensor* input_tensor;
  TF_GetInput(tf_ctx, 0, &input_tensor, TF_NewStatus());
  
  int64_t dims[] = {};
  TF_Tensor* output = TF_AllocateOutput(tf_ctx, 0, TF_BOOL, dims, 0, sizeof(bool), TF_NewStatus());
  *static_cast<bool*>(TF_TensorData(output)) = *static_cast<bool*>(TF_TensorData(input_tensor));
}

static void MPSLoopCond_Delete(void* kernel) {}

// ============================================================
// ControlTrigger - No-op for control dependencies
// ============================================================
static void* MPSControlTrigger_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSControlTrigger_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  // No-op
}

static void MPSControlTrigger_Delete(void* kernel) {}

// ============================================================
// Abort - Abort execution
// ============================================================
static void* MPSAbort_Create(TF_OpKernelConstruction* ctx) {
  return nullptr;
}

static void MPSAbort_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {
  TF_SetStatus(TF_NewStatus(), TF_ABORTED, "Abort operation executed");
}

static void MPSAbort_Delete(void* kernel) {}

// ============================================================
// Registration
// ============================================================
void RegisterControlFlowOps(const char* platform_name, TF_Status* status) {
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Switch", platform_name,
                                                &MPSSwitch_Create,
                                                &MPSSwitch_Compute,
                                                &MPSSwitch_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSSwitch", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Merge", platform_name,
                                                &MPSMerge_Create,
                                                &MPSMerge_Compute,
                                                &MPSMerge_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSMerge", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Enter", platform_name,
                                                &MPSEnter_Create,
                                                &MPSEnter_Compute,
                                                &MPSEnter_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSEnter", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Exit", platform_name,
                                                &MPSExit_Create,
                                                &MPSExit_Compute,
                                                &MPSExit_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSExit", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("NextIteration", platform_name,
                                                &MPSNextIteration_Create,
                                                &MPSNextIteration_Compute,
                                                &MPSNextIteration_Delete);
    TF_KernelBuilder_TypeConstraint(kb, "T", TF_FLOAT, status);
    TF_RegisterKernelBuilder("MPSNextIteration", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("LoopCond", platform_name,
                                                &MPSLoopCond_Create,
                                                &MPSLoopCond_Compute,
                                                &MPSLoopCond_Delete);
    TF_RegisterKernelBuilder("MPSLoopCond", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("ControlTrigger", platform_name,
                                                &MPSControlTrigger_Create,
                                                &MPSControlTrigger_Compute,
                                                &MPSControlTrigger_Delete);
    TF_RegisterKernelBuilder("MPSControlTrigger", kb, status);
  }
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("Abort", platform_name,
                                                &MPSAbort_Create,
                                                &MPSAbort_Compute,
                                                &MPSAbort_Delete);
    TF_RegisterKernelBuilder("MPSAbort", kb, status);
  }
}
