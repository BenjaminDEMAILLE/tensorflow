// Dataset operations for MPS (simplified CPU fallback)
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

// TensorSliceDataset, BatchDataset, MapDataset, etc. (CPU fallback)

static void* MPSTensorSliceDataset_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSTensorSliceDataset_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {}
static void MPSTensorSliceDataset_Delete(void* kernel) {}

static void* MPSBatchDataset_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
static void MPSBatchDataset_Compute(void* kernel, TF_OpKernelContext* tf_ctx) {}
static void MPSBatchDataset_Delete(void* kernel) {}

void RegisterDatasetOps(const char* platform_name, TF_Status* status) {
  // Dataset ops typically run on CPU
}
