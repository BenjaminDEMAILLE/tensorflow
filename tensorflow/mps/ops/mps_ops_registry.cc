// MPS Operations Registry Implementation

#include "tensorflow/mps/ops/mps_ops_registry.h"

namespace tensorflow {
namespace mps {

void RegisterAllOps(TF_Status* status) {
  // Register all neural network operations
  RegisterNNOps(status);
  if (TF_GetCode(status) != TF_OK) return;
  
  // Register all elementwise and tensor operations
  RegisterElementwiseOps(status);
  if (TF_GetCode(status) != TF_OK) return;
}

}  // namespace mps
}  // namespace tensorflow
