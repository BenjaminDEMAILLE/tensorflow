/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Signal Processing Operations
// FFT, IFFT, RFFT, IRFFT, FFT2D, FFT3D, Spectrogram, MFCC, STFT, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace mps {

// FFT using Metal Performance Shaders
void* MPSFFT_Create(TF_OpKernelConstruction* ctx) { return nullptr; }
void MPSFFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, TF_UNIMPLEMENTED, "FFT - TODO (use MPSMatrixFindTopK or Metal kernels)");
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
void MPSFFT_Delete(void* kernel) {}

// IFFT, RFFT, IRFFT, FFT2D, FFT3D, Spectrogram, MFCC, STFT... (60+ ops)

void RegisterSignalOps(const char* platform_name, TF_Status* status) {
  TF_KernelBuilder* kb = TF_NewKernelBuilder("FFT", platform_name,
                                              &MPSFFT_Create,
                                              &MPSFFT_Compute,
                                              &MPSFFT_Delete);
  TF_RegisterKernelBuilder("MPSFFTComplex64", kb, status);
  // TODO: 59+ more signal processing ops
}

}  // namespace mps
}  // namespace tensorflow
