/* Copyright 2025 The TensorFlow Authors. Licensed under the Apache License, Version 2.0 */

// MPS Signal Processing Operations
// FFT, IFFT, RFFT, IRFFT, FFT2D, FFT3D, Spectrogram, MFCC, STFT, etc.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include <complex>

namespace tensorflow {
namespace mps {

// FFT using vDSP from Accelerate framework
struct MPSFFTAttrs {
  int fft_length;
};

void* MPSFFT_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSFFTAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "fft_length", &attrs->fft_length, s);
  if (TF_GetCode(s) != TF_OK) attrs->fft_length = 0;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSFFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSFFTAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  // Input: complex tensor [batch, length]
  int64_t batch = TF_Dim(input, 0);
  int64_t length = TF_Dim(input, 1);
  
  if (attrs->fft_length == 0) attrs->fft_length = length;
  
  // Use vDSP for FFT computation
  int log2n = (int)log2(attrs->fft_length);
  FFTSetup fftSetup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
  
  std::complex<float>* input_data = static_cast<std::complex<float>*>(TF_TensorData(input));
  
  // Allocate output
  size_t output_bytes = batch * attrs->fft_length * sizeof(std::complex<float>);
  int64_t out_dims[] = {batch, attrs->fft_length};
  TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_COMPLEX64, out_dims, 2, output_bytes, s);
  
  if (TF_GetCode(s) != TF_OK) {
    vDSP_destroy_fftsetup(fftSetup);
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  std::complex<float>* output_data = static_cast<std::complex<float>*>(TF_TensorData(tf_output));
  
  // Perform FFT for each batch
  DSPSplitComplex splitComplex;
  float* realp = new float[attrs->fft_length];
  float* imagp = new float[attrs->fft_length];
  splitComplex.realp = realp;
  splitComplex.imagp = imagp;
  
  for (int64_t b = 0; b < batch; b++) {
    // Convert to split complex format
    for (int i = 0; i < attrs->fft_length; i++) {
      if (b * length + i < batch * length) {
        realp[i] = input_data[b * length + i].real();
        imagp[i] = input_data[b * length + i].imag();
      } else {
        realp[i] = 0.0f;
        imagp[i] = 0.0f;
      }
    }
    
    // Compute FFT
    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
    
    // Convert back to interleaved format
    for (int i = 0; i < attrs->fft_length; i++) {
      output_data[b * attrs->fft_length + i] = std::complex<float>(realp[i], imagp[i]);
    }
  }
  
  delete[] realp;
  delete[] imagp;
  vDSP_destroy_fftsetup(fftSetup);
  
  TF_DeleteStatus(s);
}

void MPSFFT_Delete(void* kernel) {
  delete static_cast<MPSFFTAttrs*>(kernel);
}

// IFFT (Inverse FFT)
void* MPSIFFT_Create(TF_OpKernelConstruction* ctx) {
  auto* attrs = new MPSFFTAttrs();
  TF_Status* s = TF_NewStatus();
  
  TF_OpKernelConstruction_GetAttrInt32(ctx, "fft_length", &attrs->fft_length, s);
  if (TF_GetCode(s) != TF_OK) attrs->fft_length = 0;
  
  TF_DeleteStatus(s);
  return attrs;
}

void MPSIFFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSFFTAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t batch = TF_Dim(input, 0);
  int64_t length = TF_Dim(input, 1);
  
  if (attrs->fft_length == 0) attrs->fft_length = length;
  
  int log2n = (int)log2(attrs->fft_length);
  FFTSetup fftSetup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
  
  std::complex<float>* input_data = static_cast<std::complex<float>*>(TF_TensorData(input));
  
  size_t output_bytes = batch * attrs->fft_length * sizeof(std::complex<float>);
  int64_t out_dims[] = {batch, attrs->fft_length};
  TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_COMPLEX64, out_dims, 2, output_bytes, s);
  
  if (TF_GetCode(s) != TF_OK) {
    vDSP_destroy_fftsetup(fftSetup);
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  std::complex<float>* output_data = static_cast<std::complex<float>*>(TF_TensorData(tf_output));
  
  DSPSplitComplex splitComplex;
  float* realp = new float[attrs->fft_length];
  float* imagp = new float[attrs->fft_length];
  splitComplex.realp = realp;
  splitComplex.imagp = imagp;
  
  float scale = 1.0f / attrs->fft_length;
  
  for (int64_t b = 0; b < batch; b++) {
    for (int i = 0; i < attrs->fft_length; i++) {
      if (b * length + i < batch * length) {
        realp[i] = input_data[b * length + i].real();
        imagp[i] = input_data[b * length + i].imag();
      } else {
        realp[i] = 0.0f;
        imagp[i] = 0.0f;
      }
    }
    
    // Compute IFFT
    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_INVERSE);
    
    // Scale and convert back
    for (int i = 0; i < attrs->fft_length; i++) {
      output_data[b * attrs->fft_length + i] = std::complex<float>(realp[i] * scale, imagp[i] * scale);
    }
  }
  
  delete[] realp;
  delete[] imagp;
  vDSP_destroy_fftsetup(fftSetup);
  
  TF_DeleteStatus(s);
}

void MPSIFFT_Delete(void* kernel) {
  delete static_cast<MPSFFTAttrs*>(kernel);
}

// RFFT (Real FFT)
void* MPSRFFT_Create(TF_OpKernelConstruction* ctx) {
  return MPSFFT_Create(ctx);
}

void MPSRFFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  auto* attrs = static_cast<MPSFFTAttrs*>(kernel);
  TF_Status* s = TF_NewStatus();
  
  TF_Tensor* input = nullptr;
  TF_GetInput(ctx, 0, &input, s);
  if (TF_GetCode(s) != TF_OK) { TF_OpKernelContext_Failure(ctx, s); TF_DeleteStatus(s); return; }
  
  int64_t batch = TF_Dim(input, 0);
  int64_t length = TF_Dim(input, 1);
  
  if (attrs->fft_length == 0) attrs->fft_length = length;
  
  int log2n = (int)log2(attrs->fft_length);
  FFTSetup fftSetup = vDSP_create_fftsetup(log2n, FFT_RADIX2);
  
  float* input_data = static_cast<float*>(TF_TensorData(input));
  
  // RFFT output is half size + 1
  int64_t output_length = attrs->fft_length / 2 + 1;
  size_t output_bytes = batch * output_length * sizeof(std::complex<float>);
  int64_t out_dims[] = {batch, output_length};
  TF_Tensor* tf_output = TF_AllocateOutput(ctx, 0, TF_COMPLEX64, out_dims, 2, output_bytes, s);
  
  if (TF_GetCode(s) != TF_OK) {
    vDSP_destroy_fftsetup(fftSetup);
    TF_OpKernelContext_Failure(ctx, s);
    TF_DeleteStatus(s);
    return;
  }
  
  std::complex<float>* output_data = static_cast<std::complex<float>*>(TF_TensorData(tf_output));
  
  DSPSplitComplex splitComplex;
  float* realp = new float[attrs->fft_length / 2];
  float* imagp = new float[attrs->fft_length / 2];
  splitComplex.realp = realp;
  splitComplex.imagp = imagp;
  
  for (int64_t b = 0; b < batch; b++) {
    // Real to complex FFT
    vDSP_ctoz((DSPComplex*)(input_data + b * length), 2, &splitComplex, 1, attrs->fft_length / 2);
    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
    
    for (int i = 0; i < output_length; i++) {
      if (i < attrs->fft_length / 2) {
        output_data[b * output_length + i] = std::complex<float>(realp[i], imagp[i]);
      }
    }
  }
  
  delete[] realp;
  delete[] imagp;
  vDSP_destroy_fftsetup(fftSetup);
  
  TF_DeleteStatus(s);
}

void MPSRFFT_Delete(void* kernel) {
  delete static_cast<MPSFFTAttrs*>(kernel);
}

void RegisterSignalOps(const char* platform_name, TF_Status* status) {
  // FFT
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("FFT", platform_name,
                                                &MPSFFT_Create,
                                                &MPSFFT_Compute,
                                                &MPSFFT_Delete);
    TF_RegisterKernelBuilder("MPSFFT", kb, status);
  }
  
  // IFFT
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("IFFT", platform_name,
                                                &MPSIFFT_Create,
                                                &MPSIFFT_Compute,
                                                &MPSIFFT_Delete);
    TF_RegisterKernelBuilder("MPSIFFT", kb, status);
  }
  
  // RFFT
  {
    TF_KernelBuilder* kb = TF_NewKernelBuilder("RFFT", platform_name,
                                                &MPSRFFT_Create,
                                                &MPSRFFT_Compute,
                                                &MPSRFFT_Delete);
    TF_RegisterKernelBuilder("MPSRFFT", kb, status);
  }
  
  // TODO: 57+ more signal processing ops
  // IRFFT, FFT2D, FFT3D, Spectrogram, MFCC, STFT, MelFilterbank, etc.
}

}  // namespace mps
}  // namespace tensorflow
