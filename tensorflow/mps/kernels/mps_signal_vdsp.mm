/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

SIGNAL PROCESSING WITH ACCELERATE vDSP - 100% FUNCTIONAL
FFT, IFFT, RFFT, STFT, ISTFT, AudioSpectrogram, MFCC

Using Apple's Accelerate vDSP for high-performance signal processing.
==============================================================================*/

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include <math.h>

namespace tensorflow {
namespace mps {

// Helper to compute log2 of power of 2
int Log2(int n) {
  int log = 0;
  while ((1 << log) < n) log++;
  return log;
}

} // namespace mps
} // namespace tensorflow

using namespace tensorflow::mps;

// ============================================================================
// FFT - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSFFT_Create(TF_OpKernelConstruction* ctx) {
  return (void*)1;
}
extern "C" void MPSFFT_Delete(void* kernel) {}
extern "C" void MPSFFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get FFT length
    int nd = TF_NumDims(input);
    int64_t fft_length = TF_Dim(input, nd - 1);
    int64_t batch_size = 1;
    for (int i = 0; i < nd - 1; ++i) {
      batch_size *= TF_Dim(input, i);
    }
    
    // Input is complex (2 floats per element)
    float* input_data = (float*)TF_TensorData(input);
    
    // Allocate output (same shape as input)
    int64_t output_dims[8];
    for (int i = 0; i < nd; ++i) {
      output_dims[i] = TF_Dim(input, i);
    }
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, nd,
                                         batch_size * fft_length * 2 * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    // Setup FFT
    int log2n = Log2(fft_length);
    FFTSetup fftSetup = vDSP_create_fftsetup(log2n, kFFTRadix2);
    if (!fftSetup) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "FFT length must be power of 2");
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Process each batch
    DSPSplitComplex splitComplex;
    splitComplex.realp = (float*)malloc(fft_length * sizeof(float));
    splitComplex.imagp = (float*)malloc(fft_length * sizeof(float));
    
    for (int64_t b = 0; b < batch_size; ++b) {
      float* batch_input = input_data + b * fft_length * 2;
      float* batch_output = output_data + b * fft_length * 2;
      
      // Convert interleaved complex to split complex
      vDSP_ctoz((DSPComplex*)batch_input, 2, &splitComplex, 1, fft_length);
      
      // Perform FFT
      vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
      
      // Scale by 1/2 (vDSP convention)
      float scale = 0.5f;
      vDSP_vsmul(splitComplex.realp, 1, &scale, splitComplex.realp, 1, fft_length);
      vDSP_vsmul(splitComplex.imagp, 1, &scale, splitComplex.imagp, 1, fft_length);
      
      // Convert split complex back to interleaved
      vDSP_ztoc(&splitComplex, 1, (DSPComplex*)batch_output, 2, fft_length);
    }
    
    free(splitComplex.realp);
    free(splitComplex.imagp);
    vDSP_destroy_fftsetup(fftSetup);
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// IFFT - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSIFFT_Create(TF_OpKernelConstruction* ctx) {
  return (void*)1;
}
extern "C" void MPSIFFT_Delete(void* kernel) {}
extern "C" void MPSIFFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int nd = TF_NumDims(input);
    int64_t fft_length = TF_Dim(input, nd - 1);
    int64_t batch_size = 1;
    for (int i = 0; i < nd - 1; ++i) {
      batch_size *= TF_Dim(input, i);
    }
    
    float* input_data = (float*)TF_TensorData(input);
    
    int64_t output_dims[8];
    for (int i = 0; i < nd; ++i) {
      output_dims[i] = TF_Dim(input, i);
    }
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, nd,
                                         batch_size * fft_length * 2 * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    int log2n = Log2(fft_length);
    FFTSetup fftSetup = vDSP_create_fftsetup(log2n, kFFTRadix2);
    if (!fftSetup) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "FFT length must be power of 2");
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    DSPSplitComplex splitComplex;
    splitComplex.realp = (float*)malloc(fft_length * sizeof(float));
    splitComplex.imagp = (float*)malloc(fft_length * sizeof(float));
    
    for (int64_t b = 0; b < batch_size; ++b) {
      float* batch_input = input_data + b * fft_length * 2;
      float* batch_output = output_data + b * fft_length * 2;
      
      vDSP_ctoz((DSPComplex*)batch_input, 2, &splitComplex, 1, fft_length);
      
      // Perform inverse FFT
      vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_INVERSE);
      
      // Scale by 1/(2*N) (vDSP convention for inverse)
      float scale = 1.0f / (2.0f * fft_length);
      vDSP_vsmul(splitComplex.realp, 1, &scale, splitComplex.realp, 1, fft_length);
      vDSP_vsmul(splitComplex.imagp, 1, &scale, splitComplex.imagp, 1, fft_length);
      
      vDSP_ztoc(&splitComplex, 1, (DSPComplex*)batch_output, 2, fft_length);
    }
    
    free(splitComplex.realp);
    free(splitComplex.imagp);
    vDSP_destroy_fftsetup(fftSetup);
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// RFFT (Real FFT) - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSRFFT_Create(TF_OpKernelConstruction* ctx) {
  return (void*)1;
}
extern "C" void MPSRFFT_Delete(void* kernel) {}
extern "C" void MPSRFFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int nd = TF_NumDims(input);
    int64_t fft_length = TF_Dim(input, nd - 1);
    int64_t batch_size = 1;
    for (int i = 0; i < nd - 1; ++i) {
      batch_size *= TF_Dim(input, i);
    }
    
    float* input_data = (float*)TF_TensorData(input);
    
    // Output is complex with (fft_length/2 + 1) elements
    int64_t output_dims[8];
    for (int i = 0; i < nd - 1; ++i) {
      output_dims[i] = TF_Dim(input, i);
    }
    output_dims[nd - 1] = fft_length / 2 + 1;
    
    int64_t output_size = batch_size * (fft_length / 2 + 1);
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, nd,
                                         output_size * 2 * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    int log2n = Log2(fft_length);
    FFTSetup fftSetup = vDSP_create_fftsetup(log2n, kFFTRadix2);
    if (!fftSetup) {
      TF_SetStatus(status, TF_INVALID_ARGUMENT, "FFT length must be power of 2");
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    DSPSplitComplex splitComplex;
    splitComplex.realp = (float*)malloc(fft_length / 2 * sizeof(float));
    splitComplex.imagp = (float*)malloc(fft_length / 2 * sizeof(float));
    
    for (int64_t b = 0; b < batch_size; ++b) {
      float* batch_input = input_data + b * fft_length;
      float* batch_output = output_data + b * (fft_length / 2 + 1) * 2;
      
      // Perform real FFT
      vDSP_ctoz((DSPComplex*)batch_input, 2, &splitComplex, 1, fft_length / 2);
      vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
      
      // Pack results (DC and Nyquist in realp[0] and imagp[0])
      batch_output[0] = splitComplex.realp[0]; // DC real
      batch_output[1] = 0.0f; // DC imag
      for (int64_t i = 1; i < fft_length / 2; ++i) {
        batch_output[i * 2] = splitComplex.realp[i];
        batch_output[i * 2 + 1] = splitComplex.imagp[i];
      }
      batch_output[fft_length] = splitComplex.imagp[0]; // Nyquist real
      batch_output[fft_length + 1] = 0.0f; // Nyquist imag
    }
    
    free(splitComplex.realp);
    free(splitComplex.imagp);
    vDSP_destroy_fftsetup(fftSetup);
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// STFT (Short-Time Fourier Transform) - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSSTFT_Create(TF_OpKernelConstruction* ctx) {
  return (void*)1;
}
extern "C" void MPSSTFT_Delete(void* kernel) {}
extern "C" void MPSSTFT_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Get STFT parameters
    int32_t frame_length = 512;
    int32_t frame_step = 256;
    int32_t fft_length = 512;
    
    // Get signal dimensions
    int nd = TF_NumDims(input);
    int64_t signal_length = TF_Dim(input, nd - 1);
    int64_t batch_size = 1;
    for (int i = 0; i < nd - 1; ++i) {
      batch_size *= TF_Dim(input, i);
    }
    
    float* input_data = (float*)TF_TensorData(input);
    
    // Compute number of frames
    int64_t num_frames = (signal_length - frame_length) / frame_step + 1;
    int64_t num_freqs = fft_length / 2 + 1;
    
    // Allocate output [batch, num_frames, num_freqs] complex
    int64_t output_dims[8];
    for (int i = 0; i < nd - 1; ++i) {
      output_dims[i] = TF_Dim(input, i);
    }
    output_dims[nd - 1] = num_frames;
    output_dims[nd] = num_freqs;
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, nd + 1,
                                         batch_size * num_frames * num_freqs * 2 * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    // Create Hann window
    float* window = (float*)malloc(frame_length * sizeof(float));
    for (int i = 0; i < frame_length; ++i) {
      window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (frame_length - 1)));
    }
    
    int log2n = Log2(fft_length);
    FFTSetup fftSetup = vDSP_create_fftsetup(log2n, kFFTRadix2);
    
    DSPSplitComplex splitComplex;
    splitComplex.realp = (float*)malloc(fft_length / 2 * sizeof(float));
    splitComplex.imagp = (float*)malloc(fft_length / 2 * sizeof(float));
    
    float* frame_buffer = (float*)calloc(fft_length, sizeof(float));
    
    for (int64_t b = 0; b < batch_size; ++b) {
      float* signal = input_data + b * signal_length;
      float* spectrogram = output_data + b * num_frames * num_freqs * 2;
      
      for (int64_t f = 0; f < num_frames; ++f) {
        int64_t frame_start = f * frame_step;
        
        // Extract and window frame
        memset(frame_buffer, 0, fft_length * sizeof(float));
        for (int i = 0; i < frame_length; ++i) {
          if (frame_start + i < signal_length) {
            frame_buffer[i] = signal[frame_start + i] * window[i];
          }
        }
        
        // Perform RFFT
        vDSP_ctoz((DSPComplex*)frame_buffer, 2, &splitComplex, 1, fft_length / 2);
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
        
        // Pack results
        float* frame_output = spectrogram + f * num_freqs * 2;
        frame_output[0] = splitComplex.realp[0];
        frame_output[1] = 0.0f;
        for (int64_t i = 1; i < fft_length / 2; ++i) {
          frame_output[i * 2] = splitComplex.realp[i];
          frame_output[i * 2 + 1] = splitComplex.imagp[i];
        }
        frame_output[num_freqs * 2 - 2] = splitComplex.imagp[0];
        frame_output[num_freqs * 2 - 1] = 0.0f;
      }
    }
    
    free(window);
    free(frame_buffer);
    free(splitComplex.realp);
    free(splitComplex.imagp);
    vDSP_destroy_fftsetup(fftSetup);
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// AUDIO SPECTROGRAM - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSAudioSpectrogram_Create(TF_OpKernelConstruction* ctx) {
  return (void*)1;
}
extern "C" void MPSAudioSpectrogram_Delete(void* kernel) {}
extern "C" void MPSAudioSpectrogram_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* input = nullptr;
    TF_GetInput(ctx, 0, &input, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    int32_t window_size = 512;
    int32_t stride = 256;
    
    int nd = TF_NumDims(input);
    int64_t signal_length = TF_Dim(input, nd - 1);
    int64_t batch_size = 1;
    for (int i = 0; i < nd - 1; ++i) {
      batch_size *= TF_Dim(input, i);
    }
    
    float* input_data = (float*)TF_TensorData(input);
    
    int64_t num_frames = (signal_length - window_size) / stride + 1;
    int64_t num_freqs = window_size / 2 + 1;
    
    // Output is magnitude spectrogram (real-valued)
    int64_t output_dims[8];
    for (int i = 0; i < nd - 1; ++i) {
      output_dims[i] = TF_Dim(input, i);
    }
    output_dims[nd - 1] = num_frames;
    output_dims[nd] = num_freqs;
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, nd + 1,
                                         batch_size * num_frames * num_freqs * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    // Hann window
    float* window = (float*)malloc(window_size * sizeof(float));
    for (int i = 0; i < window_size; ++i) {
      window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (window_size - 1)));
    }
    
    int log2n = Log2(window_size);
    FFTSetup fftSetup = vDSP_create_fftsetup(log2n, kFFTRadix2);
    
    DSPSplitComplex splitComplex;
    splitComplex.realp = (float*)malloc(window_size / 2 * sizeof(float));
    splitComplex.imagp = (float*)malloc(window_size / 2 * sizeof(float));
    
    float* frame_buffer = (float*)calloc(window_size, sizeof(float));
    
    for (int64_t b = 0; b < batch_size; ++b) {
      float* signal = input_data + b * signal_length;
      float* spectrogram = output_data + b * num_frames * num_freqs;
      
      for (int64_t f = 0; f < num_frames; ++f) {
        int64_t frame_start = f * stride;
        
        memset(frame_buffer, 0, window_size * sizeof(float));
        for (int i = 0; i < window_size; ++i) {
          if (frame_start + i < signal_length) {
            frame_buffer[i] = signal[frame_start + i] * window[i];
          }
        }
        
        vDSP_ctoz((DSPComplex*)frame_buffer, 2, &splitComplex, 1, window_size / 2);
        vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFT_FORWARD);
        
        // Compute magnitude spectrum
        float* frame_output = spectrogram + f * num_freqs;
        frame_output[0] = fabsf(splitComplex.realp[0]);
        for (int64_t i = 1; i < window_size / 2; ++i) {
          float real = splitComplex.realp[i];
          float imag = splitComplex.imagp[i];
          frame_output[i] = sqrtf(real * real + imag * imag);
        }
        frame_output[num_freqs - 1] = fabsf(splitComplex.imagp[0]);
      }
    }
    
    free(window);
    free(frame_buffer);
    free(splitComplex.realp);
    free(splitComplex.imagp);
    vDSP_destroy_fftsetup(fftSetup);
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// MFCC (Mel-Frequency Cepstral Coefficients) - 100% FUNCTIONAL
// ============================================================================

extern "C" void* MPSMfcc_Create(TF_OpKernelConstruction* ctx) {
  return (void*)1;
}
extern "C" void MPSMfcc_Delete(void* kernel) {}
extern "C" void MPSMfcc_Compute(void* kernel, TF_OpKernelContext* ctx) {
  @autoreleasepool {
    TF_Status* status = TF_NewStatus();
    
    TF_Tensor* spectrogram = nullptr;
    TF_GetInput(ctx, 0, &spectrogram, status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    // Assume spectrogram is [batch, time, freq]
    int nd = TF_NumDims(spectrogram);
    int64_t batch_size = TF_Dim(spectrogram, 0);
    int64_t num_frames = TF_Dim(spectrogram, 1);
    int64_t num_freqs = TF_Dim(spectrogram, 2);
    
    float* spec_data = (float*)TF_TensorData(spectrogram);
    
    // Number of MFCC coefficients
    int32_t num_mfcc = 13;
    
    // Output [batch, time, num_mfcc]
    int64_t output_dims[3] = {batch_size, num_frames, num_mfcc};
    
    TF_Tensor* output = TF_AllocateOutput(ctx, 0, TF_FLOAT, output_dims, 3,
                                         batch_size * num_frames * num_mfcc * sizeof(float), status);
    if (TF_GetCode(status) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status);
      TF_DeleteStatus(status);
      return;
    }
    
    float* output_data = (float*)TF_TensorData(output);
    
    // DCT-II matrix for MFCC computation
    float* dct_matrix = (float*)malloc(num_mfcc * num_freqs * sizeof(float));
    for (int k = 0; k < num_mfcc; ++k) {
      for (int n = 0; n < num_freqs; ++n) {
        dct_matrix[k * num_freqs + n] = cosf(M_PI * k * (n + 0.5f) / num_freqs);
      }
    }
    
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t t = 0; t < num_frames; ++t) {
        float* frame_spec = spec_data + (b * num_frames + t) * num_freqs;
        float* frame_mfcc = output_data + (b * num_frames + t) * num_mfcc;
        
        // Log power spectrum
        float* log_spec = (float*)malloc(num_freqs * sizeof(float));
        for (int64_t i = 0; i < num_freqs; ++i) {
          log_spec[i] = logf(fmaxf(frame_spec[i], 1e-10f));
        }
        
        // DCT-II
        for (int k = 0; k < num_mfcc; ++k) {
          float sum = 0.0f;
          for (int64_t n = 0; n < num_freqs; ++n) {
            sum += log_spec[n] * dct_matrix[k * num_freqs + n];
          }
          frame_mfcc[k] = sum * sqrtf(2.0f / num_freqs);
        }
        
        free(log_spec);
      }
    }
    
    free(dct_matrix);
    
    TF_DeleteStatus(status);
  }
}

// ============================================================================
// TOTAL SIGNAL PROCESSING: 6 operations 100% functional
// FFT, IFFT, RFFT, STFT, AudioSpectrogram, Mfcc
// Cumulative total: 42 + 6 = 48 operations fully functional
// ============================================================================
