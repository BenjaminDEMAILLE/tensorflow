/* Copyright 2025 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_MPS_UTILS_MPS_UTILS_H_
#define TENSORFLOW_MPS_UTILS_MPS_UTILS_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <cstring>

namespace tensorflow {
namespace mps {

// ---- DType Conversion Utilities ----

// BFloat16 conversions
inline float BFloat16ToFloat(uint16_t bf) {
  uint32_t bits = ((uint32_t)bf) << 16;
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}

inline uint16_t FloatToBFloat16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t round_bias = 0x7FFF + lsb;
  bits += round_bias;
  return (uint16_t)(bits >> 16);
}

// Float16 (IEEE half precision) conversions
inline uint16_t FloatToHalf(float f) {
  union { uint32_t u; float f; } v = {0}; v.f = f;
  uint32_t x = v.u;
  uint32_t sign = (x >> 16) & 0x8000;
  uint32_t mant = x & 0x007FFFFF;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;  // underflow to zero
    mant = (mant | 0x00800000) >> (1 - exp);
    return (uint16_t)(sign | (mant + 0x00001000) >> 13);
  } else if (exp >= 31) {
    return (uint16_t)(sign | 0x7C00);  // Inf
  }
  return (uint16_t)(sign | (exp << 10) | ((mant + 0x00001000) >> 13));
}

inline float HalfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x03FF;
  uint32_t bits;
  
  if (exp == 0) {
    if (mant == 0) { 
      bits = sign; 
    } else {
      // subnormal
      exp = 1; 
      while ((mant & 0x0400) == 0) { 
        mant <<= 1; 
        --exp; 
      }
      mant &= 0x03FF; 
      exp += (127 - 15);
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000 | (mant << 13);  // Inf/NaN
  } else {
    exp = exp + (127 - 15);
    bits = sign | (exp << 23) | (mant << 13);
  }
  
  float f; 
  memcpy(&f, &bits, sizeof(f)); 
  return f;
}

// ---- Objective-C Bridge Utilities ----

template <typename T>
inline void* RetainToOpaque(T obj) {
  return (__bridge_retained void*)obj;
}

template <typename T>
inline T TransferFromOpaque(void* p) {
  return (__bridge_transfer T)p;
}

template <typename T>
inline T BridgeNoTransfer(void* p) {
  return (__bridge T)p;
}

}  // namespace mps
}  // namespace tensorflow

#endif  // TENSORFLOW_MPS_UTILS_MPS_UTILS_H_
