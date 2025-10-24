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

#ifndef TENSORFLOW_MPS_UTILS_MPS_COMMON_H_
#define TENSORFLOW_MPS_UTILS_MPS_COMMON_H_

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Forward declarations
struct MPSDevice;
struct MPSStreamStruct;
struct MPSStream;

namespace tensorflow {
namespace mps {

// Helper functions
inline float HalfToFloat(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  
  if (exp == 0) {
    if (mant == 0) return sign ? -0.0f : 0.0f;
    exp = 1;
    while (!(mant & 0x400)) { mant <<= 1; exp--; }
    mant &= 0x3FF;
  } else if (exp == 0x1F) {
    return mant ? NAN : (sign ? -INFINITY : INFINITY);
  }
  
  uint32_t f_exp = exp - 15 + 127;
  uint32_t f_mant = mant << 13;
  uint32_t f_bits = (sign << 31) | (f_exp << 23) | f_mant;
  
  float result;
  memcpy(&result, &f_bits, sizeof(float));
  return result;
}

inline uint16_t FloatToHalf(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(float));
  
  uint32_t sign = (bits >> 31) & 0x1;
  int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (bits >> 13) & 0x3FF;
  
  if (exp <= 0) {
    if (exp < -10) return sign << 15;
    mant |= 0x400;
    mant >>= (1 - exp);
    return (sign << 15) | mant;
  }
  if (exp >= 0x1F) {
    return (sign << 15) | 0x7C00;
  }
  
  return (sign << 15) | (exp << 10) | mant;
}

inline float BFloat16ToFloat(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
}

inline uint16_t FloatToBFloat16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(float));
  return static_cast<uint16_t>(bits >> 16);
}

}  // namespace mps
}  // namespace tensorflow

#endif  // TENSORFLOW_MPS_UTILS_MPS_COMMON_H_
