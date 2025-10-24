/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <metal_stdlib>
using namespace metal;

struct MorphologyParams {
  int32_t N, H, W, C;
  int32_t H_out, W_out;
  int32_t kH, kW;
  int32_t stride_h, stride_w;
  int32_t rate_h, rate_w;
  int32_t pad_top, pad_left;
};

// Dilation2D forward: y = max(x + filter) over window
kernel void dilation2d_forward(
  device const float* input [[buffer(0)]],
  device const float* filter [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant MorphologyParams& params [[buffer(3)]],
  uint3 gid [[thread_position_in_grid]]
) {
  const int n = gid.z;
  const int oh = gid.y;
  const int ow = gid.x;
  
  if (n >= params.N || oh >= params.H_out || ow >= params.W_out) return;
  
  const int h_start = oh * params.stride_h - params.pad_top;
  const int w_start = ow * params.stride_w - params.pad_left;
  
  for (int c = 0; c < params.C; ++c) {
    float max_val = -INFINITY;
    
    for (int kh = 0; kh < params.kH; ++kh) {
      int h_in = h_start + kh * params.rate_h;
      if (h_in < 0 || h_in >= params.H) continue;
      
      for (int kw = 0; kw < params.kW; ++kw) {
        int w_in = w_start + kw * params.rate_w;
        if (w_in < 0 || w_in >= params.W) continue;
        
        int input_idx = ((n * params.H + h_in) * params.W + w_in) * params.C + c;
        int filter_idx = (kh * params.kW + kw) * params.C + c;
        float val = input[input_idx] + filter[filter_idx];
        max_val = max(max_val, val);
      }
    }
    
    int output_idx = ((n * params.H_out + oh) * params.W_out + ow) * params.C + c;
    output[output_idx] = max_val;
  }
}

// Erosion2D forward: y = min(x - filter) over window
kernel void erosion2d_forward(
  device const float* input [[buffer(0)]],
  device const float* filter [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant MorphologyParams& params [[buffer(3)]],
  uint3 gid [[thread_position_in_grid]]
) {
  const int n = gid.z;
  const int oh = gid.y;
  const int ow = gid.x;
  
  if (n >= params.N || oh >= params.H_out || ow >= params.W_out) return;
  
  const int h_start = oh * params.stride_h - params.pad_top;
  const int w_start = ow * params.stride_w - params.pad_left;
  
  for (int c = 0; c < params.C; ++c) {
    float min_val = INFINITY;
    
    for (int kh = 0; kh < params.kH; ++kh) {
      int h_in = h_start + kh * params.rate_h;
      if (h_in < 0 || h_in >= params.H) continue;
      
      for (int kw = 0; kw < params.kW; ++kw) {
        int w_in = w_start + kw * params.rate_w;
        if (w_in < 0 || w_in >= params.W) continue;
        
        int input_idx = ((n * params.H + h_in) * params.W + w_in) * params.C + c;
        int filter_idx = (kh * params.kW + kw) * params.C + c;
        float val = input[input_idx] - filter[filter_idx];
        min_val = min(min_val, val);
      }
    }
    
    int output_idx = ((n * params.H_out + oh) * params.W_out + ow) * params.C + c;
    output[output_idx] = min_val;
  }
}

// Dilation2D backward wrt input
kernel void dilation2d_backprop_input(
  device const float* input [[buffer(0)]],
  device const float* filter [[buffer(1)]],
  device const float* grad_output [[buffer(2)]],
  device float* grad_input [[buffer(3)]],
  constant MorphologyParams& params [[buffer(4)]],
  uint3 gid [[thread_position_in_grid]]
) {
  const int n = gid.z;
  const int oh = gid.y;
  const int ow = gid.x;
  
  if (n >= params.N || oh >= params.H_out || ow >= params.W_out) return;
  
  const int h_start = oh * params.stride_h - params.pad_top;
  const int w_start = ow * params.stride_w - params.pad_left;
  const float eps = 1e-6f;
  
  for (int c = 0; c < params.C; ++c) {
    // Find max value
    float max_val = -INFINITY;
    for (int kh = 0; kh < params.kH; ++kh) {
      int h_in = h_start + kh * params.rate_h;
      if (h_in < 0 || h_in >= params.H) continue;
      for (int kw = 0; kw < params.kW; ++kw) {
        int w_in = w_start + kw * params.rate_w;
        if (w_in < 0 || w_in >= params.W) continue;
        int input_idx = ((n * params.H + h_in) * params.W + w_in) * params.C + c;
        int filter_idx = (kh * params.kW + kw) * params.C + c;
        float val = input[input_idx] + filter[filter_idx];
        max_val = max(max_val, val);
      }
    }
    
    // Distribute gradient to input positions that achieved max
    int grad_out_idx = ((n * params.H_out + oh) * params.W_out + ow) * params.C + c;
    float grad = grad_output[grad_out_idx];
    
    for (int kh = 0; kh < params.kH; ++kh) {
      int h_in = h_start + kh * params.rate_h;
      if (h_in < 0 || h_in >= params.H) continue;
      for (int kw = 0; kw < params.kW; ++kw) {
        int w_in = w_start + kw * params.rate_w;
        if (w_in < 0 || w_in >= params.W) continue;
        int input_idx = ((n * params.H + h_in) * params.W + w_in) * params.C + c;
        int filter_idx = (kh * params.kW + kw) * params.C + c;
        float val = input[input_idx] + filter[filter_idx];
        if (fabs(val - max_val) <= eps) {
          atomic_fetch_add_explicit((device atomic<float>*)&grad_input[input_idx], grad, memory_order_relaxed);
        }
      }
    }
  }
}

// Dilation2D backward wrt filter
kernel void dilation2d_backprop_filter(
  device const float* input [[buffer(0)]],
  device const float* filter [[buffer(1)]],
  device const float* grad_output [[buffer(2)]],
  device float* grad_filter [[buffer(3)]],
  constant MorphologyParams& params [[buffer(4)]],
  uint3 gid [[thread_position_in_grid]]
) {
  const int n = gid.z;
  const int oh = gid.y;
  const int ow = gid.x;
  
  if (n >= params.N || oh >= params.H_out || ow >= params.W_out) return;
  
  const int h_start = oh * params.stride_h - params.pad_top;
  const int w_start = ow * params.stride_w - params.pad_left;
  const float eps = 1e-6f;
  
  for (int c = 0; c < params.C; ++c) {
    // Find max value
    float max_val = -INFINITY;
    for (int kh = 0; kh < params.kH; ++kh) {
      int h_in = h_start + kh * params.rate_h;
      if (h_in < 0 || h_in >= params.H) continue;
      for (int kw = 0; kw < params.kW; ++kw) {
        int w_in = w_start + kw * params.rate_w;
        if (w_in < 0 || w_in >= params.W) continue;
        int input_idx = ((n * params.H + h_in) * params.W + w_in) * params.C + c;
        int filter_idx = (kh * params.kW + kw) * params.C + c;
        float val = input[input_idx] + filter[filter_idx];
        max_val = max(max_val, val);
      }
    }
    
    // Distribute gradient to filter positions that achieved max
    int grad_out_idx = ((n * params.H_out + oh) * params.W_out + ow) * params.C + c;
    float grad = grad_output[grad_out_idx];
    
    for (int kh = 0; kh < params.kH; ++kh) {
      int h_in = h_start + kh * params.rate_h;
      if (h_in < 0 || h_in >= params.H) continue;
      for (int kw = 0; kw < params.kW; ++kw) {
        int w_in = w_start + kw * params.rate_w;
        if (w_in < 0 || w_in >= params.W) continue;
        int input_idx = ((n * params.H + h_in) * params.W + w_in) * params.C + c;
        int filter_idx = (kh * params.kW + kw) * params.C + c;
        float val = input[input_idx] + filter[filter_idx];
        if (fabs(val - max_val) <= eps) {
          atomic_fetch_add_explicit((device atomic<float>*)&grad_filter[filter_idx], grad, memory_order_relaxed);
        }
      }
    }
  }
}
