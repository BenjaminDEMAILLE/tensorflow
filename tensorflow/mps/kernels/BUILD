# TensorFlow MPS Backend - Metal Performance Shaders implementation

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# MPS Kernel Library
cc_library(
    name = "mps_kernels",
    srcs = [
        "mps_conv2d_impl.mm",
        "mps_matmul_impl.mm",
        "mps_activation_impl.mm",
        "mps_pooling_impl.mm",
        "mps_batchnorm_impl.mm",
        "mps_reduction_impl.mm",
        "mps_math_extended_impl.mm",
        "mps_tensor_manip_impl.mm",
        "mps_loss_impl.mm",
        "mps_optimizer_impl.mm",
        "mps_conv_advanced_impl.mm",
        "mps_rnn_impl.mm",
        "mps_attention_impl.mm",
        "mps_image_impl.mm",
        "mps_sparse_impl.mm",
        "mps_embedding_impl.mm",
        "mps_conv3d_fft_impl.mm",
        "mps_control_flow_impl.mm",
        "mps_random_quant_impl.mm",
        # New complete implementations (61 ops)
        "mps_graph_executor.mm",
        "mps_data_executor.mm",
        "mps_linalg_accelerate.mm",
        "mps_signal_vdsp.mm",
        "mps_image_complete.mm",
        "mps_quantization_complete.mm",
        "mps_kernel_registration.cc",
    ],
    hdrs = [],
    copts = [
        "-x",
        "objective-c++",
        "-std=c++17",
        "-fobjc-arc",
        "-fno-objc-exceptions",
        "-Wno-deprecated-declarations",
    ],
    linkopts = [
        "-framework",
        "Metal",
        "-framework",
        "MetalPerformanceShaders",
        "-framework",
        "MetalPerformanceShadersGraph",
        "-framework",
        "Foundation",
        "-framework",
        "Accelerate",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

# MPS Device Plugin
cc_library(
    name = "mps_device",
    srcs = ["mps_device.mm"],
    hdrs = ["mps_device.h"],
    copts = [
        "-x",
        "objective-c++",
        "-std=c++17",
        "-fobjc-arc",
    ],
    linkopts = [
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

# Complete MPS Backend
cc_library(
    name = "mps_backend",
    deps = [
        ":mps_device",
        ":mps_kernels",
    ],
    alwayslink = 1,
)

# Shared library for Python
cc_binary(
    name = "_mps_ops.so",
    linkshared = 1,
    deps = [
        ":mps_backend",
    ],
)

# Test binary
cc_test(
    name = "mps_ops_test",
    size = "small",
    srcs = ["mps_ops_test.cc"],
    deps = [
        ":mps_backend",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_googletest//:gtest",
    ],
)
