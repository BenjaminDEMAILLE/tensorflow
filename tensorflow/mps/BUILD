load("//tensorflow:strict.default.bzl", "internal_visibility_allowlist")

package(default_visibility = internal_visibility_allowlist())

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "is_macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
)

# Configuration for modular vs monolithic build
config_setting(
    name = "use_modular_mps",
    define_values = {
        "mps_modular": "true",
    },
)

# ============================================================================
# Modular Build Components (CUDA-like structure)
# ============================================================================

# Utilities library
cc_library(
    name = "mps_utils",
    hdrs = ["utils/mps_utils.h"],
    copts = select({
        ":is_macos": ["-fobjc-arc"],
        "//conditions:default": [],
    }),
)

# Device infrastructure
objc_library(
    name = "mps_device",
    srcs = ["device/mps_device.mm"],
    hdrs = ["device/mps_device.h"],
    deps = [
        ":mps_utils",
        "//tensorflow/c:c_api",
        "//tensorflow/c/experimental/stream_executor:stream_executor_hdrs",
    ],
    copts = ["-fobjc-arc"],
)

# Operation registration layer
cc_library(
    name = "mps_ops_registry_hdr",
    hdrs = ["ops/mps_ops_registry.h"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:kernels",
    ],
)

# Main plugin - Monolithic version (default, working)
cc_binary(
    name = "libtensorflow_mps_plugin.dylib",
    srcs = [
        "mps_pluggable_device_plugin.mm",
    ],
    linkshared = 1,
    defines = [
        "PLUGGABLE_DEVICE_SUPPORTED_MACOS",
    ],
    copts = select({
        ":is_macos": [
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }),
    linkopts = select({
        ":is_macos": [
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "MetalPerformanceShadersGraph",
            "-framework", "Foundation",
        ],
        "//conditions:default": [],
    }),
)

# Modular version (experimental - for future migration)
# TODO: Complete kernel extraction before enabling
# cc_binary(
#     name = "libtensorflow_mps_plugin_modular.dylib",
#     srcs = [],
#     deps = [
#         ":mps_device_factory",
#     ],
#     linkshared = 1,
#     defines = [
#         "PLUGGABLE_DEVICE_SUPPORTED_MACOS",
#     ],
#     copts = select({
#         ":is_macos": ["-fobjc-arc"],
#         "//conditions:default": [],
#     }),
#     linkopts = select({
#         ":is_macos": [
#             "-framework", "Metal",
#             "-framework", "MetalPerformanceShaders",
#             "-framework", "MetalPerformanceShadersGraph",
#             "-framework", "Foundation",
#         ],
#         "//conditions:default": [],
#     }),
# )

# Place the plugin into the expected wheel path so TensorFlow auto-loads it.
genrule(
    name = "mps_plugin_wheel_payload",
    srcs = [":libtensorflow_mps_plugin.dylib"],
    outs = ["tensorflow-plugins/libtensorflow_mps_plugin.dylib"],
    cmd = "cp $(location :libtensorflow_mps_plugin.dylib) $@",
)

py_test(
    name = "mps_device_test",
    size = "medium",
    srcs = ["mps_device_test.py"],
    tags = [
        "manual",  # Run manually on macOS with Metal support
        "no_oss",
        "requires_gpu_apple",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/platform:test",
    ],
)

# Extended ops tests
py_test(
    name = "ops_test",
    size = "medium",
    srcs = ["ops_test.py"],
    tags = [
        "manual",
        "no_oss",
        "requires_gpu_apple",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/platform:test",
    ],
)

# Performance benchmarks
py_binary(
    name = "benchmark_ops",
    srcs = ["benchmark_ops.py"],
    tags = [
        "manual",
        "no_oss",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

# Smoke test (can also be run as py_test)
py_test(
    name = "smoke_test",
    size = "small",
    srcs = ["//ci/smoke:mps_ops_smoke.py"],
    main = "mps_ops_smoke.py",
    tags = [
        "manual",
        "no_oss",
        "requires_gpu_apple",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
