load("//tensorflow:strict.default.bzl", "internal_visibility_allowlist")

package(default_visibility = internal_visibility_allowlist())

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "is_macos",
    constraint_values = [
        "@platforms//os:macos",
    ],
)

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
