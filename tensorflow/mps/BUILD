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
