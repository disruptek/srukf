from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy
import os


class SrukfConan(ConanFile):
    name = "srukf"
    version = "1.0.0"
    license = "MIT"
    author = "disruptek"
    url = "https://github.com/disruptek/srukf"
    description = (
        "Square-Root Unscented Kalman Filter - "
        "numerically stable state estimation for nonlinear systems"
    )
    topics = ("kalman-filter", "state-estimation", "sensor-fusion", "robotics")
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "shared": [True, False],
        "single_precision": [True, False],
        "build_tests": [True, False],
        "build_examples": [True, False],
    }

    default_options = {
        "shared": False,
        "single_precision": False,
        "build_tests": False,
        "build_examples": False,
    }

    exports_sources = (
        "CMakeLists.txt",
        "srukf.pc.in",
        "*.c",
        "*.h",
        "cmake/*",
        "tests/*",
        "examples/*",
        "benchmark/*",
    )

    def system_requirements(self):
        if self.settings.os == "Linux":
            self.output.info(
                "Requires: libopenblas-dev liblapacke-dev (apt) "
                "or openblas lapack-reference (Gentoo) "
                "or openblas-devel lapack-devel (yum)"
            )
        elif self.settings.os == "Macos":
            self.output.info("Requires: openblas (brew)")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["SRUKF_BUILD_SHARED"] = self.options.shared
        tc.variables["SRUKF_BUILD_STATIC"] = not self.options.shared
        tc.variables["SRUKF_SINGLE_PRECISION"] = self.options.single_precision
        tc.variables["SRUKF_BUILD_TESTS"] = self.options.build_tests
        tc.variables["SRUKF_BUILD_EXAMPLES"] = self.options.build_examples
        tc.variables["SRUKF_BUILD_BENCHMARKS"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if self.options.build_tests:
            cmake.test()

    def package(self):
        copy(self, "LICENSE", src=self.source_folder,
             dst=os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["srukf"]
        self.cpp_info.system_libs = ["m"]

        self.cpp_info.components["libsrukf"].libs = ["srukf"]
        self.cpp_info.components["libsrukf"].system_libs = ["m"]
