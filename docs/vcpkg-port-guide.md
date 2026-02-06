# vcpkg Port Submission Guide

To submit srukf to the vcpkg registry:

1. Fork https://github.com/microsoft/vcpkg
2. Create `ports/srukf/portfile.cmake`:

```cmake
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO disruptek/srukf
    REF v1.0.0
    SHA512 <compute with `vcpkg_from_github` or sha512sum>
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DSRUKF_BUILD_TESTS=OFF
        -DSRUKF_BUILD_EXAMPLES=OFF
        -DSRUKF_BUILD_BENCHMARKS=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/srukf)

vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
```

3. Copy `vcpkg.json` to `ports/srukf/vcpkg.json`
4. Test: `vcpkg install srukf --overlay-ports=ports/srukf`
5. Submit PR to microsoft/vcpkg with both files
