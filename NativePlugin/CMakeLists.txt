cmake_minimum_required(VERSION 3.20)
project(NativeRenderPlugin)

set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------------------------
# Helper: download and unzip a versioned package (uses a .ver sentinel file
# so re-configuration is a no-op when the correct version is already present)
# ---------------------------------------------------------------------------
function(CheckAndDownloadPackage NAME VERSION LOCAL_PATH URL ARCHIVE_TYPE)
    if(NOT EXISTS "${LOCAL_PATH}/${VERSION}.ver")
        if(EXISTS "${LOCAL_PATH}")
            message(STATUS "Deleting old ${NAME}")
            file(REMOVE_RECURSE "${LOCAL_PATH}")
        endif()
        set(_ARCHIVE "${LOCAL_PATH}.${ARCHIVE_TYPE}")
        message(STATUS "Downloading ${NAME} ${VERSION} from ${URL}")
        file(DOWNLOAD "${URL}" "${_ARCHIVE}" SHOW_PROGRESS STATUS _DL_STATUS)
        list(GET _DL_STATUS 0 _DL_ERR)
        if(NOT _DL_ERR EQUAL 0)
            list(GET _DL_STATUS 1 _DL_MSG)
            file(REMOVE "${_ARCHIVE}")
            message(FATAL_ERROR "Failed to download ${NAME}: ${_DL_MSG}")
        endif()
        message(STATUS "Extracting ${NAME} to ${LOCAL_PATH}")
        file(ARCHIVE_EXTRACT INPUT "${_ARCHIVE}" DESTINATION "${LOCAL_PATH}")
        file(REMOVE "${_ARCHIVE}")
        file(WRITE "${LOCAL_PATH}/${VERSION}.ver")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# omm-lib – auto-downloaded pre-built SDK (CPU baker only, no GPU/NVRHI)
# Source: https://github.com/NVIDIA-RTX/OMM/releases
# ---------------------------------------------------------------------------
CheckAndDownloadPackage(
    "OMM SDK" "v1.9.1"
    "${CMAKE_CURRENT_SOURCE_DIR}/External/omm"
    "https://github.com/NVIDIA-RTX/OMM/releases/download/v1.9.1/Opacity.Micro-Map.SDK.Windows.zip"
    "zip"
)
set(OMM_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/omm")

add_library(omm-lib SHARED IMPORTED GLOBAL)
set_target_properties(omm-lib PROPERTIES
    IMPORTED_IMPLIB   "${OMM_INSTALL_DIR}/lib/omm-lib.lib"
    IMPORTED_LOCATION "${OMM_INSTALL_DIR}/bin/omm-lib.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${OMM_INSTALL_DIR}/include"
)

# ---------------------------------------------------------------------------
# OMMBakerPlugin �?standalone CPU baker DLL (Editor-only, no D3D12)
# ---------------------------------------------------------------------------
add_library(OMMBakerPlugin SHARED
    OMMBakerPlugin/src/OMMBakerPlugin.cpp
)

target_include_directories(OMMBakerPlugin PRIVATE
    "${OMM_INSTALL_DIR}/include"
)

target_compile_options(OMMBakerPlugin PRIVATE /utf-8)

target_link_libraries(OMMBakerPlugin PRIVATE omm-lib)

# ---------------------------------------------------------------------------
# Find Windows SDK paths (for DXC and D3D12 libs)
# ---------------------------------------------------------------------------

# Detect Windows SDK install path from registry
execute_process(
    COMMAND reg query "HKLM\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots" /v KitsRoot10
    OUTPUT_VARIABLE WIN_SDK_REG OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "C:[^\r\n]+" WIN_SDK_PATH "${WIN_SDK_REG}")
if(NOT WIN_SDK_PATH)
    set(WIN_SDK_PATH "C:/Program Files (x86)/Windows Kits/10")
endif()

# Find latest SDK version
file(GLOB WIN_SDK_VERSIONS LIST_DIRECTORIES true "${WIN_SDK_PATH}/Include/10.*")
if(WIN_SDK_VERSIONS)
    list(SORT WIN_SDK_VERSIONS ORDER DESCENDING)
    list(GET WIN_SDK_VERSIONS 0 WIN_SDK_VERSION_DIR)
    get_filename_component(WIN_SDK_VERSION "${WIN_SDK_VERSION_DIR}" NAME)
else()
    set(WIN_SDK_VERSION "10.0.26100.0")
endif()

message(STATUS "Windows SDK: ${WIN_SDK_PATH} version ${WIN_SDK_VERSION}")

set(WIN_SDK_BIN "${WIN_SDK_PATH}/bin/${WIN_SDK_VERSION}/x64")
set(WIN_SDK_INC "${WIN_SDK_PATH}/Include/${WIN_SDK_VERSION}")
set(WIN_SDK_LIB "${WIN_SDK_PATH}/Lib/${WIN_SDK_VERSION}/um/x64")

# ---------------------------------------------------------------------------
# DXC – auto-downloaded from NuGet (SM 6.9 capable, v1.9.2602.17)
# NuGet zip layout: build/native/{bin,inc,lib}/x64
# ---------------------------------------------------------------------------
CheckAndDownloadPackage(
    "DXC" "v1.9.2602.17"
    "${CMAKE_CURRENT_SOURCE_DIR}/External/dxc"
    "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DXC/1.9.2602.17"
    "zip"
)
set(DXC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/External/dxc")
set(DXC_INC   "${DXC_DIR}/build/native/inc")
set(DXC_LIB   "${DXC_DIR}/build/native/lib/x64")
set(DXC_BIN   "${DXC_DIR}/build/native/bin/x64")

# ---------------------------------------------------------------------------
# Unity plugin API headers
# Must be copied from: <UnityInstall>/Editor/Data/PluginAPI/
# ---------------------------------------------------------------------------
set(UNITY_PLUGIN_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shared/include/unity")
if(NOT EXISTS "${UNITY_PLUGIN_API_DIR}/IUnityInterface.h")
    message(WARNING
        "Unity PluginAPI headers not found at ${UNITY_PLUGIN_API_DIR}.\n"
        "Copy the following files from <UnityInstall>/Editor/Data/PluginAPI/ :\n"
        "  IUnityInterface.h\n"
        "  IUnityGraphics.h\n"
        "  IUnityGraphicsD3D12.h\n"
        "  IUnityRenderingExtensions.h")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
add_library(NativeRenderPlugin SHARED
    NativeRenderPlugin/src/Plugin.cpp
    NativeRenderPlugin/src/RayTraceShader.cpp
    NativeRenderPlugin/src/ComputeShader.cpp
    NativeRenderPlugin/src/AccelerationStructure.cpp
    NativeRenderPlugin/src/DescriptorHeapAllocator.cpp
    NativeRenderPlugin/src/BindlessTexture.cpp
    NativeRenderPlugin/src/BindlessBuffer.cpp
)

target_include_directories(NativeRenderPlugin PRIVATE
    NativeRenderPlugin/include
    shared/include
    "${UNITY_PLUGIN_API_DIR}"
    "${DXC_INC}"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

# Suppress C4819: treat source and execution character set as UTF-8
target_compile_options(NativeRenderPlugin PRIVATE /utf-8)

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------
target_link_directories(NativeRenderPlugin PRIVATE
    "${WIN_SDK_LIB}"
    "${DXC_LIB}"
)

target_link_libraries(NativeRenderPlugin PRIVATE
    d3d12.lib
    dxgi.lib
    dxguid.lib
    dxcompiler.lib  # from dxc_2026_02_20, supports SM 6.9
)

# ---------------------------------------------------------------------------
# ShaderCompilerPlugin – standalone HLSL-to-DXIL compiler DLL.
# Uses IUnityLog for in-Editor log output; no other Unity dependency.
# ---------------------------------------------------------------------------
add_library(ShaderCompilerPlugin SHARED
    ShaderCompilerPlugin/src/ShaderCompilerPlugin.cpp
)

target_include_directories(ShaderCompilerPlugin PRIVATE
    ShaderCompilerPlugin/include
    shared/include
    "${UNITY_PLUGIN_API_DIR}"
    "${DXC_INC}"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

target_compile_options(ShaderCompilerPlugin PRIVATE /utf-8)

target_link_directories(ShaderCompilerPlugin PRIVATE
    "${WIN_SDK_LIB}"
    "${DXC_LIB}"
)

target_link_libraries(ShaderCompilerPlugin PRIVATE
    dxcompiler.lib
)

# ---------------------------------------------------------------------------
# Output directory �?skip Unity copy when NR_SKIP_UNITY_COPY=ON (CI / AI checks)
# ---------------------------------------------------------------------------
option(NR_SKIP_UNITY_COPY "Build only; do not copy output to the Unity project" OFF)

if(NR_SKIP_UNITY_COPY)
    message(STATUS "NR_SKIP_UNITY_COPY=ON: output stays in build/Release, Unity folder untouched.")
else()
    set_target_properties(NativeRenderPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        PDB_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        PDB_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
    )

    # Copy dxcompiler.dll and dxil.dll from dxc_2026_02_20 only
    set(DXCOMPILER_DLL "${DXC_BIN}/dxcompiler.dll")
    set(DXIL_DLL       "${DXC_BIN}/dxil.dll")

    if(NOT EXISTS "${DXCOMPILER_DLL}")
        message(FATAL_ERROR "dxcompiler.dll not found at ${DXCOMPILER_DLL}")
    endif()
    if(NOT EXISTS "${DXIL_DLL}")
        message(FATAL_ERROR "dxil.dll not found at ${DXIL_DLL}")
    endif()

    message(STATUS "DXCOMPILER_DLL: ${DXCOMPILER_DLL}")
    message(STATUS "DXIL_DLL:       ${DXIL_DLL}")

    if(DXCOMPILER_DLL)
        add_custom_command(TARGET NativeRenderPlugin POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DXCOMPILER_DLL}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64/dxcompiler.dll"
        )
        message(STATUS "Will copy ${DXCOMPILER_DLL} to Unity Plugins folder")
    endif()

    if(DXIL_DLL)
        add_custom_command(TARGET NativeRenderPlugin POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${DXIL_DLL}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64/dxil.dll"
        )
        message(STATUS "Will copy ${DXIL_DLL} to Unity Plugins folder")
    endif()

    # ShaderCompilerPlugin output dir
    set_target_properties(ShaderCompilerPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
    )

    # OMMBakerPlugin output dir + copy itself and omm-lib.dll
    set_target_properties(OMMBakerPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64"
    )
    add_custom_command(TARGET OMMBakerPlugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OMM_INSTALL_DIR}/bin/omm-lib.dll"
            "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64/omm-lib.dll"
        COMMENT "Copying omm-lib.dll to Unity Plugins"
    )
endif()
