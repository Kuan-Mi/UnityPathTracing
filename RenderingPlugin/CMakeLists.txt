cmake_minimum_required(VERSION 3.22)
project(RenderingPlugin)

set(CMAKE_CXX_STANDARD 20)

# ---------------------------------------------------------------------------
# Find Windows SDK paths (for D3D12 headers)
# ---------------------------------------------------------------------------
execute_process(
    COMMAND reg query "HKLM\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots" /v KitsRoot10
    OUTPUT_VARIABLE WIN_SDK_REG OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "[C-Z]:[^\r\n]+" WIN_SDK_PATH "${WIN_SDK_REG}")
if(NOT WIN_SDK_PATH)
    set(WIN_SDK_PATH "C:/Program Files (x86)/Windows Kits/10")
endif()

file(GLOB WIN_SDK_VERSIONS LIST_DIRECTORIES true "${WIN_SDK_PATH}/Include/10.*")
if(WIN_SDK_VERSIONS)
    list(SORT WIN_SDK_VERSIONS ORDER DESCENDING)
    list(GET WIN_SDK_VERSIONS 0 WIN_SDK_VERSION_DIR)
    get_filename_component(WIN_SDK_VERSION "${WIN_SDK_VERSION_DIR}" NAME)
else()
    set(WIN_SDK_VERSION "10.0.26100.0")
endif()

message(STATUS "Windows SDK: ${WIN_SDK_PATH} version ${WIN_SDK_VERSION}")
set(WIN_SDK_INC "${WIN_SDK_PATH}/Include/${WIN_SDK_VERSION}")
set(WIN_SDK_LIB "${WIN_SDK_PATH}/Lib/${WIN_SDK_VERSION}/um/x64")

# ---------------------------------------------------------------------------
# Keep FetchContent downloads outside _Build so they survive cache resets
# ---------------------------------------------------------------------------
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_deps" CACHE PATH "FetchContent download cache")

# ---------------------------------------------------------------------------
# NRI – D3D12 only, shared by both plugin targets
# ---------------------------------------------------------------------------
option(NRI_STATIC_LIBRARY "" OFF)
option(NRI_ENABLE_VK_SUPPORT "" OFF)
option(NRI_ENABLE_D3D11_SUPPORT "" OFF)
option(NRI_ENABLE_NONE_SUPPORT "" OFF)
option(NRI_ENABLE_VALIDATION_SUPPORT "" OFF)
option(NRI_ENABLE_NVTX_SUPPORT "" OFF)
option(NRI_ENABLE_AMDAGS "" OFF)
option(NRI_ENABLE_NVAPI "" OFF)
option(NRI_ENABLE_NIS_SDK "" OFF)
option(NRI_ENABLE_IMGUI_EXTENSION "" OFF)
option(NRI_ENABLE_NGX_SDK "" ON)

add_subdirectory(External/NRI "${CMAKE_CURRENT_SOURCE_DIR}/_ExternalBuild/NRI_build")

# ---------------------------------------------------------------------------
# NRD – used by Denoiser only; NRI target already present above
# ---------------------------------------------------------------------------
option(NRD_NRI "" OFF)
option(NRD_STATIC_LIBRARY "" OFF)
option(NRD_EMBEDS_SPIRV_SHADERS "" OFF)
option(NRD_EMBEDS_DXBC_SHADERS "" OFF)

add_subdirectory(External/NRD "${CMAKE_CURRENT_SOURCE_DIR}/_ExternalBuild/NRD_build")

# ---------------------------------------------------------------------------
# Output / copy option
# ---------------------------------------------------------------------------
option(NR_SKIP_UNITY_COPY "Build only; do not copy output to the Unity project" OFF)

set(UNITY_PLUGINS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Assets/Plugins/x86_64")

# Helper macro: configure output dir and post-build DLL copy for a plugin target
# Usage: setup_unity_plugin(<target> [COPY_NRD])
macro(setup_unity_plugin TARGET)
    cmake_parse_arguments(_P "COPY_NRD" "" "" ${ARGN})

    if(NR_SKIP_UNITY_COPY)
        message(STATUS "NR_SKIP_UNITY_COPY=ON: output for ${TARGET} stays in build dir.")
    else()
        set_target_properties(${TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY         "${UNITY_PLUGINS_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${UNITY_PLUGINS_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${UNITY_PLUGINS_DIR}"
            PDB_OUTPUT_DIRECTORY             "${UNITY_PLUGINS_DIR}"
            PDB_OUTPUT_DIRECTORY_RELEASE     "${UNITY_PLUGINS_DIR}"
            PDB_OUTPUT_DIRECTORY_DEBUG       "${UNITY_PLUGINS_DIR}"
        )

        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${UNITY_PLUGINS_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:NRI>
                "${UNITY_PLUGINS_DIR}/$<TARGET_FILE_NAME:NRI>"
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
                "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:${TARGET}>>"
                "$<$<CONFIG:Debug>:${UNITY_PLUGINS_DIR}/$<TARGET_PDB_FILE_NAME:${TARGET}>>"
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
                "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:NRI>>"
                "$<$<CONFIG:Debug>:${UNITY_PLUGINS_DIR}/$<TARGET_PDB_FILE_NAME:NRI>>"
            COMMENT "Copying ${TARGET} and NRI DLLs (and PDBs in Debug) to Unity Plugins"
        )

        if(_P_COPY_NRD)
            # NRD.dll + PDB
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:NRD>
                    "${UNITY_PLUGINS_DIR}/$<TARGET_FILE_NAME:NRD>"
                COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
                    "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:NRD>>"
                    "$<$<CONFIG:Debug>:${UNITY_PLUGINS_DIR}/$<TARGET_PDB_FILE_NAME:NRD>>"
                COMMENT "Copying NRD DLL (and PDB in Debug) to Unity Plugins"
            )
            # dxcompiler.dll + dxil.dll
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FETCHCONTENT_BASE_DIR}/dxc-src/bin/x64/dxcompiler.dll"
                    "${UNITY_PLUGINS_DIR}/dxcompiler.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FETCHCONTENT_BASE_DIR}/dxc-src/bin/x64/dxil.dll"
                    "${UNITY_PLUGINS_DIR}/dxil.dll"
                COMMENT "Copying dxcompiler.dll and dxil.dll to Unity Plugins"
            )
            # nvngx_dlss.dll / nvngx_dlssd.dll  (rel in Release, dev in Debug)
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FETCHCONTENT_BASE_DIR}/ngx-src/lib/Windows_x86_64/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlss.dll"
                    "${UNITY_PLUGINS_DIR}/nvngx_dlss.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FETCHCONTENT_BASE_DIR}/ngx-src/lib/Windows_x86_64/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlssd.dll"
                    "${UNITY_PLUGINS_DIR}/nvngx_dlssd.dll"
                COMMENT "Copying nvngx_dlss.dll and nvngx_dlssd.dll to Unity Plugins"
            )
        endif()
    endif()
endmacro()

# ---------------------------------------------------------------------------
# D3D12HeapHook – shared DLL for SetDescriptorHeaps vtable hook
# Both NativeRenderPlugin and Denoiser link against this so they share
# a single set of globals (one hook, one cache, one TLS counter).
# ---------------------------------------------------------------------------
add_library(D3D12HeapHook SHARED
    D3D12HeapHook/D3D12HeapHook.cpp
)

target_include_directories(D3D12HeapHook PRIVATE
    D3D12HeapHook
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

target_compile_definitions(D3D12HeapHook PRIVATE D3D12HEAPHOOK_EXPORTS)
target_compile_options(D3D12HeapHook PRIVATE /utf-8)

target_link_directories(D3D12HeapHook PRIVATE "${WIN_SDK_LIB}")
target_link_libraries(D3D12HeapHook PRIVATE d3d12.lib)

# ---------------------------------------------------------------------------
# Denoiser DLL  (NRD + NRI denoising plugin)
# ---------------------------------------------------------------------------
add_library(Denoiser SHARED
    Denoiser/DLRRInstance.cpp
    Denoiser/NrdInstance.cpp
    Denoiser/RenderingPlugin.cpp
    Denoiser/RenderSystem.cpp
)

target_include_directories(Denoiser PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    D3D12HeapHook
    # NRD Integration helper (NRDIntegration.h, NRDIntegration.hpp)
    "${CMAKE_CURRENT_SOURCE_DIR}/External/NRD/Integration"
    # Agility SDK headers MUST come before Windows SDK so its d3d12.h overrides the older system one
    "${FETCHCONTENT_BASE_DIR}/agilitysdk-src/build/native/include"
    "${FETCHCONTENT_BASE_DIR}/agilitysdk-src/build/native/include/d3dx12"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

# Suppress C4819: treat source and execution character set as UTF-8
target_compile_options(Denoiser PRIVATE /utf-8)

# NRD and NRI CMake targets carry their public include dirs transitively
target_link_libraries(Denoiser PRIVATE NRD NRI D3D12HeapHook)

setup_unity_plugin(Denoiser COPY_NRD)

# ---------------------------------------------------------------------------
# PrepareLight DLL  (NRI bindless light-preparation plugin)
# ---------------------------------------------------------------------------
add_library(PrepareLight SHARED
    PrepareLight/BindlessInstance.cpp
    PrepareLight/RenderingPlugin.cpp
    PrepareLight/RenderSystem.cpp
)

target_include_directories(PrepareLight PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    "${FETCHCONTENT_BASE_DIR}/agilitysdk-src/build/native/include"
    "${FETCHCONTENT_BASE_DIR}/agilitysdk-src/build/native/include/d3dx12"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

target_compile_options(PrepareLight PRIVATE /utf-8)

# NRI carries its public include dirs transitively
target_link_libraries(PrepareLight PRIVATE NRI)

setup_unity_plugin(PrepareLight)

# ===========================================================================
# NativeRenderPlugin, ShaderCompilerPlugin, OMMBakerPlugin
# (formerly in NativePlugin/)
# ===========================================================================

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
# ---------------------------------------------------------------------------
CheckAndDownloadPackage(
    "OMM SDK" "v1.9.1"
    "${FETCHCONTENT_BASE_DIR}/omm-src"
    "https://github.com/NVIDIA-RTX/OMM/releases/download/v1.9.1/Opacity.Micro-Map.SDK.Windows.zip"
    "zip"
)
set(OMM_INSTALL_DIR "${FETCHCONTENT_BASE_DIR}/omm-src")

add_library(omm-lib SHARED IMPORTED GLOBAL)
set_target_properties(omm-lib PROPERTIES
    IMPORTED_IMPLIB   "${OMM_INSTALL_DIR}/lib/omm-lib.lib"
    IMPORTED_LOCATION "${OMM_INSTALL_DIR}/bin/omm-lib.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${OMM_INSTALL_DIR}/include"
)

# ---------------------------------------------------------------------------
# DXC – auto-downloaded from NuGet (SM 6.9 capable, v1.9.2602.17)
# ---------------------------------------------------------------------------
CheckAndDownloadPackage(
    "DXC" "v1.9.2602.17"
    "${FETCHCONTENT_BASE_DIR}/dxc-nuget"
    "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DXC/1.9.2602.17"
    "zip"
)
set(DXC_DIR "${FETCHCONTENT_BASE_DIR}/dxc-nuget")
set(DXC_INC "${DXC_DIR}/build/native/inc")
set(DXC_LIB "${DXC_DIR}/build/native/lib/x64")
set(DXC_BIN "${DXC_DIR}/build/native/bin/x64")

# ---------------------------------------------------------------------------
# Unity plugin API headers (shared with Denoiser/PrepareLight)
# ---------------------------------------------------------------------------
set(UNITY_PLUGIN_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Unity")
if(NOT EXISTS "${UNITY_PLUGIN_API_DIR}/IUnityInterface.h")
    message(WARNING
        "Unity PluginAPI headers not found at ${UNITY_PLUGIN_API_DIR}.\n"
        "Copy the following files from <UnityInstall>/Editor/Data/PluginAPI/ :\n"
        "  IUnityInterface.h\n"
        "  IUnityGraphics.h\n"
        "  IUnityGraphicsD3D12.h\n"
        "  IUnityRenderingExtensions.h")
endif()

set(NATIVE_PLUGINS_UNITY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Packages/top.kuanmi.native-ray-tracing/Plugins/x86_64")

# ---------------------------------------------------------------------------
# OMMBakerPlugin – standalone CPU baker DLL (Editor-only, no D3D12)
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
# NativeRenderPlugin – D3D12 DXR ray-tracing plugin
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
    D3D12HeapHook
    "${UNITY_PLUGIN_API_DIR}"
    "${DXC_INC}"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

target_compile_options(NativeRenderPlugin PRIVATE /utf-8)

target_link_directories(NativeRenderPlugin PRIVATE
    "${WIN_SDK_LIB}"
    "${DXC_LIB}"
)

target_link_libraries(NativeRenderPlugin PRIVATE
    d3d12.lib
    dxgi.lib
    dxguid.lib
    dxcompiler.lib
    D3D12HeapHook
)

# ---------------------------------------------------------------------------
# ShaderCompilerPlugin – standalone HLSL-to-DXIL compiler DLL
# ---------------------------------------------------------------------------
add_library(ShaderCompilerPlugin SHARED
    ShaderCompilerPlugin/src/ShaderCompilerPlugin.cpp
)

target_include_directories(ShaderCompilerPlugin PRIVATE
    ShaderCompilerPlugin/include
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
# Output directories and post-build copies for the three native plugins
# ---------------------------------------------------------------------------
if(NOT NR_SKIP_UNITY_COPY)
    foreach(_TARGET NativeRenderPlugin ShaderCompilerPlugin OMMBakerPlugin)
        set_target_properties(${_TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY         "${NATIVE_PLUGINS_UNITY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${NATIVE_PLUGINS_UNITY_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${NATIVE_PLUGINS_UNITY_DIR}"
            PDB_OUTPUT_DIRECTORY             "${NATIVE_PLUGINS_UNITY_DIR}"
            PDB_OUTPUT_DIRECTORY_RELEASE     "${NATIVE_PLUGINS_UNITY_DIR}"
            PDB_OUTPUT_DIRECTORY_DEBUG       "${NATIVE_PLUGINS_UNITY_DIR}"
        )
    endforeach()

    # D3D12HeapHook.dll (copy to both plugin directories)
    add_custom_command(TARGET D3D12HeapHook POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NATIVE_PLUGINS_UNITY_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:D3D12HeapHook>
            "${NATIVE_PLUGINS_UNITY_DIR}/$<TARGET_FILE_NAME:D3D12HeapHook>"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${UNITY_PLUGINS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:D3D12HeapHook>
            "${UNITY_PLUGINS_DIR}/$<TARGET_FILE_NAME:D3D12HeapHook>"
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
            "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:D3D12HeapHook>>"
            "$<$<CONFIG:Debug>:${NATIVE_PLUGINS_UNITY_DIR}/$<TARGET_PDB_FILE_NAME:D3D12HeapHook>>"
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
            "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:D3D12HeapHook>>"
            "$<$<CONFIG:Debug>:${UNITY_PLUGINS_DIR}/$<TARGET_PDB_FILE_NAME:D3D12HeapHook>>"
        COMMENT "Copying D3D12HeapHook DLL (and PDB in Debug) to Unity plugin directories"
    )

    # dxcompiler.dll + dxil.dll (copy on NativeRenderPlugin build)
    add_custom_command(TARGET NativeRenderPlugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NATIVE_PLUGINS_UNITY_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DXC_BIN}/dxcompiler.dll"
            "${NATIVE_PLUGINS_UNITY_DIR}/dxcompiler.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DXC_BIN}/dxil.dll"
            "${NATIVE_PLUGINS_UNITY_DIR}/dxil.dll"
        COMMENT "Copying dxcompiler.dll and dxil.dll to Unity Packages Plugins"
    )

    # omm-lib.dll (copy on OMMBakerPlugin build)
    add_custom_command(TARGET OMMBakerPlugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NATIVE_PLUGINS_UNITY_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OMM_INSTALL_DIR}/bin/omm-lib.dll"
            "${NATIVE_PLUGINS_UNITY_DIR}/omm-lib.dll"
        COMMENT "Copying omm-lib.dll to Unity Packages Plugins"
    )
endif()
