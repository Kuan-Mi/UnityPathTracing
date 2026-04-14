cmake_minimum_required(VERSION 3.22)
project(RenderingPlugin)

set(CMAKE_CXX_STANDARD 17)

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

# ---------------------------------------------------------------------------
# NRI – D3D12 only, disable unnecessary backends and SDKs
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

add_subdirectory(External/NRI NRI_build)

# ---------------------------------------------------------------------------
# NRD – NRI target already present, no FetchContent needed
# ---------------------------------------------------------------------------
option(NRD_NRI "" OFF)           # NRI target already exists above
option(NRD_STATIC_LIBRARY "" OFF)
option(NRD_EMBEDS_SPIRV_SHADERS "" OFF)
option(NRD_EMBEDS_DXBC_SHADERS "" OFF)

add_subdirectory(External/NRD NRD_build)

# ---------------------------------------------------------------------------
# RenderingPlugin DLL
# ---------------------------------------------------------------------------
add_library(RenderingPlugin SHARED
    RenderingPlugin/DLRRInstance.cpp
    RenderingPlugin/NrdInstance.cpp
    RenderingPlugin/RenderingPlugin.cpp
    RenderingPlugin/RenderSystem.cpp
)

target_include_directories(RenderingPlugin PRIVATE
    RenderingPlugin/Unity
    # NRD Integration helper (NRDIntegration.h, NRDIntegration.hpp)
    "${CMAKE_CURRENT_SOURCE_DIR}/External/NRD/Integration"
    # Agility SDK headers MUST come before Windows SDK so its d3d12.h overrides the older system one
    "${CMAKE_BINARY_DIR}/_deps/agilitysdk-src/build/native/include"
    "${CMAKE_BINARY_DIR}/_deps/agilitysdk-src/build/native/include/d3dx12"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

# Suppress C4819: treat source and execution character set as UTF-8
target_compile_options(RenderingPlugin PRIVATE /utf-8)

# NRD and NRI CMake targets carry their public include dirs transitively
target_link_libraries(RenderingPlugin PRIVATE NRD NRI)

# ---------------------------------------------------------------------------
# Output – copy DLLs to Unity Plugins folder unless skipped (CI)
# ---------------------------------------------------------------------------
option(NR_SKIP_UNITY_COPY "Build only; do not copy output to the Unity project" OFF)

set(UNITY_PLUGINS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../UnityProject/Assets/Plugins/x86_64")

if(NR_SKIP_UNITY_COPY)
    message(STATUS "NR_SKIP_UNITY_COPY=ON: output stays in build dir, Unity folder untouched.")
else()
    set_target_properties(RenderingPlugin PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY         "${UNITY_PLUGINS_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${UNITY_PLUGINS_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${UNITY_PLUGINS_DIR}"
        PDB_OUTPUT_DIRECTORY             "${UNITY_PLUGINS_DIR}"
        PDB_OUTPUT_DIRECTORY_RELEASE     "${UNITY_PLUGINS_DIR}"
        PDB_OUTPUT_DIRECTORY_DEBUG       "${UNITY_PLUGINS_DIR}"
    )

    # Copy NRD.dll and NRI.dll after build
    add_custom_command(TARGET RenderingPlugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${UNITY_PLUGINS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:NRD>
            "${UNITY_PLUGINS_DIR}/$<TARGET_FILE_NAME:NRD>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:NRI>
            "${UNITY_PLUGINS_DIR}/$<TARGET_FILE_NAME:NRI>"
        # Copy PDB files in Debug config (cmake -E true is a no-op in Release)
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
            "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:RenderingPlugin>>"
            "$<$<CONFIG:Debug>:${UNITY_PLUGINS_DIR}/$<TARGET_PDB_FILE_NAME:RenderingPlugin>>"
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
            "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:NRD>>"
            "$<$<CONFIG:Debug>:${UNITY_PLUGINS_DIR}/$<TARGET_PDB_FILE_NAME:NRD>>"
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<CONFIG:Debug>,copy_if_different,true>
            "$<$<CONFIG:Debug>:$<TARGET_PDB_FILE:NRI>>"
            "$<$<CONFIG:Debug>:${UNITY_PLUGINS_DIR}/$<TARGET_PDB_FILE_NAME:NRI>>"
        COMMENT "Copying RenderingPlugin, NRD, NRI DLLs (and PDBs in Debug) to Unity Plugins"
    )
endif()
