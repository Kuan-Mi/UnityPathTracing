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
                    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/dxc-src/bin/x64/dxcompiler.dll"
                    "${UNITY_PLUGINS_DIR}/dxcompiler.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/dxc-src/bin/x64/dxil.dll"
                    "${UNITY_PLUGINS_DIR}/dxil.dll"
                COMMENT "Copying dxcompiler.dll and dxil.dll to Unity Plugins"
            )
            # nvngx_dlss.dll / nvngx_dlssd.dll  (rel in Release, dev in Debug)
            add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/ngx-src/lib/Windows_x86_64/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlss.dll"
                    "${UNITY_PLUGINS_DIR}/nvngx_dlss.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/ngx-src/lib/Windows_x86_64/$<IF:$<CONFIG:Debug>,dev,rel>/nvngx_dlssd.dll"
                    "${UNITY_PLUGINS_DIR}/nvngx_dlssd.dll"
                COMMENT "Copying nvngx_dlss.dll and nvngx_dlssd.dll to Unity Plugins"
            )
        endif()
    endif()
endmacro()

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
    # NRD Integration helper (NRDIntegration.h, NRDIntegration.hpp)
    "${CMAKE_CURRENT_SOURCE_DIR}/External/NRD/Integration"
    # Agility SDK headers MUST come before Windows SDK so its d3d12.h overrides the older system one
    "${CMAKE_BINARY_DIR}/_deps/agilitysdk-src/build/native/include"
    "${CMAKE_BINARY_DIR}/_deps/agilitysdk-src/build/native/include/d3dx12"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

# Suppress C4819: treat source and execution character set as UTF-8
target_compile_options(Denoiser PRIVATE /utf-8)

# NRD and NRI CMake targets carry their public include dirs transitively
target_link_libraries(Denoiser PRIVATE NRD NRI)

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
    "${CMAKE_BINARY_DIR}/_deps/agilitysdk-src/build/native/include"
    "${CMAKE_BINARY_DIR}/_deps/agilitysdk-src/build/native/include/d3dx12"
    "${WIN_SDK_INC}/um"
    "${WIN_SDK_INC}/shared"
)

target_compile_options(PrepareLight PRIVATE /utf-8)

# NRI carries its public include dirs transitively
target_link_libraries(PrepareLight PRIVATE NRI)

setup_unity_plugin(PrepareLight)
