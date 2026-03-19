 

using System;

namespace Nri
{
    [Flags]
    public enum AccessBits : uint
    {
        NONE = 0, // Mapped to "COMMON" (aka "GENERAL" access), if AgilitySDK is not available, leading to potential discrepancies with VK

        // Buffer                                // Access  Compatible "StageBits" (including ALL)
        INDEX_BUFFER = (1 << 0), // R   INDEX_INPUT
        VERTEX_BUFFER = (1 << 1), // R   VERTEX_SHADER
        CONSTANT_BUFFER = (1 << 2), // R   GRAPHICS_SHADERS, COMPUTE_SHADER, RAY_TRACING_SHADERS
        ARGUMENT_BUFFER = (1 << 3), // R   INDIRECT
        SCRATCH_BUFFER = (1 << 4), // RW  ACCELERATION_STRUCTURE, MICROMAP

        // Attachment
        COLOR_ATTACHMENT = (1 << 5), // RW  COLOR_ATTACHMENT
        SHADING_RATE_ATTACHMENT = (1 << 6), // R   FRAGMENT_SHADER
        DEPTH_STENCIL_ATTACHMENT_READ = (1 << 7), // R   DEPTH_STENCIL_ATTACHMENT
        DEPTH_STENCIL_ATTACHMENT_WRITE = (1 << 8), //  W  DEPTH_STENCIL_ATTACHMENT

        // Acceleration structure
        ACCELERATION_STRUCTURE_READ = (1 << 9), // R   COMPUTE_SHADER, RAY_TRACING_SHADERS, ACCELERATION_STRUCTURE
        ACCELERATION_STRUCTURE_WRITE = (1 << 10), //  W  ACCELERATION_STRUCTURE

        // Micromap
        MICROMAP_READ = (1 << 11), // R   MICROMAP, ACCELERATION_STRUCTURE
        MICROMAP_WRITE = (1 << 12), //  W  MICROMAP

        // Shader resource
        SHADER_RESOURCE = (1 << 13), // R   GRAPHICS_SHADERS, COMPUTE_SHADER, RAY_TRACING_SHADERS
        SHADER_RESOURCE_STORAGE = (1 << 14), // RW  GRAPHICS_SHADERS, COMPUTE_SHADER, RAY_TRACING_SHADERS, CLEAR_STORAGE + shaders
        SHADER_BINDING_TABLE = (1 << 15), // R   RAY_TRACING_SHADERS

        // Copy
        COPY_SOURCE = (1 << 16), // R   COPY
        COPY_DESTINATION = (1 << 17), //  W  COPY

        // Resolve
        RESOLVE_SOURCE = (1 << 18), // R   RESOLVE
        RESOLVE_DESTINATION = (1 << 19), //  W  RESOLVE

        // Clear storage
        CLEAR_STORAGE = (1 << 20) //  W  CLEAR_STORAGE
    }

    public enum Layout : uint
    {
        // Compatible "AccessBits":
        // Special
        UNDEFINED, // https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html#d3d12_barrier_layout_undefined
        GENERAL, // ~ALL access, but potentially not optimal (required for "SharingMode::SIMULTANEOUS")
        PRESENT, // NONE (use "after.stages = StageBits::NONE")

        // Access specific
        COLOR_ATTACHMENT, // COLOR_ATTACHMENT
        SHADING_RATE_ATTACHMENT, // SHADING_RATE_ATTACHMENT
        DEPTH_STENCIL_ATTACHMENT, // DEPTH_STENCIL_ATTACHMENT_WRITE
        DEPTH_STENCIL_READONLY, // DEPTH_STENCIL_ATTACHMENT_READ, SHADER_RESOURCE
        SHADER_RESOURCE, // SHADER_RESOURCE
        SHADER_RESOURCE_STORAGE, // SHADER_RESOURCE_STORAGE
        COPY_SOURCE, // COPY_SOURCE
        COPY_DESTINATION, // COPY_DESTINATION
        RESOLVE_SOURCE, // RESOLVE_SOURCE
        RESOLVE_DESTINATION // RESOLVE_DESTINATION
    }
    
}
 