using System;

namespace NativeRender
{
    public enum NativeShaderType : uint
    {
        None = 0,
        Compute = 5
    }

    public struct NativeShaderDesc
    {
        public NativeShaderType ShaderType;
        public string DebugName;
        public string EntryName;
        public int HlslExtensionsUAV;

        public static NativeShaderDesc Compute(string debugName, string entryName = "main")
            => new NativeShaderDesc
            {
                ShaderType = NativeShaderType.Compute,
                DebugName = debugName,
                EntryName = string.IsNullOrEmpty(entryName) ? "main" : entryName,
                HlslExtensionsUAV = -1
            };
    }

    /// <summary>
    /// NVRHI-style shader handle: wraps native shader metadata plus compiled DXIL.
    /// Pipelines reference this handle through their PipelineDesc.
    /// </summary>
    public sealed class NativeShaderHandle : IDisposable
    {
        private ulong _handle;

        public ulong Handle => _handle;
        public bool IsValid => _handle != 0;
        public NativeShaderDesc Desc { get; }

        public NativeShaderHandle(NativeShaderDesc desc, byte[] dxilBytes)
        {
            if (dxilBytes == null || dxilBytes.Length == 0)
                throw new ArgumentException("Shader bytecode is empty.", nameof(dxilBytes));

            Desc = desc;
            var nativeDesc = new NativeRenderPlugin.NR_ShaderDesc
            {
                shaderType = (uint)desc.ShaderType,
                debugName = desc.DebugName,
                entryName = string.IsNullOrEmpty(desc.EntryName) ? "main" : desc.EntryName,
                hlslExtensionsUAV = desc.HlslExtensionsUAV
            };

            _handle = NativeRenderPlugin.NR_CreateShader(
                ref nativeDesc, dxilBytes, (uint)dxilBytes.Length);
            if (_handle == 0)
                throw new InvalidOperationException($"NR_CreateShader returned 0 for: {desc.DebugName}");
        }

        public static NativeShaderHandle FromComputeShader(NativeComputeShader shader)
        {
            if (shader == null)
                throw new ArgumentNullException(nameof(shader));

            byte[] dxil = shader.GetOrCompileDxil();
            if (dxil == null || dxil.Length == 0)
                throw new InvalidOperationException(
                    $"[NativeShaderHandle] Shader compilation failed for: {shader.GetHlslPath()}");

            return new NativeShaderHandle(
                NativeShaderDesc.Compute(shader.name, shader.EntryPoint),
                dxil);
        }

        public void Dispose()
        {
            if (_handle != 0)
            {
                NativeRenderPlugin.NR_DestroyShader(_handle);
                _handle = 0;
            }
            GC.SuppressFinalize(this);
        }

        ~NativeShaderHandle()
        {
            Dispose();
        }
    }
}
