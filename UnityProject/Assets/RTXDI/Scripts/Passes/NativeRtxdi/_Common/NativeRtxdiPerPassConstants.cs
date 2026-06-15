using System.Runtime.InteropServices;

namespace PathTracing
{
    /// <summary>
    /// CPU-side mirror of <c>PerPassConstants</c> defined in
    /// <c>UnityProject/Assets/RTXDI/Shaders/SharedShaderInclude/SharedShaderInclude/ShaderParameters.h</c>.
    /// Bound to the <c>g_PerPassConstants</c> CBV (b1) of every RTXDI lighting pass.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct NativeRtxdiPerPassConstants
    {
        public int rayCountBufferIndex;
    }
}
