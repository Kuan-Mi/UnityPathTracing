using System;
using System.Runtime.InteropServices;

namespace DefaultNamespace
{
    public enum ReSTIRDI_ResamplingMode : uint
    {
        None,
        Temporal,
        Spatial,
        TemporalAndSpatial,
        FusedSpatiotemporal
    };


    public class ReSTIRDIContext
    {
        private const string DllName = "UnityRtxdi";

        // ================= Getters Imports =================
        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern unsafe ReSTIRDIStaticParameters* GetStaticParameters(IntPtr context);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern RTXDI_ReservoirBufferParameters GetReservoirBufferParameters(IntPtr context);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern ReSTIRDI_ResamplingMode GetResamplingMode(IntPtr contextPtr);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern RTXDI_RuntimeParameters GetRuntimeParameters(IntPtr contextPtr);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern ReSTIRDI_BufferIndices GetBufferIndices(IntPtr contextPtr);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern ReSTIRDI_InitialSamplingParameters GetInitialSamplingParameters(IntPtr contextPtr);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern ReSTIRDI_TemporalResamplingParameters GetTemporalResamplingParameters(IntPtr contextPtr);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern ReSTIRDI_SpatialResamplingParameters GetSpatialResamplingParameters(IntPtr contextPtr);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern ReSTIRDI_ShadingParameters GetShadingParameters(IntPtr contextPtr);

        // ================= Setters Imports (New) =================
        // 注意：为了匹配 C++，这里可以直接传值，或者为了性能用 ref。
        // 因为 C++ 导出层是按值接收 (Copy)，如果想优化，C++ 导出层应改写为接收指针，这里用 ref。
        // 下面按照 C++ 导出层为按值传递 (pass-by-value) 编写。

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        private static extern void SetFrameIndex(IntPtr context, uint frameIndex);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        private static extern void SetResamplingMode(IntPtr context, ReSTIRDI_ResamplingMode resamplingMode);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        private static extern void SetInitialSamplingParameters(IntPtr context, ReSTIRDI_InitialSamplingParameters parameters);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        private static extern void SetTemporalResamplingParameters(IntPtr context, ReSTIRDI_TemporalResamplingParameters parameters);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        private static extern void SetSpatialResamplingParameters(IntPtr context, ReSTIRDI_SpatialResamplingParameters parameters);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        private static extern void SetShadingParameters(IntPtr context, ReSTIRDI_ShadingParameters parameters);

        // ================= Lifecycle Imports =================
        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr CreateReSTIRDIContext(int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern void DestroyReSTIRDIContext(IntPtr context);


        IntPtr contextPtr;
        private bool disposedValue;


        public ReSTIRDIContext(int width, int height)
        {
            contextPtr = CreateReSTIRDIContext(width, height);
            if (contextPtr == IntPtr.Zero)
            {
                throw new Exception("Failed to create ReSTIR DI Context.");
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (contextPtr != IntPtr.Zero)
                {
                    DestroyReSTIRDIContext(contextPtr);
                    contextPtr = IntPtr.Zero;
                }
                disposedValue = true;
            }
        }

        ~ReSTIRDIContext()
        {
            Dispose(disposing: false);
        }
        
        
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        
        

        // ================= Public Methods =================

        public RTXDI_ReservoirBufferParameters GetReservoirBufferParameters() => GetReservoirBufferParameters(contextPtr);
        public ReSTIRDI_ResamplingMode GetResamplingMode() => GetResamplingMode(contextPtr);
        public RTXDI_RuntimeParameters GetRuntimeParams() => GetRuntimeParameters(contextPtr);
        public ReSTIRDI_BufferIndices GetBufferIndices() => GetBufferIndices(contextPtr);
        public ReSTIRDI_InitialSamplingParameters GetInitialSamplingParameters() => GetInitialSamplingParameters(contextPtr);
        public ReSTIRDI_TemporalResamplingParameters GetTemporalResamplingParameters() => GetTemporalResamplingParameters(contextPtr);
        public ReSTIRDI_SpatialResamplingParameters GetSpatialResamplingParameters() => GetSpatialResamplingParameters(contextPtr);
        public ReSTIRDI_ShadingParameters GetShadingParameters() => GetShadingParameters(contextPtr);
        
        public unsafe ReSTIRDIStaticParameters* GetStaticParameters() => GetStaticParameters(contextPtr);

        public void SetFrameIndex(uint frameIndex)
        { 
            SetFrameIndex(contextPtr, frameIndex);
        }

        public void SetResamplingMode(ReSTIRDI_ResamplingMode resamplingMode)
        {
            SetResamplingMode(contextPtr, resamplingMode);
        }

        public void SetInitialSamplingParameters(ReSTIRDI_InitialSamplingParameters parameters)
        {
            SetInitialSamplingParameters(contextPtr, parameters);
        }

        public void SetTemporalResamplingParameters(ReSTIRDI_TemporalResamplingParameters parameters)
        {
            SetTemporalResamplingParameters(contextPtr, parameters);
        }

        public void SetSpatialResamplingParameters(ReSTIRDI_SpatialResamplingParameters parameters)
        {
            SetSpatialResamplingParameters(contextPtr, parameters);
        }

        public void SetShadingParameters(ReSTIRDI_ShadingParameters parameters)
        {
            SetShadingParameters(contextPtr, parameters);
        }
    }
}