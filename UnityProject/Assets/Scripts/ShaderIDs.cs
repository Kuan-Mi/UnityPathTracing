using UnityEngine;

namespace PathTracing
{
    public static class ShaderIDs
    {
        
        public static readonly int paramsID = Shader.PropertyToID("PathTracingParams");
        
        public static readonly int g_HashEntriesID = Shader.PropertyToID("gInOut_SharcHashEntriesBuffer");
        public static readonly int g_AccumulationBufferID = Shader.PropertyToID("gInOut_SharcAccumulated");
        public static readonly int g_ResolvedBufferID = Shader.PropertyToID("gInOut_SharcResolved");
        
        
        
        
        public static int g_ScramblingRankingID = Shader.PropertyToID("gIn_ScramblingRanking");
        public static int g_SobolID = Shader.PropertyToID("gIn_Sobol");

        // 测试用
        public static int g_OutputID = Shader.PropertyToID("g_Output");

        //  传入NRD的无噪声资源
        public static int g_MvID = Shader.PropertyToID("gOut_Mv");
        public static int g_ViewZID = Shader.PropertyToID("gOut_ViewZ");
        public static int g_Normal_RoughnessID = Shader.PropertyToID("gOut_Normal_Roughness");
        public static int g_BaseColor_MetalnessID = Shader.PropertyToID("gOut_BaseColor_Metalness");

        // 不传入NRD的资源
        public static int g_DirectLightingID = Shader.PropertyToID("gOut_DirectLighting");
        public static int g_DirectEmissionID = Shader.PropertyToID("gOut_DirectEmission");
        public static int g_PsrThroughputID = Shader.PropertyToID("gOut_PsrThroughput");

        // 传入NRD的有噪声资源
        public static int g_ShadowDataID = Shader.PropertyToID("gOut_ShadowData");
        public static int g_DiffID = Shader.PropertyToID("gOut_Diff");
        public static int g_SpecID = Shader.PropertyToID("gOut_Spec");
        
        public static int gIn_PrevComposedDiffID = Shader.PropertyToID("gIn_PrevComposedDiff");
        public static int gIn_PrevComposedSpec_PrevViewZID = Shader.PropertyToID("gIn_PrevComposedSpec_PrevViewZ");


        public static int g_AccelStructID = Shader.PropertyToID("gWorldTlas");


        // TraceOpaque
        public static int gIn_ViewZID = Shader.PropertyToID("gIn_ViewZ");
        public static int gIn_Normal_RoughnessID = Shader.PropertyToID("gIn_Normal_Roughness");
        public static int gIn_BaseColor_MetalnessID = Shader.PropertyToID("gIn_BaseColor_Metalness");
        public static int gIn_DirectLightingID = Shader.PropertyToID("gIn_DirectLighting");
        public static int gIn_DirectEmissionID = Shader.PropertyToID("gIn_DirectEmission");
        public static int gIn_ShadowID = Shader.PropertyToID("gIn_Shadow");
        public static int gIn_DiffID = Shader.PropertyToID("gIn_Diff");
        public static int gIn_SpecID = Shader.PropertyToID("gIn_Spec");
        public static int gIn_PsrThroughputID = Shader.PropertyToID("gIn_PsrThroughput");
        
        public static int gOut_ComposedDiffID = Shader.PropertyToID("gOut_ComposedDiff");
        public static int gOut_ComposedSpec_ViewZID = Shader.PropertyToID("gOut_ComposedSpec_ViewZ");

        // TraceTransparency
        public static int gIn_ComposedDiffID = Shader.PropertyToID("gIn_ComposedDiff");
        public static int gIn_ComposedSpec_ViewZID = Shader.PropertyToID("gIn_ComposedSpec_ViewZ");
        public static int gOut_ComposedID = Shader.PropertyToID("gOut_Composed");

        // TAA
        public static int gIn_MvID = Shader.PropertyToID("gIn_Mv");
        public static int gIn_ComposedID = Shader.PropertyToID("gIn_Composed");
        public static int gIn_HistoryID = Shader.PropertyToID("gIn_History");
        public static int gOut_ResultID = Shader.PropertyToID("gOut_Result");
        public static int gOut_DebugID = Shader.PropertyToID("gOut_Debug");
    }
}