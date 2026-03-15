// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using UnityEngine;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.ReGIR;
using Rtxdi.LightSampling;

namespace Rtxdi
{
    public struct ImportanceSamplingContext_StaticParameters
    {
        // RIS buffer params for light presampling
        public RISBufferSegmentParameters localLightRISBufferParams;
        public RISBufferSegmentParameters environmentLightRISBufferParams;

        // Shared options for ReSTIRDI and ReSTIRGI
        public uint NeighborOffsetCount;
        public uint renderWidth;
        public uint renderHeight;
        public CheckerboardMode CheckerboardSamplingMode;

        // ReGIR params
        public ReGIRStaticParameters regirStaticParams;

        public static ImportanceSamplingContext_StaticParameters Default()
        {
            return new ImportanceSamplingContext_StaticParameters
            {
                localLightRISBufferParams = new RISBufferSegmentParameters { tileSize = 1024, tileCount = 128 },
                environmentLightRISBufferParams = new RISBufferSegmentParameters { tileSize = 1024, tileCount = 128 },
                NeighborOffsetCount = 8192,
                renderWidth = 0,
                renderHeight = 0,
                CheckerboardSamplingMode = CheckerboardMode.Off,
                regirStaticParams = ReGIRStaticParameters.Default(),
            };
        }
    }

    public class ImportanceSamplingContext
    {
        private RISBufferSegmentAllocator m_risBufferSegmentAllocator;
        private ReSTIRDIContext m_restirDIContext;
        private ReGIRContext m_regirContext;
        private ReSTIRGIContext m_restirGIContext;

        private RTXDI_LightBufferParameters m_lightBufferParams;
        private RTXDI_RISBufferSegmentParameters m_localLightRISBufferSegmentParams;
        private RTXDI_RISBufferSegmentParameters m_environmentLightRISBufferSegmentParams;

        public ImportanceSamplingContext(ImportanceSamplingContext_StaticParameters isParams)
        {
            m_lightBufferParams = new RTXDI_LightBufferParameters();

            DebugCheckParameters(isParams.localLightRISBufferParams, isParams.environmentLightRISBufferParams);

            m_risBufferSegmentAllocator = new RISBufferSegmentAllocator();

            m_localLightRISBufferSegmentParams.bufferOffset =
                m_risBufferSegmentAllocator.AllocateSegment(isParams.localLightRISBufferParams.tileCount * isParams.localLightRISBufferParams.tileSize);
            m_localLightRISBufferSegmentParams.tileCount = isParams.localLightRISBufferParams.tileCount;
            m_localLightRISBufferSegmentParams.tileSize = isParams.localLightRISBufferParams.tileSize;

            m_environmentLightRISBufferSegmentParams.bufferOffset =
                m_risBufferSegmentAllocator.AllocateSegment(isParams.environmentLightRISBufferParams.tileCount * isParams.environmentLightRISBufferParams.tileSize);
            m_environmentLightRISBufferSegmentParams.tileCount = isParams.environmentLightRISBufferParams.tileCount;
            m_environmentLightRISBufferSegmentParams.tileSize = isParams.environmentLightRISBufferParams.tileSize;

            var restirDIStaticParams = new ReSTIRDIStaticParameters
            {
                CheckerboardSamplingMode = isParams.CheckerboardSamplingMode,
                NeighborOffsetCount = isParams.NeighborOffsetCount,
                RenderWidth = isParams.renderWidth,
                RenderHeight = isParams.renderHeight,
            };
            m_restirDIContext = new ReSTIRDIContext(restirDIStaticParams);

            m_regirContext = new ReGIRContext(isParams.regirStaticParams, m_risBufferSegmentAllocator);

            var restirGIStaticParams = new ReSTIRGIStaticParameters
            {
                CheckerboardSamplingMode = isParams.CheckerboardSamplingMode,
                RenderWidth = isParams.renderWidth,
                RenderHeight = isParams.renderHeight,
            };
            m_restirGIContext = new ReSTIRGIContext(restirGIStaticParams);
        }

        // Accessors
        public ReSTIRDIContext GetReSTIRDIContext()         => m_restirDIContext;
        public ReGIRContext GetReGIRContext()               => m_regirContext;
        public ReSTIRGIContext GetReSTIRGIContext()         => m_restirGIContext;
        public RISBufferSegmentAllocator GetRISBufferSegmentAllocator() => m_risBufferSegmentAllocator;

        public ref readonly RTXDI_LightBufferParameters GetLightBufferParameters()
            => ref m_lightBufferParams;
        public ref readonly RTXDI_RISBufferSegmentParameters GetLocalLightRISBufferSegmentParams()
            => ref m_localLightRISBufferSegmentParams;
        public ref readonly RTXDI_RISBufferSegmentParameters GetEnvironmentLightRISBufferSegmentParams()
            => ref m_environmentLightRISBufferSegmentParams;

        public uint GetNeighborOffsetCount()
        {
            return m_restirDIContext.GetStaticParameters().NeighborOffsetCount;
        }

        public bool IsLocalLightPowerRISEnabled()
        {
            var iss = m_restirDIContext.GetInitialSamplingParameters();
            if (iss.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode.Power_RIS)
                return true;
            if (iss.localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS)
            {
                if (m_regirContext.GetReGIRDynamicParameters().presamplingMode == LocalLightReGIRPresamplingMode.Power_RIS ||
                    m_regirContext.GetReGIRDynamicParameters().fallbackSamplingMode == LocalLightReGIRFallbackSamplingMode.Power_RIS)
                    return true;
            }
            return false;
        }

        public bool IsReGIREnabled()
        {
            return m_restirDIContext.GetInitialSamplingParameters().localLightSamplingMode == ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS;
        }

        public void SetLightBufferParams(RTXDI_LightBufferParameters lightBufferParams)
        {
            m_lightBufferParams = lightBufferParams;
        }

        // Private helpers
        private static bool IsNonzeroPowerOf2(uint i)
        {
            return ((i & (i - 1)) == 0) && (i > 0);
        }

        private static void DebugCheckParameters(
            RISBufferSegmentParameters localLightRISBufferParams,
            RISBufferSegmentParameters environmentLightRISBufferParams)
        {
            Debug.Assert(IsNonzeroPowerOf2(localLightRISBufferParams.tileSize));
            Debug.Assert(IsNonzeroPowerOf2(localLightRISBufferParams.tileCount));
            Debug.Assert(IsNonzeroPowerOf2(environmentLightRISBufferParams.tileSize));
            Debug.Assert(IsNonzeroPowerOf2(environmentLightRISBufferParams.tileCount));
        }
    }
}
