// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using Rtxdi.LightSampling;

namespace Rtxdi.ReGIR
{
    public enum ReGIRMode : uint
    {
        Disabled = ReGIRConstants.RTXDI_REGIR_DISABLED,
        Grid     = ReGIRConstants.RTXDI_REGIR_GRID,
        Onion    = ReGIRConstants.RTXDI_REGIR_ONION,
    }

    public struct ReGIRGridStaticParameters
    {
        public uint3 GridSize;

        public static ReGIRGridStaticParameters Default()
        {
            return new ReGIRGridStaticParameters
            {
                GridSize = new uint3(16, 16, 16)
            };
        }
    }

    public struct ReGIROnionStaticParameters
    {
        public uint OnionDetailLayers;
        public uint OnionCoverageLayers;

        public static ReGIROnionStaticParameters Default()
        {
            return new ReGIROnionStaticParameters
            {
                OnionDetailLayers = 5,
                OnionCoverageLayers = 10,
            };
        }
    }

    public struct ReGIRStaticParameters
    {
        public ReGIRMode Mode;
        public uint LightsPerCell;
        public ReGIRGridStaticParameters gridParameters;
        public ReGIROnionStaticParameters onionParameters;

        public static ReGIRStaticParameters Default()
        {
            return new ReGIRStaticParameters
            {
                Mode = ReGIRMode.Onion,
                // Mode = ReGIRMode.Grid,
                LightsPerCell = 512,
                gridParameters = ReGIRGridStaticParameters.Default(),
                onionParameters = ReGIROnionStaticParameters.Default(),
            };
        }
    }

    public struct ReGIRGridCalculatedParameters
    {
        public uint lightSlotCount;
        public uint pad;
    }

    public class ReGIROnionCalculatedParameters
    {
        public uint lightSlotCount;
        public uint regirOnionCells;
        public List<ReGIR_OnionLayerGroup> regirOnionLayers = new List<ReGIR_OnionLayerGroup>();
        public List<ReGIR_OnionRing> regirOnionRings = new List<ReGIR_OnionRing>();
        public float regirOnionCubicRootFactor;
        public float regirOnionLinearFactor;
    }

    public enum LocalLightReGIRPresamplingMode : uint
    {
        Uniform   = ReGIRConstants.REGIR_LOCAL_LIGHT_PRESAMPLING_MODE_UNIFORM,
        Power_RIS = ReGIRConstants.REGIR_LOCAL_LIGHT_PRESAMPLING_MODE_POWER_RIS,
    }

    public enum LocalLightReGIRFallbackSamplingMode : uint
    {
        Uniform   = ReGIRConstants.REGIR_LOCAL_LIGHT_FALLBACK_MODE_UNIFORM,
        Power_RIS = ReGIRConstants.REGIR_LOCAL_LIGHT_FALLBACK_MODE_POWER_RIS,
    }

    [Serializable]
    public struct ReGIRDynamicParameters
    {
        public float regirCellSize;
        public float3 center;
        public LocalLightReGIRFallbackSamplingMode fallbackSamplingMode;
        public LocalLightReGIRPresamplingMode presamplingMode;
        public float regirSamplingJitter;
        public uint regirNumBuildSamples;

        public static ReGIRDynamicParameters Default()
        {
            return new ReGIRDynamicParameters
            {
                regirCellSize = 1.0f,
                center = new float3(0.0f, 0.0f, 0.0f),
                fallbackSamplingMode = LocalLightReGIRFallbackSamplingMode.Power_RIS,
                presamplingMode = LocalLightReGIRPresamplingMode.Power_RIS,
                regirSamplingJitter = 1.0f,
                regirNumBuildSamples = 8,
            };
        }
    }

    public class ReGIRContext
    {
        private const float c_pi = 3.1415926535f;

        private uint m_regirCellOffset;
        private ReGIRStaticParameters m_regirStaticParameters;
        private ReGIRDynamicParameters m_regirDynamicParameters;
        private ReGIROnionCalculatedParameters m_regirOnionCalculatedParameters;
        private ReGIRGridCalculatedParameters m_regirGridCalculatedParameters;

        public ReGIRContext(ReGIRStaticParameters staticParams, RISBufferSegmentAllocator risBufferSegmentAllocator)
        {
            m_regirCellOffset = 0;
            m_regirStaticParameters = staticParams;
            m_regirDynamicParameters = ReGIRDynamicParameters.Default();
            m_regirOnionCalculatedParameters = new ReGIROnionCalculatedParameters();
            m_regirGridCalculatedParameters = new ReGIRGridCalculatedParameters();

            ComputeGridLightSlotCount();
            InitializeOnion(staticParams);
            ComputeOnionJitterCurve();
            AllocateRISBufferSegment(risBufferSegmentAllocator);
        }

        public bool IsLocalLightPowerRISEnable()
        {
            return (m_regirDynamicParameters.presamplingMode == LocalLightReGIRPresamplingMode.Power_RIS) ||
                   (m_regirDynamicParameters.fallbackSamplingMode == LocalLightReGIRFallbackSamplingMode.Power_RIS);
        }

        public uint GetReGIRCellOffset()           => m_regirCellOffset;
        public ReGIRGridCalculatedParameters GetReGIRGridCalculatedParameters() => m_regirGridCalculatedParameters;
        public ReGIROnionCalculatedParameters GetReGIROnionCalculatedParameters() => m_regirOnionCalculatedParameters;
        public ReGIRDynamicParameters GetReGIRDynamicParameters() => m_regirDynamicParameters;
        public ReGIRStaticParameters GetReGIRStaticParameters()   => m_regirStaticParameters;

        public uint GetReGIRLightSlotCount()
        {
            switch (m_regirStaticParameters.Mode)
            {
                case ReGIRMode.Grid:
                    return m_regirGridCalculatedParameters.lightSlotCount;
                case ReGIRMode.Onion:
                    return m_regirOnionCalculatedParameters.lightSlotCount;
                default:
                    return 0;
            }
        }

        public void SetDynamicParameters(ReGIRDynamicParameters regirDynamicParameters)
        {
            m_regirDynamicParameters = regirDynamicParameters;
        }

        // ---- Private methods ----

        private void ComputeGridLightSlotCount()
        {
            m_regirGridCalculatedParameters.lightSlotCount =
                m_regirStaticParameters.gridParameters.GridSize.x
                * m_regirStaticParameters.gridParameters.GridSize.y
                * m_regirStaticParameters.gridParameters.GridSize.z
                * m_regirStaticParameters.LightsPerCell;
        }

        private void AllocateRISBufferSegment(RISBufferSegmentAllocator risBufferSegmentAllocator)
        {
            switch (m_regirStaticParameters.Mode)
            {
                default:
                case ReGIRMode.Disabled:
                    m_regirCellOffset = 0;
                    break;
                case ReGIRMode.Grid:
                    m_regirCellOffset = risBufferSegmentAllocator.AllocateSegment(m_regirGridCalculatedParameters.lightSlotCount);
                    break;
                case ReGIRMode.Onion:
                    m_regirCellOffset = risBufferSegmentAllocator.AllocateSegment(m_regirOnionCalculatedParameters.lightSlotCount);
                    break;
            }
        }

        private void InitializeOnion(ReGIRStaticParameters staticParams)
        {
            int numLayerGroups = Math.Max(1, Math.Min(ReGIRConstants.RTXDI_ONION_MAX_LAYER_GROUPS, (int)staticParams.onionParameters.OnionDetailLayers));

            float innerRadius = 1f;

            int totalLayers = 0;
            int totalCells = 1;

            for (int layerGroupIndex = 0; layerGroupIndex < numLayerGroups; layerGroupIndex++)
            {
                int partitions = layerGroupIndex * 4 + 8;
                int layerCount = (layerGroupIndex < numLayerGroups - 1)
                    ? 1
                    : (int)staticParams.onionParameters.OnionCoverageLayers + 1;

                float radiusRatio = ((float)partitions + c_pi) / ((float)partitions - c_pi);
                float outerRadius = innerRadius * (float)Math.Pow(radiusRatio, layerCount);
                float equatorialAngle = 2 * c_pi / (float)partitions;

                ReGIR_OnionLayerGroup layerGroup = new ReGIR_OnionLayerGroup();
                layerGroup.ringOffset = m_regirOnionCalculatedParameters.regirOnionRings.Count;
                layerGroup.innerRadius = innerRadius;
                layerGroup.outerRadius = outerRadius;
                layerGroup.invLogLayerScale = 1f / (float)Math.Log(radiusRatio);
                layerGroup.invEquatorialCellAngle = 1f / equatorialAngle;
                layerGroup.equatorialCellAngle = equatorialAngle;
                layerGroup.ringCount = partitions / 4 + 1;
                layerGroup.layerScale = radiusRatio;
                layerGroup.layerCellOffset = totalCells;

                ReGIR_OnionRing ring = new ReGIR_OnionRing();
                ring.cellCount = partitions;
                ring.cellOffset = 0;
                ring.invCellAngle = (float)partitions / (2 * c_pi);
                ring.cellAngle = 1f / ring.invCellAngle;
                m_regirOnionCalculatedParameters.regirOnionRings.Add(ring);

                int cellsPerLayer = partitions;
                for (int ringIndex = 1; ringIndex < layerGroup.ringCount; ringIndex++)
                {
                    ring = new ReGIR_OnionRing();
                    ring.cellCount = Math.Max(1, (int)Math.Floor((float)partitions * (float)Math.Cos(ringIndex * equatorialAngle)));
                    ring.cellOffset = cellsPerLayer;
                    ring.invCellAngle = (float)ring.cellCount / (2 * c_pi);
                    ring.cellAngle = 1f / ring.invCellAngle;
                    m_regirOnionCalculatedParameters.regirOnionRings.Add(ring);

                    cellsPerLayer += ring.cellCount * 2;
                }

                layerGroup.cellsPerLayer = cellsPerLayer;
                layerGroup.layerCount = layerCount;
                m_regirOnionCalculatedParameters.regirOnionLayers.Add(layerGroup);

                innerRadius = outerRadius;

                totalCells += cellsPerLayer * layerCount;
                totalLayers += layerCount;
            }

            m_regirOnionCalculatedParameters.regirOnionCells = (uint)totalCells;
            m_regirOnionCalculatedParameters.lightSlotCount =
                m_regirOnionCalculatedParameters.regirOnionCells * m_regirStaticParameters.LightsPerCell;
        }

        private static float3 SphericalToCartesian(float radius, float azimuth, float elevation)
        {
            return new float3(
                radius * (float)Math.Cos(azimuth) * (float)Math.Cos(elevation),
                radius * (float)Math.Sin(elevation),
                radius * (float)Math.Sin(azimuth) * (float)Math.Cos(elevation)
            );
        }

        private static float Distance(float3 a, float3 b)
        {
            float3 d = a - b;
            return (float)Math.Sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        }

        private void ComputeOnionJitterCurve()
        {
            List<float> cubicRootFactors = new List<float>();
            List<float> linearFactors = new List<float>();

            int layerGroupIndex = 0;
            foreach (var layerGroup in m_regirOnionCalculatedParameters.regirOnionLayers)
            {
                for (int layerIndex = 0; layerIndex < layerGroup.layerCount; layerIndex++)
                {
                    float innerRadius = layerGroup.innerRadius * (float)Math.Pow(layerGroup.layerScale, layerIndex);
                    float outerRadius = innerRadius * layerGroup.layerScale;
                    float middleRadius = (innerRadius + outerRadius) * 0.5f;
                    float maxCellRadius = 0f;

                    for (int ringIndex = 0; ringIndex < layerGroup.ringCount; ringIndex++)
                    {
                        var ring = m_regirOnionCalculatedParameters.regirOnionRings[layerGroup.ringOffset + ringIndex];

                        float middleElevation = layerGroup.equatorialCellAngle * ringIndex;
                        float vertexElevation = (ringIndex == 0)
                            ? layerGroup.equatorialCellAngle * 0.5f
                            : middleElevation - layerGroup.equatorialCellAngle * 0.5f;

                        float middleAzimuth = 0f;
                        float vertexAzimuth = ring.cellAngle;

                        float3 middlePoint = SphericalToCartesian(middleRadius, middleAzimuth, middleElevation);
                        float3 vertexPoint = SphericalToCartesian(outerRadius, vertexAzimuth, vertexElevation);

                        float cellRadius = Distance(middlePoint, vertexPoint);
                        maxCellRadius = Math.Max(maxCellRadius, cellRadius);
                    }

                    if (layerGroupIndex < m_regirOnionCalculatedParameters.regirOnionLayers.Count - 1)
                    {
                        float cubicRootFactor = maxCellRadius * (float)Math.Pow(middleRadius, -1f / 3f);
                        cubicRootFactors.Add(cubicRootFactor);
                    }
                    else
                    {
                        float linearFactor = maxCellRadius / middleRadius;
                        linearFactors.Add(linearFactor);
                    }
                }

                layerGroupIndex++;
            }

            // Compute the median of the cubic root factors, there are some outliers in the curve
            if (cubicRootFactors.Count > 0)
            {
                cubicRootFactors.Sort();
                m_regirOnionCalculatedParameters.regirOnionCubicRootFactor = cubicRootFactors[cubicRootFactors.Count / 2];
            }
            else
            {
                m_regirOnionCalculatedParameters.regirOnionCubicRootFactor = 0f;
            }

            // Compute the average of the linear factors, they're all the same anyway
            float sumOfLinearFactors = linearFactors.Sum();
            m_regirOnionCalculatedParameters.regirOnionLinearFactor =
                sumOfLinearFactors / Math.Max(linearFactors.Count, 1f);
        }
    }
}
