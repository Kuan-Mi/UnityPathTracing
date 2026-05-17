using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace NativeRender
{
    /// <summary>
    /// Burst-compatible math utilities for primitive data encoding.
    /// Mirrors the logic in NRDSampleResource.EncodeUnitVector / SafeSign.
    /// </summary>
    internal static class PrimitiveMathUtil
    {
        public static float SafeSign(float x) => x >= 0f ? 1f : -1f;

        public static float2 SafeSign(float2 v) => new float2(SafeSign(v.x), SafeSign(v.y));

        public static half2 EncodeUnitVectorSigned(float3 v) => EncodeUnitVector(v, true);

        public static half2 EncodeUnitVector(float3 v, bool bSigned)
        {
            v /= math.dot(math.abs(v), 1f);
            float2 octWrap = (1f - math.abs(v.yx)) * SafeSign(v.xy);
            v.xy = v.z >= 0f ? v.xy : octWrap;
            return new half2(bSigned ? v.xy : 0.5f * v.xy + 0.5f);
        }
    }

    /// <summary>
    /// Burst-compiled parallel job that fills a contiguous slice of
    /// <see cref="PrimitiveDataNRD"/> for a single submesh.
    ///
    /// One job instance is scheduled per submesh.
    /// <c>Execute(i)</c> processes the i-th triangle (indices [i*3 .. i*3+2]).
    ///
    /// The caller passes:
    ///   - <see cref="Indices"/>: the submesh's raw index buffer (length = indexCount)
    ///   - vertex attribute arrays spanning the whole mesh (full vertex count)
    ///   - <see cref="Output"/>: a <c>GetSubArray</c> slice of the flat output array
    ///     starting at this submesh's primitive offset
    /// </summary>
    [BurstCompile]
    internal struct BuildPrimitivesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int>    Indices;   // indexCount = triCount * 3
        [ReadOnly] public NativeArray<float3> Positions; // full mesh vertex count
        [ReadOnly] public NativeArray<float3> Normals;
        [ReadOnly] public NativeArray<float4> Tangents;
        [ReadOnly] public NativeArray<float2> UVs;

        public bool HasN;
        public bool HasT;
        public bool HasUV;

        // Caller-provided slice: primNative.GetSubArray(localOffset, triCount).
        // Slices are non-overlapping by construction; disable the alias safety check
        // so multiple jobs targeting different ranges of the same backing array can
        // be scheduled in parallel without a false-positive aliasing error.
        [NativeDisableContainerSafetyRestriction]
        public NativeArray<PrimitiveDataNRD> Output;

        public void Execute(int triIdx)
        {
            int i0 = Indices[triIdx * 3];
            int i1 = Indices[triIdx * 3 + 1];
            int i2 = Indices[triIdx * 3 + 2];

            float3 p0 = Positions[i0];
            float3 p1 = Positions[i1];
            float3 p2 = Positions[i2];

            float2 uv0 = HasUV ? UVs[i0] : float2.zero;
            float2 uv1 = HasUV ? UVs[i1] : float2.zero;
            float2 uv2 = HasUV ? UVs[i2] : float2.zero;

            float  worldArea = 0.5f * math.length(math.cross(p1 - p0, p2 - p0));
            float2 du1       = uv1 - uv0;
            float2 du2       = uv2 - uv0;
            float  uvArea    = 0.5f * math.abs(du1.x * du2.y - du1.y * du2.x);

            float3 rn0 = HasN ? Normals[i0] : new float3(0f, 1f, 0f);
            float3 rn1 = HasN ? Normals[i1] : new float3(0f, 1f, 0f);
            float3 rn2 = HasN ? Normals[i2] : new float3(0f, 1f, 0f);
            float3 n0  = math.lengthsq(rn0) > 1e-12f ? math.normalize(rn0) : new float3(0f, 1f, 0f);
            float3 n1  = math.lengthsq(rn1) > 1e-12f ? math.normalize(rn1) : new float3(0f, 1f, 0f);
            float3 n2  = math.lengthsq(rn2) > 1e-12f ? math.normalize(rn2) : new float3(0f, 1f, 0f);

            float4 t0Raw = HasT ? Tangents[i0] : new float4(1f, 0f, 0f, 1f);
            float4 t1Raw = HasT ? Tangents[i1] : new float4(1f, 0f, 0f, 1f);
            float4 t2Raw = HasT ? Tangents[i2] : new float4(1f, 0f, 0f, 1f);

            Output[triIdx] = new PrimitiveDataNRD
            {
                uv0           = new half2(uv0),
                uv1           = new half2(uv1),
                uv2           = new half2(uv2),
                worldArea     = worldArea,
                n0            = PrimitiveMathUtil.EncodeUnitVectorSigned(n0),
                n1            = PrimitiveMathUtil.EncodeUnitVectorSigned(n1),
                n2            = PrimitiveMathUtil.EncodeUnitVectorSigned(n2),
                uvArea        = uvArea,
                t0            = PrimitiveMathUtil.EncodeUnitVectorSigned(t0Raw.xyz),
                t1            = PrimitiveMathUtil.EncodeUnitVectorSigned(t1Raw.xyz),
                t2            = PrimitiveMathUtil.EncodeUnitVectorSigned(t2Raw.xyz),
                bitangentSign = -t0Raw.w,
            };
        }
    }

    // =========================================================================
    // Merged-BLAS jobs
    // =========================================================================

    /// <summary>
    /// Transforms local-space positions to world space and writes them into a
    /// contiguous sub-array of the merged VB.
    /// </summary>
    [BurstCompile]
    internal struct TransformVerticesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> LocalPositions;
        public float4x4 LocalToWorld;

        [NativeDisableContainerSafetyRestriction]
        public NativeArray<float3> Output; // mergedVB.GetSubArray(vertBase, vertCount)

        public void Execute(int i)
        {
            Output[i] = math.transform(LocalToWorld, LocalPositions[i]);
        }
    }

    /// <summary>
    /// Remaps submesh-local indices to global merged-VB indices
    /// (output[i] = vertBase + localIndices[i]).
    /// </summary>
    [BurstCompile]
    internal struct RemapIndicesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int> LocalIndices;
        public int VertBase;

        [NativeDisableContainerSafetyRestriction]
        public NativeArray<uint> Output; // mergedIB.GetSubArray(iBase, indexCount)

        public void Execute(int i)
        {
            Output[i] = (uint)(VertBase + LocalIndices[i]);
        }
    }

    /// <summary>
    /// Fills <see cref="PrimitiveDataNRD"/> for merged-BLAS geometry.
    /// Normals are transformed to world space via <see cref="NormalMatrix"/>
    /// (inverse-transpose of LocalToWorld); tangents via the upper-3×3 of
    /// <see cref="LocalToWorld"/>.
    /// </summary>
    [BurstCompile]
    internal struct BuildMergedPrimitivesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int>    Indices;        // local submesh indices
        [ReadOnly] public NativeArray<float3> LocalPositions;
        [ReadOnly] public NativeArray<float3> LocalNormals;
        [ReadOnly] public NativeArray<float4> LocalTangents;
        [ReadOnly] public NativeArray<float2> UVs;

        public bool HasN;
        public bool HasT;
        public bool HasUV;

        public float4x4 LocalToWorld;   // for positions + tangent direction
        public float4x4 NormalMatrix;   // inverse-transpose, for normal direction

        [NativeDisableContainerSafetyRestriction]
        public NativeArray<PrimitiveDataNRD> Output;

        public void Execute(int triIdx)
        {
            int i0 = Indices[triIdx * 3];
            int i1 = Indices[triIdx * 3 + 1];
            int i2 = Indices[triIdx * 3 + 2];

            // World-space positions.
            float3 p0 = math.transform(LocalToWorld, LocalPositions[i0]);
            float3 p1 = math.transform(LocalToWorld, LocalPositions[i1]);
            float3 p2 = math.transform(LocalToWorld, LocalPositions[i2]);

            float2 uv0 = HasUV ? UVs[i0] : float2.zero;
            float2 uv1 = HasUV ? UVs[i1] : float2.zero;
            float2 uv2 = HasUV ? UVs[i2] : float2.zero;

            float  worldArea = 0.5f * math.length(math.cross(p1 - p0, p2 - p0));
            float2 du1       = uv1 - uv0;
            float2 du2       = uv2 - uv0;
            float  uvArea    = 0.5f * math.abs(du1.x * du2.y - du1.y * du2.x);

            // World-space normals via normal matrix (inverse-transpose).
            float3x3 nm  = new float3x3(NormalMatrix);
            float3   rn0 = HasN ? math.mul(nm, LocalNormals[i0]) : new float3(0f, 1f, 0f);
            float3   rn1 = HasN ? math.mul(nm, LocalNormals[i1]) : new float3(0f, 1f, 0f);
            float3   rn2 = HasN ? math.mul(nm, LocalNormals[i2]) : new float3(0f, 1f, 0f);
            float3   n0  = math.lengthsq(rn0) > 1e-12f ? math.normalize(rn0) : new float3(0f, 1f, 0f);
            float3   n1  = math.lengthsq(rn1) > 1e-12f ? math.normalize(rn1) : new float3(0f, 1f, 0f);
            float3   n2  = math.lengthsq(rn2) > 1e-12f ? math.normalize(rn2) : new float3(0f, 1f, 0f);

            // World-space tangent directions via upper-3×3 of LocalToWorld.
            float3x3 m33  = new float3x3(LocalToWorld);
            float4   t0Raw = HasT ? LocalTangents[i0] : new float4(1f, 0f, 0f, 1f);
            float4   t1Raw = HasT ? LocalTangents[i1] : new float4(1f, 0f, 0f, 1f);
            float4   t2Raw = HasT ? LocalTangents[i2] : new float4(1f, 0f, 0f, 1f);
            float3   td0   = math.mul(m33, t0Raw.xyz);
            float3   td1   = math.mul(m33, t1Raw.xyz);
            float3   td2   = math.mul(m33, t2Raw.xyz);
            float3   t0    = math.lengthsq(td0) > 1e-12f ? math.normalize(td0) : new float3(1f, 0f, 0f);
            float3   t1    = math.lengthsq(td1) > 1e-12f ? math.normalize(td1) : new float3(1f, 0f, 0f);
            float3   t2    = math.lengthsq(td2) > 1e-12f ? math.normalize(td2) : new float3(1f, 0f, 0f);

            Output[triIdx] = new PrimitiveDataNRD
            {
                uv0           = new half2(uv0),
                uv1           = new half2(uv1),
                uv2           = new half2(uv2),
                worldArea     = worldArea,
                n0            = PrimitiveMathUtil.EncodeUnitVectorSigned(n0),
                n1            = PrimitiveMathUtil.EncodeUnitVectorSigned(n1),
                n2            = PrimitiveMathUtil.EncodeUnitVectorSigned(n2),
                uvArea        = uvArea,
                t0            = PrimitiveMathUtil.EncodeUnitVectorSigned(t0),
                t1            = PrimitiveMathUtil.EncodeUnitVectorSigned(t1),
                t2            = PrimitiveMathUtil.EncodeUnitVectorSigned(t2),
                bitangentSign = -t0Raw.w,
            };
        }
    }
}
