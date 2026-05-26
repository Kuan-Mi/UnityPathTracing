// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Mirror of RenderingPlugin\External\RTXPT\Rtxpt\Shaders\PathTracer\Config.h
// Keep in sync with that file when tuning parameters.

namespace PathTracing
{
    /// <summary>
    /// C# mirror of Config.h — core path-tracer configuration constants in one place.
    /// </summary>
    internal static class PathTracerConfig
    {
        // ── Bounce / ray limits ───────────────────────────────────────────────

        /// <summary>Max value that BounceCount can be set to.</summary>
        internal const int MAX_BOUNCE_COUNT = 96;

        /// <summary>General max distance between any two surface points in the scene (excluding env-map).</summary>
        internal const float kMaxSceneDistance = 50000.0f;

        /// <summary>Effective environment map distance — large enough to avoid parallax.</summary>
        internal const float kEnvironmentMapSceneDistance = kMaxSceneDistance * 100.0f;

        /// <summary>Absolute max ray travel distance (one AU ≈ 1.5e11; 1e15 avoids precision issues).</summary>
        internal const float kMaxRayTravel = 1e15f;

        // ── Stable planes ─────────────────────────────────────────────────────

        /// <summary>Number of stable denoising planes. More than 3 is not supported.</summary>
        internal const int cStablePlaneCount = 3;

        // ── Compute thread dims ───────────────────────────────────────────────

        /// <summary>Compute shader tile size per dimension for path-tracer passes.</summary>
        internal const int NUM_COMPUTE_THREADS_PER_DIM = 8;

        // ── Environment map presampling ───────────────────────────────────────

        /// <summary>Number of presampled env-map entries (RTXDI). 2048 is a good compromise between quality and memory.</summary>
        internal const uint ENVMAP_PRESAMPLED_COUNT = 2048u;

        // ── HitInfo packing ───────────────────────────────────────────────────

        internal const int HIT_INFO_INSTANCE_ID_BITS      = 29;
        internal const int HIT_INFO_PRIMITIVE_INDEX_BITS  = 32;
        internal const int HIT_INFO_TYPE_BITS             = 3;
        internal const int HIT_INFO_USE_COMPRESSION       = 0;

        // ── Feature flags (must match shader #defines) ────────────────────────

        /// <summary>0 = disable STF; 1 = enable Stochastic Texture Filtering.</summary>
        internal const int RTXPT_STOCHASTIC_TEXTURE_FILTERING_ENABLE = 0;

        internal const int ENABLE_DEBUG_VIZUALISATIONS           = 1;
        internal const int ENABLE_DEBUG_DELTA_TREE_VIZUALISATION = 0;
        internal const int ENABLE_DEBUG_RTXDI_VIZUALISATION      = 0;

        // ── Path tracer mode IDs ──────────────────────────────────────────────

        /// <summary>Reference mode — stable planes ignored.</summary>
        internal const int PATH_TRACER_MODE_REFERENCE           = 0;

        /// <summary>Build stable planes pass — non-noisy rays only, stops at diffuse vertices.</summary>
        internal const int PATH_TRACER_MODE_BUILD_STABLE_PLANES = 1;

        /// <summary>Fill stable planes pass — standard noisy ray tracing with stable plane tracking.</summary>
        internal const int PATH_TRACER_MODE_FILL_STABLE_PLANES  = 2;
    }
}
