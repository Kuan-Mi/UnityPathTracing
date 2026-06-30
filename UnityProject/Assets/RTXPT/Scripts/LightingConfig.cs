// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Mirror of RenderingPlugin\External\RTXPT\Rtxpt\Shaders\PathTracer\Lighting\LightingConfig.h
// Keep in sync with that file when tuning parameters.

namespace PathTracing
{
    /// <summary>
    /// C# mirror of LightingConfig.h — all LightingConfig constants in one place.
    /// </summary>
    internal static class LightingConfig
    {
        // ── General settings ──────────────────────────────────────────────────

        /// <summary>Number of PolymorphicLightInfo slots (currently 48 bytes each).</summary>
        internal const int RTXPT_LIGHTING_MAX_LIGHTS = 512 * 1024;

        /// <summary>Every light can have this many proxies on average.</summary>
        internal const int RTXPT_LIGHTING_SAMPLING_PROXY_RATIO = 12;

        /// <summary>Total buffer size required for proxies, worst-case scenario.</summary>
        internal const int RTXPT_LIGHTING_MAX_SAMPLING_PROXIES = RTXPT_LIGHTING_SAMPLING_PROXY_RATIO * RTXPT_LIGHTING_MAX_LIGHTS;

        /// <summary>One light can have no more than this many (global sampling) proxies.</summary>
        internal const int RTXPT_LIGHTING_MAX_SAMPLING_PROXIES_PER_LIGHT = 256 * 1024;

        // ── Tile (local) sampling settings  — default preset ─────────────────

        internal const int RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE       = 8;
        internal const int RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE     = 8;
        internal const int RTXPT_LIGHTING_LOCAL_PROXY_COUNT               = 128;
        internal const int RTXPT_LIGHTING_LOCAL_PROXY_BINARY_SEARCH_STEPS = 8;

        internal const int RTXPT_LIGHTING_TOP_UP_SAMPLES =
            RTXPT_LIGHTING_LOCAL_PROXY_COUNT
            - RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE * RTXPT_LIGHTING_SAMPLING_BUFFER_WINDOW_SIZE;

        // ── Early feedback ────────────────────────────────────────────────────

        internal const int RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE = 2;

        // ── Environment map quad-tree settings ───────────────────────────────

        /// <summary>First-pass starting point — must be a power of two.</summary>
        internal const int RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION = 4;

        /// <summary>First-pass subdivision count.</summary>
        internal const int RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS = 24;

        /// <summary>For each subdivision: one node goes out, four are added → net +3.</summary>
        internal const int RTXPT_NEEAT_ENVMAP_QT_ADDITIONAL_NODES =
            3 * RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS; // 72

        internal const int RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT =
            RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION * RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION
            + RTXPT_NEEAT_ENVMAP_QT_ADDITIONAL_NODES; // 88

        /// <summary>Stop base subdivision this many levels early to leave room for boost pass.</summary>
        internal const int RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION_DPT = 3;

        /// <summary>How many times to subdivide in the boost pass.</summary>
        internal const int RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION = 20;

        internal const int RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT =
            RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION * 3 + 1; // 61

        internal const int RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT =
            RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT * RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT; // 5368

        // ── Misc settings ─────────────────────────────────────────────────────

        /// <summary>Provide NEE-AT feedback from BSDF rays hitting emissive surface/sky.</summary>
        internal const int RTXPT_LIGHTING_ENABLE_BSDF_FEEDBACK = 0;

        /// <summary>Counters are packed into 6 bits, so max is 63.</summary>
        internal const int RTXPT_LIGHTING_MAX_SAMPLE_COUNT = 63;

        /// <summary>When counting lights for global feedback, process only one candidate to avoid InterlockedAdd bottleneck.</summary>
        internal const int RTXPT_LIGHTING_COUNT_ONLY_ONE_GLOBAL_FEEDBACK = 1;

        internal const float RTXPT_LIGHTING_SCREEN_SPACE_COHERENT_FEEDBACK_BIAS = 1.0f;

        // ── Derived constants used by buffer allocation ───────────────────────

        /// <summary>Mirrors RTXPT_LIGHTING_WEIGHTS_COUNT_HALF = MaxLights + 1 (ping-pong half).</summary>
        internal const int RTXPT_LIGHTING_WEIGHTS_COUNT_HALF = RTXPT_LIGHTING_MAX_LIGHTS + 1;
    }
}