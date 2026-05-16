using System;
using System.Collections.Generic;

namespace Nrd
{
    // -----------------------------------------------------------------------
    // RELAX – spatiotemporal accumulation denoiser
    // -----------------------------------------------------------------------
    public sealed class RelaxDenoiser : NrdDenoiser
    {
        private readonly Denoiser _denoiser;

        /// <summary>Per-frame input for RELAX denoising.</summary>
        public struct FrameInput
        {
            public CommonFrameInput common;

            // --- 累积帧数 ---
            /// <summary>[0; RELAX_MAX_HISTORY_FRAME_NUM] Default: 30</summary>
            public uint diffuseMaxAccumulatedFrameNum;
            /// <summary>[0; RELAX_MAX_HISTORY_FRAME_NUM] Default: 30</summary>
            public uint specularMaxAccumulatedFrameNum;
            /// <summary>快速历史长度，通常为 maxAccumulated/5~7。Default: 6</summary>
            public uint diffuseMaxFastAccumulatedFrameNum;
            /// <summary>Default: 6</summary>
            public uint specularMaxFastAccumulatedFrameNum;

            // --- 棋盘格 / 重建 ---
            public CheckerboardMode              checkerboardMode;
            public HitDistanceReconstructionMode hitDistanceReconstructionMode;

            // --- 预通道模糊半径 ---
            /// <summary>(pixels) 0 = 禁用，概率采样时需要开启。Default: 30</summary>
            public float diffusePrepassBlurRadius;
            /// <summary>Default: 50</summary>
            public float specularPrepassBlurRadius;

            // --- 快速历史钳制 ---
            /// <summary>[1; 3] Default: 2.0f</summary>
            public float fastHistoryClampingSigmaScale;

            // --- A-Trous 灵敏度 ---
            /// <summary>Default: 2.0f</summary>
            public float diffusePhiLuminance;
            /// <summary>Default: 1.0f</summary>
            public float specularPhiLuminance;

            // --- 其他 ---
            /// <summary>Default: false</summary>
            public bool enableAntiFirefly;
        }

        private static readonly HashSet<Denoiser> ValidDenoisers = new()
        {
            Denoiser.RELAX_DIFFUSE,
            Denoiser.RELAX_DIFFUSE_SH,
            Denoiser.RELAX_SPECULAR,
            Denoiser.RELAX_SPECULAR_SH,
            Denoiser.RELAX_DIFFUSE_SPECULAR,
            Denoiser.RELAX_DIFFUSE_SPECULAR_SH,
        };

        public RelaxDenoiser(string camName, Denoiser denoiser)
            : base(camName, new NrdDenoiserDesc(denoiser))
        {
            if (!ValidDenoisers.Contains(denoiser))
                throw new ArgumentException(
                    $"RelaxNrdDenoiser requires a RELAX_* denoiser, got {denoiser}.", nameof(denoiser));
            _denoiser   = denoiser;
        }

        public IntPtr GetInteropDataPtr(FrameInput fi)
        {
            var data = NrdFrameData._default;
            FillCommonSettings(ref data, fi.common);

            data.denoiserCount = 1;
            ref var entry = ref NrdFrameData.GetEntry(ref data, 0);
            entry.identifier = 0;
            entry.denoiser   = _denoiser;

            var s = RelaxSettings._default;
            s.diffuseMaxAccumulatedFrameNum      = fi.diffuseMaxAccumulatedFrameNum  > 0 ? fi.diffuseMaxAccumulatedFrameNum  : 30;
            s.specularMaxAccumulatedFrameNum     = fi.specularMaxAccumulatedFrameNum > 0 ? fi.specularMaxAccumulatedFrameNum : 30;
            s.diffuseMaxFastAccumulatedFrameNum  = fi.diffuseMaxFastAccumulatedFrameNum  > 0 ? fi.diffuseMaxFastAccumulatedFrameNum  : 6;
            s.specularMaxFastAccumulatedFrameNum = fi.specularMaxFastAccumulatedFrameNum > 0 ? fi.specularMaxFastAccumulatedFrameNum : 6;
            s.checkerboardMode              = fi.checkerboardMode;
            s.hitDistanceReconstructionMode = fi.hitDistanceReconstructionMode;
            s.diffusePrepassBlurRadius      = fi.diffusePrepassBlurRadius  > 0f ? fi.diffusePrepassBlurRadius  : 30.0f;
            s.specularPrepassBlurRadius     = fi.specularPrepassBlurRadius > 0f ? fi.specularPrepassBlurRadius : 50.0f;
            s.fastHistoryClampingSigmaScale = fi.fastHistoryClampingSigmaScale > 0f ? fi.fastHistoryClampingSigmaScale : 2.0f;
            s.diffusePhiLuminance           = fi.diffusePhiLuminance  > 0f ? fi.diffusePhiLuminance  : 2.0f;
            s.specularPhiLuminance          = fi.specularPhiLuminance > 0f ? fi.specularPhiLuminance : 1.0f;
            s.enableAntiFirefly             = fi.enableAntiFirefly;
            entry.Write(s);

            return StoreAndGetPtr(data, fi.common.frameIndex);
        }
    }
}