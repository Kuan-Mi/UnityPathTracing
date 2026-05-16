using System;

namespace Nrd
{

    // -----------------------------------------------------------------------
    // SIGMA – shadow / translucency denoiser
    // -----------------------------------------------------------------------
    public sealed class SigmaDenoiser : NrdDenoiser
    {
        private readonly Denoiser _denoiser;

        public SigmaDenoiser(string camName, Denoiser denoiser)
            : base(camName, new NrdDenoiserDesc(denoiser))
        {
            if (denoiser != Denoiser.SIGMA_SHADOW && denoiser != Denoiser.SIGMA_SHADOW_TRANSLUCENCY)
                throw new ArgumentException(
                    $"SigmaNrdDenoiser requires a SIGMA_* denoiser, got {denoiser}.", nameof(denoiser));
            _denoiser = denoiser;
        }

        public unsafe IntPtr GetInteropDataPtr(CommonSettings common, SigmaSettings settings)
        {
            var data = NrdFrameData._default;
            data.instanceId     = _nrdInstanceId;
            data.width          = common.resourceSize[0];
            data.height         = common.resourceSize[1];
            data.commonSettings = common;

            data.denoiserCount = 1;
            ref var entry = ref NrdFrameData.GetEntry(ref data, 0);
            entry.identifier = 0;
            entry.denoiser   = _denoiser;
            entry.Write(settings);

            return StoreAndGetPtr(data, common.frameIndex);
        }
    }

}