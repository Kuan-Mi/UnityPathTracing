using System;
using System.Collections.Generic;

namespace Nrd
{
    // -----------------------------------------------------------------------
    // REBLUR – diffuse / specular radiance + hit-distance denoiser
    // -----------------------------------------------------------------------
    public sealed class ReblurDenoiser : NrdDenoiser
    {
        private readonly Denoiser _denoiser;

        private static readonly HashSet<Denoiser> ValidDenoisers = new()
        {
            Denoiser.REBLUR_DIFFUSE,
            Denoiser.REBLUR_DIFFUSE_OCCLUSION,
            Denoiser.REBLUR_DIFFUSE_SH,
            Denoiser.REBLUR_SPECULAR,
            Denoiser.REBLUR_SPECULAR_OCCLUSION,
            Denoiser.REBLUR_SPECULAR_SH,
            Denoiser.REBLUR_DIFFUSE_SPECULAR,
            Denoiser.REBLUR_DIFFUSE_SPECULAR_OCCLUSION,
            Denoiser.REBLUR_DIFFUSE_SPECULAR_SH,
            Denoiser.REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION,
        };

        public ReblurDenoiser(string camName, Denoiser denoiser)
            : base(camName, new NrdDenoiserDesc(denoiser))
        {
            if (!ValidDenoisers.Contains(denoiser))
                throw new ArgumentException(
                    $"ReblurNrdDenoiser requires a REBLUR_* denoiser, got {denoiser}.", nameof(denoiser));
            _denoiser = denoiser;
        }

        public unsafe IntPtr  GetInteropDataPtr(CommonSettings common, ReblurSettings settings)
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