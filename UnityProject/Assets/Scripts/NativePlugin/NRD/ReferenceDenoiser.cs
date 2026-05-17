using System;

namespace Nrd
{

    // -----------------------------------------------------------------------
    // SIGMA – shadow / translucency denoiser
    // -----------------------------------------------------------------------
    public sealed class ReferenceDenoiser : NrdDenoiser<ReferenceSettings>
    {
        public ReferenceDenoiser(string camName, Denoiser denoiser)
            : base(camName, new NrdDenoiserDesc(denoiser))
        {
            if (denoiser != Denoiser.REFERENCE)
                throw new ArgumentException(
                    $"ReferenceDenoiser requires the REFERENCE denoiser, got {denoiser}.", nameof(denoiser));
            _denoiser = denoiser;
        }
    }
}