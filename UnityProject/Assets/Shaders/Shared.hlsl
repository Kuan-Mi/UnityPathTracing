
float3 ApplyExposure( float3 Lsum )
{
    // if( gOnScreen <= SHOW_DENOISED_SPECULAR )
        Lsum *= gExposure;

    return Lsum;
}