namespace PathTracing
{
    // 0 showValidation     Blend Alpha
    // 1 showShadow         解码后输出阴影
    // 2 showMv             VM
    // 3 ShowNormal         解码后输出法线 转到NRD坐标系
    // 4 showOut            Blend Alpha
    // 5 showAlpha          灰度输出
    // 6 ShowRoughness      解码后输出粗糙度
    // 7 ShowRadiance       解码后RGB输出
    public enum ShowPass : int
    {
        Validation,
        Shadow,
        Mv,
        Normal,
        Out,
        Alpha,
        Roughness,
        Radiance,
        NoiseShadow,
        Dlss,
        ViewZ,
    }
}