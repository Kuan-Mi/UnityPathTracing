#ifndef RAB_SPATIAL_HELPERS_HLSLI
#define RAB_SPATIAL_HELPERS_HLSLI

// This function is called in the spatial resampling passes to make sure that 
// the samples actually land on the screen and not outside of its boundaries.
// It can clamp the position or reflect it about the nearest screen edge.
// The simplest implementation will just return the input pixelPosition.

// 此函数在空间重采样过程中调用，以确保采样点实际落在屏幕上，而不是超出屏幕边界。
// 它可以限制位置，也可以将其沿最近的屏幕边缘反射。最简单的实现方式是直接返回输入像素的位置。
int2 RAB_ClampSamplePositionIntoView(int2 pixelPosition, bool previousFrame)
{
    return clamp(pixelPosition, 0, int2(g_Const.view.viewportSize) - 1);
}

bool RAB_ValidateGISampleWithJacobian(inout float jacobian)
{
    return true;
}

#endif // RAB_SPATIAL_HELPERS_HLSLI
