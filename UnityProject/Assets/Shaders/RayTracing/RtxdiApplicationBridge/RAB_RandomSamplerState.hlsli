#ifndef RTXDI_RAB_RANDOM_SAMPLER_STATE_HLSLI
#define RTXDI_RAB_RANDOM_SAMPLER_STATE_HLSLI

#include "../HelperFunctions.hlsli"

// 存储随机数生成器 (RNG) 的可变状态，即随机种子、采样索引以及生成随机数所需的其他任何信息。
// 应用程序需要自行初始化随机采样器，并将状态传递给采样函数。
// 采样函数随后会将 RNG 状态传递给 RAB_GetNextRandom ，每次重采样函数调用都会多次传递该状态。
typedef RandomSamplerState RAB_RandomSamplerState;

RAB_RandomSamplerState RAB_InitRandomSampler(uint2 index, uint pass)
{
    return initRandomSampler(index, g_Const.frameIndex + pass * 13);
}

// 从提供的随机数生成器 (RNG) 状态返回下一个随机数。这些数字必须介于 0 和 1 之间，因此对于每个返回的 x ， 0.0 <= x < 1.0 。
// 使用高质量的随机数生成器对于 RTXDI 至关重要，否则可能会出现各种瑕疵。
// 例如，当屏幕大部分区域的光照结果“卡住”在某个特定光源上时，就会出现这种瑕疵，这是由于随机数的随机性不足造成的。
float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return sampleUniformRng(rng);
}

#endif // RTXDI_RAB_RANDOM_SAMPLER_STATE_HLSLI
