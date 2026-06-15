using UnityEngine;

namespace PathTracing
{
    [CreateAssetMenu(fileName = "RtxdiSettingPreset", menuName = "PathTracing/Rtxdi Setting Preset")]
    public class RtxdiSettingPreset : ScriptableObject
    {
        public RtxdiSetting setting = new RtxdiSetting();
    }
}
