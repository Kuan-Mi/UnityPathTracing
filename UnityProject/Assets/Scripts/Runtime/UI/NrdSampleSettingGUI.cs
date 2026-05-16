using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class NrdSampleSettingGUI : SettingGUI<NativeNrdFeature,NativeNrdSampleSetting>
    {
        protected override string GetSettingName()
        {
            return "Native NRD Sample";
        }

        protected override object GetSettingValue()
        {
            return _feature.setting;
        }
    }
}