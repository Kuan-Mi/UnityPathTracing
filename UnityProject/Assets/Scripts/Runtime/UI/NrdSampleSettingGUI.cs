using System;
using System.Collections.Generic;
using System.Reflection;
using Nrd;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public class NrdSampleSettingGUI : SettingGUI<NativeNrdFeature, NativeNrdSampleSetting>
    {
        protected override string GetSettingName()
        {
            return "Native NRD Sample";
        }

        protected override object GetSettingValue()
        {
            return _feature.setting;
        }

        protected override void DeawOtherGUI()
        {
            var setting = _feature.setting;

            if (setting.RR) return;
            DrawObjectFieldsInPlace(_feature.commonSettings, typeof(CommonSettings), "commonSettings");

            if (setting.denoiser == DenoiserType.DENOISER_REFERENCE)
            {
                DrawObjectFieldsInPlace(_feature.referenceSettings, typeof(ReferenceSettings), "referenceSettings");
            }
            else
            {
                DrawObjectFieldsInPlace(_feature.sigmaSettings, typeof(SigmaSettings), "sigmaSettings");

                if (setting.denoiser == DenoiserType.DENOISER_REBLUR)
                {
                    DrawObjectFieldsInPlace(_feature.reblurSettings, typeof(ReblurSettings), "reblurSettings");
                }
                else if (setting.denoiser == DenoiserType.DENOISER_RELAX)
                {
                    DrawObjectFieldsInPlace(_feature.relaxSettings, typeof(RelaxSettings), "relaxSettings");
                }
            }
        }
    }
}