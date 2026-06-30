namespace PathTracing
{
    public class NativeRtxptSettingGUI : SettingGUI<NativeRtxptFeature, NativeRtxptSetting>
    {
        protected override string GetSettingName()
        {
            return "Native RTXPT";
        }

        protected override object GetSettingValue()
        {
            return _feature.setting;
        }

        protected override void DeawOtherGUI()
        {
        }
    }
}
