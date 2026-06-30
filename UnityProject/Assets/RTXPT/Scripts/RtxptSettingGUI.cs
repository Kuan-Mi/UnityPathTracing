namespace PathTracing
{
    public class RtxptSettingGUI : SettingGUI<RtxptFeature, RtxptSetting>
    {
        protected override string GetSettingName()
        {
            return "RTXPT";
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
