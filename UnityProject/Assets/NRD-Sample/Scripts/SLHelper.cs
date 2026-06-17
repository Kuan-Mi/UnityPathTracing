using UnityEngine;

namespace PathTracing
{
    public class SLHelper : MonoBehaviour
    {
        [SerializeField]
        private KeyCode _key = KeyCode.G;


        private void Update()
        {
            // if (Input.GetKeyDown(_key))
            // {
            //     bool on = StreamlineFrameDriver.IsFrameGenerationOn();
            //     StreamlineFrameDriver.SetFrameGeneration(!on);
            // }
        }
    }
}