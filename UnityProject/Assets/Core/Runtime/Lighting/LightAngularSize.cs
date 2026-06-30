using UnityEngine;

namespace PathTracing
{
    public class LightAngularSize : MonoBehaviour
    {
        [Range(0.001f, 0.1f)]
        public float angularSize;

        public float GetAngularSize()
        {
            return angularSize;
        }
    }
}