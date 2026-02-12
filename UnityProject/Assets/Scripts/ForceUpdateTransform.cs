using UnityEngine;

namespace DefaultNamespace
{
    public class ForceUpdateTransform : MonoBehaviour
    {
        private void LateUpdate()
        {
            var s = 1 + 0.01f * Mathf.Sin(Time.time * 10);
            transform.localScale = new Vector3(s, s, s);
        }
    }
}