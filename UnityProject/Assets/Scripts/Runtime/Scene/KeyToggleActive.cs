using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.InputSystem;

namespace PathTracing
{
    public class KeyToggleActive : MonoBehaviour
    {
        [SerializeField]
        private Key _key = Key.Space;

        public GameObject target;

        private void Update()
        {
            if (Keyboard.current?[_key].wasPressedThisFrame == true)
            {
                target.SetActive(!target.activeSelf);
            }
        }
    }
}
