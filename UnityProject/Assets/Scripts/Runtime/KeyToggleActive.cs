using UnityEngine;
using UnityEngine.Serialization;

namespace Runtime
{
    public class KeyToggleActive : MonoBehaviour
    {
        [SerializeField]
        private KeyCode _key = KeyCode.Space;

        public GameObject target;

        private void Update()
        {
            if (Input.GetKeyDown(_key))
            {
                target.SetActive(!target.activeSelf);
            }
        }
    }
}