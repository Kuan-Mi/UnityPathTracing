using System;
using UnityEngine;

namespace DefaultNamespace
{
    public class AutoMove : MonoBehaviour
    {
        public Vector3 center;
        public  float length = 5f;
        public  float duration;
        public Vector3 axis = Vector3.up;

        private void OnEnable()
        {
            center = transform.position;
        }
        
        private void Update()
        {
            float t = (Time.time % duration) / duration;
            Vector3 offset = axis.normalized * Mathf.Sin(t * Mathf.PI * 2f) * length;
            transform.position = center + offset;
        }
    }
}