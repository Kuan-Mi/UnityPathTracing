using UnityEngine;

[ExecuteInEditMode] // 让脚本在编辑模式下也能生效
public class SetCameraNear : MonoBehaviour
{
    public float nearValue = 0.001f;

    void Update()
    {
        Camera cam = GetComponent<Camera>();
        if (cam != null)
        {
            cam.nearClipPlane = nearValue;
        }
    }
}