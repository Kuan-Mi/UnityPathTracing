using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteInEditMode]
public class FalcorCamera : MonoBehaviour
{
    [Header("Falcor LookAt Parameters")]
    [Tooltip("Eye position in Falcor right-hand coordinate space")]
    public Vector3 falcorEye    = new Vector3(-1.658f,  1.577f,  1.69f);

    [Tooltip("Target position in Falcor right-hand coordinate space")]
    public Vector3 falcorTarget = new Vector3(-0.9645f, 1.2672f, 1.0396f);

    [Header("Up Vector")]
    public Vector3 upVector = Vector3.up;

    void Update()
    {
        ApplyLookAt();
    }

    void ApplyLookAt()
    {
        // Falcor (right-hand, +Z toward viewer) → Unity (left-hand, +Z away from viewer)
        // 只需对 Z 轴取反
        Vector3 unityEye    = new Vector3( falcorEye.x,     falcorEye.y,    -falcorEye.z);
        Vector3 unityTarget = new Vector3( falcorTarget.x,  falcorTarget.y, -falcorTarget.z);

        transform.position = unityEye;
        transform.LookAt(unityTarget, upVector);
    }

#if UNITY_EDITOR
    // 编辑器下 Inspector 修改参数时实时刷新
    void OnValidate()
    {
        ApplyLookAt();
    }
#endif
}