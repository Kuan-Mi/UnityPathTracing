using UnityEngine;
using UnityEngine.InputSystem;

namespace PathTracing
{
    /// <summary>
    /// Mouse picker for the RTXPT debug pixel — Unity equivalent of the C++ right-click pick
    /// (Sample.cpp m_pick / SampleUIData::DebugPixel). Add it to any GameObject in the scene.
    ///
    ///   Game view (play mode):  Ctrl + right-click (configurable below).
    ///   Scene view (edit OR play mode): Ctrl + right-click directly in the Scene view.
    ///
    /// The pick converts the mouse position to the camera's render-resolution coordinates
    /// (debug-pixel coords are pre-upscale) and writes setting.debugPixelX/Y on the feature.
    /// Optionally turns on continuous feedback so the readout/lines appear immediately.
    /// </summary>
    [ExecuteAlways]
    public class RtxptDebugPixelPicker : MonoBehaviour
    {
        [Tooltip("RTXPT renderer feature whose setting.debugPixel is written. Auto-found when left empty.")]
        public NativeRtxptFeature feature;

        [Tooltip("Mouse button used for picking (0 = left, 1 = right, 2 = middle).")]
        public int pickButton = 1;

        [Tooltip("Require Ctrl to be held — avoids clashing with camera navigation.")]
        public bool requireCtrl = true;

        [Tooltip("Continuously follow the mouse instead of waiting for a click (still gated by Ctrl " +
                 "when 'Require Ctrl' is on — hold Ctrl and move the mouse to scrub the debug pixel).")]
        public bool followMouse = false;

        [Tooltip("Turn on 'Continuous feedback' automatically when picking.")]
        public bool enableContinuousFeedback = true;

        [Tooltip("Log the picked render-resolution pixel coordinates to the console.")]
        public bool logPicks = true;

        private NativeRtxptFeature Feature
        {
            get
            {
                if (feature == null)
                {
                    var all = Resources.FindObjectsOfTypeAll<NativeRtxptFeature>();
                    if (all.Length > 0) feature = all[0];
                }
                return feature;
            }
        }

        // ── Game view (play mode) ─────────────────────────────────────────────
        private void Update()
        {
            if (!Application.isPlaying) return;

            var keyboard = Keyboard.current;
            bool ctrlHeld = keyboard?.leftCtrlKey.isPressed == true || keyboard?.rightCtrlKey.isPressed == true;
            bool trigger = followMouse
                ? (!requireCtrl || ctrlHeld)                                       // hover/scrub mode
                : IsMouseButtonPressedThisFrame(pickButton) && (!requireCtrl || ctrlHeld); // click mode
            if (!trigger) return;

            var cam = Camera.main;
            if (cam == null) return;

            // Input.mousePosition is bottom-left origin; debug-pixel coords are top-left.
            Vector2 mp = Mouse.current?.position.ReadValue() ?? Vector2.zero;
            var uv = new Vector2(mp.x / cam.pixelWidth, 1f - mp.y / cam.pixelHeight);
            Pick(cam, uv, log: !followMouse);
        }

        private static bool IsMouseButtonPressedThisFrame(int button)
        {
            var mouse = Mouse.current;
            if (mouse == null) return false;

            return button switch
            {
                0 => mouse.leftButton.wasPressedThisFrame,
                1 => mouse.rightButton.wasPressedThisFrame,
                2 => mouse.middleButton.wasPressedThisFrame,
                3 => mouse.forwardButton.wasPressedThisFrame,
                4 => mouse.backButton.wasPressedThisFrame,
                _ => false
            };
        }

        private void Pick(Camera cam, Vector2 uvTopLeft, bool log = true)
        {
            var f = Feature;
            if (f == null || f.setting == null) return;
            if (uvTopLeft.x < 0f || uvTopLeft.x >= 1f || uvTopLeft.y < 0f || uvTopLeft.y >= 1f) return;

            if (!f.TrySetDebugPixelFromViewport(cam, uvTopLeft)) return;

            if (enableContinuousFeedback)
                f.setting.continuousDebugFeedback = true;
            if (logPicks && log)
                Debug.Log($"[RtxptDebugPixelPicker] debug pixel = ({f.setting.debugPixelX}, {f.setting.debugPixelY})  camera: {cam.name}");
        }

        // ── Scene view (edit + play mode) ─────────────────────────────────────
#if UNITY_EDITOR
        private void OnEnable()  => UnityEditor.SceneView.duringSceneGui += OnSceneGUI;
        private void OnDisable() => UnityEditor.SceneView.duringSceneGui -= OnSceneGUI;

        private void OnSceneGUI(UnityEditor.SceneView sceneView)
        {
            var e = Event.current;

            bool isClick = e.type == EventType.MouseDown && e.button == pickButton;
            bool isMove  = followMouse && (e.type == EventType.MouseMove || e.type == EventType.MouseDrag);
            if (!isClick && !isMove) return;
            if (requireCtrl && !e.control) return;

            var cam = sceneView.camera;
            if (cam == null) return;

            // Event.mousePosition: GUI points, top-left origin → scale to pixels.
            Vector2 mp = e.mousePosition * UnityEditor.EditorGUIUtility.pixelsPerPoint;
            var uv = new Vector2(mp.x / cam.pixelWidth, mp.y / cam.pixelHeight);

            Pick(cam, uv, log: isClick);
            if (isClick)
                e.Use(); // consume clicks only; moves must keep flowing to the scene view
        }
#endif
    }
}
