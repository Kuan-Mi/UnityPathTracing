using UnityEngine;

namespace PathTracing
{
    public class ShowFPS : MonoBehaviour
    {
        private float deltaTime = 0.0f;
        
        
        private int frameCount = 0;
        private float startTime = 0.0f;

        void Update()
        {
            deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
            
            
            if(Time.time - startTime >= 5.0f)
            {
                float fps = frameCount / (Time.time - startTime);
                Debug.Log("TimeInMS for 5 seconds: " + (1000.0f / fps) + " ms (" + fps + " fps)");
                frameCount = 0;
                startTime = Time.time;
            }
            frameCount++;
            
            
            
            
        }

        void OnGUI()
        {
            int w = Screen.width, h = Screen.height;

            GUIStyle style = new GUIStyle();

            Rect rect = new Rect(0, 0, w, h * 2 / 100);
            style.alignment = TextAnchor.UpperLeft;
            style.fontSize = h * 2 / 100;
            style.normal.textColor = Color.white;
            float msec = deltaTime * 1000.0f;
            float fps = 1.0f / deltaTime;
            string text = string.Format("{0:0.0} ms ({1:0.} fps)", msec, fps);
            GUI.Label(rect, text, style);
        }
        
    }
}