#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Imports RTXPT SampleGame ship props (<c>*.prop.json</c>) into the scene as flying RX6 ships.
    ///
    /// For each prop it instantiates the <c>rx6</c> model, bakes in the <c>RX6SpaceShip.model.json</c>
    /// model pose, assigns the <see cref="RtxptMaterial"/> slot assets by name (prefix <c>rx6</c>),
    /// overrides the 4 analytic lights (2 blue spots + red/green blobs) with the authoritative
    /// model.json values, and attaches a <see cref="PropFlightPath"/> filled with the keyframed flight
    /// path (converted Falcor→Unity via glTFast negate-X). Lights ride the moving transform and are
    /// picked up by the existing per-frame light collection; the proxy meshes resolve to their nearest
    /// ancestor Light automatically (see <see cref="RtxptAnalyticLightProxy"/> / NativeRtxptGPUScene).
    ///
    /// Open via  RTXPT ▸ Import Ship Props…
    /// </summary>
    public class RtxptShipImporter : EditorWindow
    {
        private const string kGltfPath       = "Assets/RTXPTAssets/Models/RX6/rx6.gltf";
        private const string kMaterialFolder = "Assets/RTXPTAssets/Materials";
        private const string kMaterialPrefix = "rx6";
        private const string kShipsRootName  = "Ships";

        // RX6SpaceShip.model.json modelPose (model-local offset applied on top of the prop transform).
        private static readonly Vector3    kModelPosePosition = new(0f, -0.08f, 0f);
        private static readonly Quaternion kModelPoseRotation = Quaternion.Euler(0f, -90f, 0f); // euler Y = -1.5708 rad; negated so the ship's nose faces the flight direction
        private const float                kModelPoseScale    = 0.04f;

        private string _propsFolder = @"F:\RTXPT\Assets\SampleGame\bistro-programmer-art.scene\props";

        private static readonly string[] kDefaultProps = { "SHIP_racer_0", "SHIP_random_0", "SHIP_racer_5" };

        private string  _report = "";
        private Vector2 _scroll;

        [MenuItem("RTXPT/Import Ship Props…")]
        public static void Open() =>
            GetWindow<RtxptShipImporter>("Import Ship Props").minSize = new Vector2(460, 320);

        private void OnGUI()
        {
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("RX6 Ship Prop Importer", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Instantiates the rx6 model per prop, wires materials + the 4 analytic lights, and " +
                "attaches a looping PropFlightPath. Model: " + kGltfPath, MessageType.None);

            EditorGUILayout.Space(6);
            EditorGUILayout.BeginHorizontal();
            _propsFolder = EditorGUILayout.TextField("Props Folder", _propsFolder);
            if (GUILayout.Button("…", GUILayout.Width(26)))
            {
                string picked = EditorUtility.OpenFolderPanel("Select RTXPT props folder", _propsFolder, "");
                if (!string.IsNullOrEmpty(picked)) _propsFolder = picked;
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("Subset (validation):", EditorStyles.miniBoldLabel);
            EditorGUILayout.LabelField("  " + string.Join(", ", kDefaultProps), EditorStyles.miniLabel);

            EditorGUILayout.Space(6);
            if (GUILayout.Button($"Import Subset ({kDefaultProps.Length} ships)", GUILayout.Height(28)))
                ImportNamed(kDefaultProps);

            if (GUILayout.Button("Import All Ships", GUILayout.Height(28)))
                ImportAll();

            if (GUILayout.Button("Import Selected *.prop.json…", GUILayout.Height(22)))
            {
                string file = EditorUtility.OpenFilePanel("Select a *.prop.json", _propsFolder, "json");
                if (!string.IsNullOrEmpty(file)) ImportFiles(new[] { file });
            }

            if (!string.IsNullOrEmpty(_report))
            {
                EditorGUILayout.Space(8);
                _scroll = EditorGUILayout.BeginScrollView(_scroll, GUILayout.ExpandHeight(true));
                EditorGUILayout.HelpBox(_report, MessageType.None);
                EditorGUILayout.EndScrollView();
            }
        }

        private void ImportAll()
        {
            if (!Directory.Exists(_propsFolder))
            {
                _report = $"Props folder not found: {_propsFolder}";
                return;
            }

            var files = Directory.GetFiles(_propsFolder, "SHIP_*.prop.json", SearchOption.TopDirectoryOnly);
            Array.Sort(files, StringComparer.OrdinalIgnoreCase);
            if (files.Length == 0)
            {
                _report = $"No SHIP_*.prop.json files found in: {_propsFolder}";
                return;
            }
            ImportFiles(files);
        }

        private void ImportNamed(IEnumerable<string> names)
        {
            var files = new List<string>();
            foreach (var n in names)
                files.Add(Path.Combine(_propsFolder, n + ".prop.json"));
            ImportFiles(files);
        }

        private void ImportFiles(IEnumerable<string> files)
        {
            var gltf = AssetDatabase.LoadAssetAtPath<GameObject>(kGltfPath);
            if (gltf == null)
            {
                _report = $"Model not found: {kGltfPath}";
                return;
            }

            var matMap = BuildMaterialMap();
            var shipsRoot = FindOrCreateShipsRoot();

            var sb = new StringBuilder();
            int ok = 0;
            foreach (var file in files)
            {
                try
                {
                    string name = Path.GetFileName(file).Replace(".prop.json", "").Replace(".json", "");
                    GameObject ship = ImportOne(file, name, gltf, matMap, shipsRoot);
                    if (ship != null) { ok++; sb.AppendLine($"✓ {name}"); }
                }
                catch (Exception ex)
                {
                    sb.AppendLine($"✗ {Path.GetFileName(file)} — {ex.Message}");
                }
            }

            EditorUtility.SetDirty(shipsRoot);
            UnityEditor.SceneManagement.EditorSceneManager.MarkAllScenesDirty();
            _report = $"Imported {ok} ship(s) under '{kShipsRootName}'.\n\n" + sb;
        }

        private GameObject ImportOne(
            string file, string name, GameObject gltf,
            Dictionary<string, RtxptMaterial> matMap, GameObject shipsRoot)
        {
            string json = File.ReadAllText(file);
            var root = MiniJson.Parse(json) as Dictionary<string, object>;
            if (root == null) throw new Exception("root is not a JSON object");

            float playbackSpeed = root.TryGetValue("animPlaybackSpeed", out var ps) ? (float)(double)ps : 1f;
            if (!(root.TryGetValue("animation", out var animObj) && animObj is List<object> anim) || anim.Count == 0)
                throw new Exception("missing/empty 'animation' array");

            int n = anim.Count;
            var times = new float[n];
            var positions = new Vector3[n];
            var rotations = new Quaternion[n];

            for (int i = 0; i < n; i++)
            {
                var key = (Dictionary<string, object>)anim[i];
                times[i]     = (float)(double)key["keytime"];
                positions[i] = ConvPos((List<object>)key["translation"]);
                rotations[i] = ConvRot((List<object>)key["rotation"]);
            }

            // --- Build the ship: outer (path-driven) → model instance (modelPose baked in) ---
            var outer = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(outer, "Import Ship Prop");
            outer.transform.SetParent(shipsRoot.transform, worldPositionStays: false);

            // Pose node carries the model.json modelPose; the gltf instance keeps its own intrinsic
            // transform (scale 0.1 etc.) so the two compose correctly regardless of glTFast wrapping.
            var poseNode = new GameObject("Pose");
            poseNode.transform.SetParent(outer.transform, worldPositionStays: false);
            poseNode.transform.localPosition = kModelPosePosition;
            poseNode.transform.localRotation = kModelPoseRotation;
            poseNode.transform.localScale    = Vector3.one * kModelPoseScale;

            var instance = (GameObject)PrefabUtility.InstantiatePrefab(gltf);
            instance.name = "Model";
            instance.transform.SetParent(poseNode.transform, worldPositionStays: false);

            AssignMaterials(instance.transform, matMap);
            SetupLights(instance.transform);

            var path = outer.AddComponent<PropFlightPath>();
            path.times = times;
            path.positions = positions;
            path.rotations = rotations;
            path.duration = times[n - 1];
            path.playbackSpeed = playbackSpeed;
            path.loop = true;

            return outer;
        }

        // ---- Materials: add RtxptRenderer + assign slots by matching Unity material name to rx6.<name> ----
        private static void AssignMaterials(Transform root, Dictionary<string, RtxptMaterial> matMap)
        {
            foreach (var rend in root.GetComponentsInChildren<Renderer>(includeInactive: true))
            {
                var mats = rend.sharedMaterials ?? Array.Empty<Material>();
                var mf   = rend.GetComponent<MeshFilter>();
                int slotCount = mf != null && mf.sharedMesh != null ? mf.sharedMesh.subMeshCount : mats.Length;
                if (slotCount == 0) continue;

                var rr = rend.GetComponent<RtxptRenderer>() ?? rend.gameObject.AddComponent<RtxptRenderer>();
                while (rr.Slots.Count < slotCount) rr.Slots.Add(null);
                if (rr.Slots.Count > slotCount) rr.Slots.RemoveRange(slotCount, rr.Slots.Count - slotCount);

                for (int s = 0; s < slotCount; s++)
                {
                    Material mat = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[^1] : null);
                    if (mat == null) continue;
                    string key = $"{kMaterialPrefix}.{mat.name}";
                    if (matMap.TryGetValue(key, out var asset))
                        rr.Slots[s] = asset;
                }

                rr.RebuildGroups();
            }
        }

        // ---- Lights: override the imported KHR_lights_punctual lights with the model.json values ----
        private static void SetupLights(Transform root)
        {
            Color blue = new(0.80f, 0.90f, 1.00f);
            // outer cone half-angle 12.5° / inner 11.875° (glTF/model.json half-angles → Unity full angles).
            SetSpot(root, "SpotLeft",  blue, 50f, outerHalfDeg: 12.5f, innerHalfDeg: 11.875f);
            SetSpot(root, "SpotRight", blue, 50f, outerHalfDeg: 12.5f, innerHalfDeg: 11.875f);

            SetPoint(root, "BlobLeft",  new Color(1.00f, 0.01f, 0.01f), 0.002f);
            SetPoint(root, "BlobRight", new Color(0.01f, 1.00f, 0.01f), 0.002f);
        }

        private static void SetSpot(Transform root, string node, Color color, float intensity, float outerHalfDeg, float innerHalfDeg)
        {
            var light = GetOrAddLight(root, node);
            if (light == null)
            {
                Debug.LogWarning($"[RtxptShipImporter] spot light node '{node}' not found under {root.name}");
                return;
            }
            light.type           = LightType.Spot;
            light.color          = color;
            light.intensity      = intensity;
            light.spotAngle      = outerHalfDeg * 2f;
            light.innerSpotAngle = innerHalfDeg * 2f;
            light.shadows        = LightShadows.None; // path traced
            // The model pose was negated (Euler Y -90) to face the nose forward, which also rotated
            // these child spot nodes 180°. Spin them back so the beams point along the flight direction.
            light.transform.localRotation *= Quaternion.Euler(0f, 180f, 0f);
        }

        private static void SetPoint(Transform root, string node, Color color, float intensity)
        {
            var light = GetOrAddLight(root, node);
            if (light == null)
            {
                Debug.LogWarning($"[RtxptShipImporter] point light node '{node}' not found under {root.name}");
                return;
            }
            light.type      = LightType.Point;
            light.color     = color;
            light.intensity = intensity;
            light.shadows   = LightShadows.None;
        }

        private static Light GetOrAddLight(Transform root, string node)
        {
            var t = FindDeep(root, node);
            if (t == null)
            {
                Debug.LogWarning($"[RtxptShipImporter] light node '{node}' not found under {root.name}");
                return null;
            }
            
            Debug.Log($"[RtxptShipImporter] GetOrAddLight  for (t={t.name}, node={node})");
            
            var light = t.GetComponent<Light>();
            if (light != null) return light;
            
            return t.gameObject.AddComponent<Light>();
        }

        // ---- Helpers ----

        private static Dictionary<string, RtxptMaterial> BuildMaterialMap()
        {
            var map = new Dictionary<string, RtxptMaterial>(StringComparer.OrdinalIgnoreCase);
            foreach (var guid in AssetDatabase.FindAssets("t:RtxptMaterial", new[] { kMaterialFolder }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                var asset = AssetDatabase.LoadAssetAtPath<RtxptMaterial>(path);
                if (asset == null) continue;
                string keyName = Path.GetFileNameWithoutExtension(path);
                map.TryAdd(keyName, asset);
            }
            return map;
        }

        private GameObject FindOrCreateShipsRoot()
        {
            var existing = GameObject.Find("/" + kShipsRootName);
            if (existing != null) return existing;
            var go = new GameObject(kShipsRootName);
            Undo.RegisterCreatedObjectUndo(go, "Create Ships Root");
            return go;
        }

        private static Transform FindDeep(Transform root, string name)
        {
            if (root.name == name) return root;
            foreach (Transform child in root)
            {
                var found = FindDeep(child, name);
                if (found != null) return found;
            }
            return null;
        }

        // Falcor → Unity (glTFast negate-X), verified against the static Chess instance.
        private static Vector3 ConvPos(List<object> v)
            => new(-(float)(double)v[0], (float)(double)v[1], (float)(double)v[2]);

        private static Quaternion ConvRot(List<object> v)
            => new((float)(double)v[0], -(float)(double)v[1], -(float)(double)v[2], (float)(double)v[3]);
    }

    /// <summary>Minimal allocation-tolerant JSON parser (objects, arrays, numbers as double, strings,
    /// bool, null) — enough for RTXPT prop files; the project has no Newtonsoft dependency.</summary>
    internal static class MiniJson
    {
        public static object Parse(string s) { int i = 0; return ParseValue(s, ref i); }

        private static object ParseValue(string s, ref int i)
        {
            SkipWs(s, ref i);
            char c = s[i];
            switch (c)
            {
                case '{': return ParseObject(s, ref i);
                case '[': return ParseArray(s, ref i);
                case '"': return ParseString(s, ref i);
                case 't': i += 4; return true;
                case 'f': i += 5; return false;
                case 'n': i += 4; return null;
                default:  return ParseNumber(s, ref i);
            }
        }

        private static Dictionary<string, object> ParseObject(string s, ref int i)
        {
            var d = new Dictionary<string, object>();
            i++; // {
            SkipWs(s, ref i);
            if (s[i] == '}') { i++; return d; }
            while (true)
            {
                SkipWs(s, ref i);
                string k = ParseString(s, ref i);
                SkipWs(s, ref i);
                i++; // :
                d[k] = ParseValue(s, ref i);
                SkipWs(s, ref i);
                if (s[i++] == '}') break; // else ','
            }
            return d;
        }

        private static List<object> ParseArray(string s, ref int i)
        {
            var a = new List<object>();
            i++; // [
            SkipWs(s, ref i);
            if (s[i] == ']') { i++; return a; }
            while (true)
            {
                a.Add(ParseValue(s, ref i));
                SkipWs(s, ref i);
                if (s[i++] == ']') break; // else ','
            }
            return a;
        }

        private static string ParseString(string s, ref int i)
        {
            i++; // opening quote
            var sb = new StringBuilder();
            while (true)
            {
                char c = s[i++];
                if (c == '"') break;
                if (c == '\\')
                {
                    char e = s[i++];
                    switch (e)
                    {
                        case '"': sb.Append('"'); break;
                        case '\\': sb.Append('\\'); break;
                        case '/': sb.Append('/'); break;
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case 'b': sb.Append('\b'); break;
                        case 'f': sb.Append('\f'); break;
                        case 'u': sb.Append((char)Convert.ToInt32(s.Substring(i, 4), 16)); i += 4; break;
                    }
                }
                else sb.Append(c);
            }
            return sb.ToString();
        }

        private static object ParseNumber(string s, ref int i)
        {
            int start = i;
            while (i < s.Length)
            {
                char c = s[i];
                if (c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || (c >= '0' && c <= '9')) i++;
                else break;
            }
            return double.Parse(s.Substring(start, i - start), CultureInfo.InvariantCulture);
        }

        private static void SkipWs(string s, ref int i)
        {
            while (i < s.Length)
            {
                char c = s[i];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i++;
                else break;
            }
        }
    }
}
#endif
