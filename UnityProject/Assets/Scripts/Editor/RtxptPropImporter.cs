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
    /// Imports RTXPT SampleGame props (<c>*.prop.json</c>) into the scene, driven entirely by the
    /// matching <c>&lt;modelName&gt;.model.json</c> (ships, glass, lights, ConvergenceTest, …).
    ///
    /// For each prop it reads <c>modelName</c>, loads <c>models/&lt;modelName&gt;.model.json</c> for the
    /// scene model + pose + analytic lights, instantiates the corresponding glTFast model under a Pose
    /// node carrying the model.json <c>modelPose</c>, assigns the <see cref="RtxptMaterial"/> slot assets
    /// by name (prefix = <c>sceneModelName</c>), recreates any model.json lights, and finally places the
    /// prop either via a keyframed <see cref="PropFlightPath"/> (<c>animation</c>) or a static
    /// <c>startPose</c>. Falcor→Unity conversion is the glTFast negate-X convention.
    ///
    /// Open via  RTXPT ▸ Import Props…
    /// </summary>
    public class RtxptPropImporter : EditorWindow
    {
        private const string kModelsRoot     = "Assets/RTXPTAssets/Models";
        private const string kMaterialFolder = "Assets/RTXPTAssets/Materials";
        private const string kPropsRootName  = "Props";

        private string _propsFolder = @"C:\Data\RTXPT\Assets\SampleGame\bistro-programmer-art.scene\props";

        // A handful of props (one of each shape) for quick validation.
        private static readonly string[] kDefaultProps =
            { "SHIP_racer_0", "SHIP_random_1", "GlassSphere_1", "OrbLight_0", "SpotLight_0", "ConvergenceTest_0" };

        private string  _report = "";
        private Vector2 _scroll;

        [MenuItem("RTXPT/Import Props…")]
        public static void Open() =>
            GetWindow<RtxptPropImporter>("Import Props").minSize = new Vector2(460, 320);

        private void OnGUI()
        {
            EditorGUILayout.Space(6);
            EditorGUILayout.LabelField("RTXPT Prop Importer", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Imports any *.prop.json, driven by its <modelName>.model.json (model, pose, materials, " +
                "lights). The model.json files are read from the 'models' folder next to the props folder.",
                MessageType.None);

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
            if (GUILayout.Button($"Import Subset ({kDefaultProps.Length} props)", GUILayout.Height(28)))
                ImportNamed(kDefaultProps);

            if (GUILayout.Button("Import All Props", GUILayout.Height(28)))
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

            var files = Directory.GetFiles(_propsFolder, "*.prop.json", SearchOption.TopDirectoryOnly);
            Array.Sort(files, StringComparer.OrdinalIgnoreCase);
            if (files.Length == 0)
            {
                _report = $"No *.prop.json files found in: {_propsFolder}";
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
            // model.json files live in the 'models' folder next to the props folder.
            string modelsJsonFolder = Path.Combine(Path.GetDirectoryName(_propsFolder) ?? "", "models");
            if (!Directory.Exists(modelsJsonFolder))
            {
                _report = $"models folder not found next to props: {modelsJsonFolder}";
                return;
            }

            var matMap = BuildMaterialMap();
            var modelDefs = new Dictionary<string, ModelDef>(StringComparer.OrdinalIgnoreCase);
            var propsRoot = FindOrCreateRoot();

            var sb = new StringBuilder();
            int ok = 0;
            foreach (var file in files)
            {
                try
                {
                    string name = Path.GetFileName(file).Replace(".prop.json", "").Replace(".json", "");
                    GameObject prop = ImportOne(file, name, modelsJsonFolder, modelDefs, matMap, propsRoot);
                    if (prop != null) { ok++; sb.AppendLine($"✓ {name}"); }
                }
                catch (Exception ex)
                {
                    // Log the full exception so the stack trace + file:line land in the Console / Editor.log.
                    Debug.LogException(ex);
                    sb.AppendLine($"✗ {Path.GetFileName(file)} — {ex.Message}");
                }
            }

            EditorUtility.SetDirty(propsRoot);
            UnityEditor.SceneManagement.EditorSceneManager.MarkAllScenesDirty();
            _report = $"Imported {ok} prop(s) under '{kPropsRootName}'.\n\n" + sb;
        }

        private GameObject ImportOne(
            string file, string name, string modelsJsonFolder,
            Dictionary<string, ModelDef> modelDefs,
            Dictionary<string, RtxptMaterial> matMap, GameObject propsRoot)
        {
            var root = MiniJson.Parse(File.ReadAllText(file)) as Dictionary<string, object>;
            if (root == null) throw new Exception("root is not a JSON object");

            if (!(root.TryGetValue("modelName", out var mn) && mn is string modelName))
                throw new Exception("prop has no 'modelName'");

            var model = GetModelDef(modelName, modelsJsonFolder, modelDefs);
            var gltf = AssetDatabase.LoadAssetAtPath<GameObject>(model.GltfAssetPath);
            if (gltf == null)
                throw new Exception($"model '{model.SceneModelName}' not imported at {model.GltfAssetPath} " +
                                    "(let Unity reimport the .gltf, then retry)");

            float playbackSpeed = root.TryGetValue("animPlaybackSpeed", out var ps) ? (float)(double)ps : 1f;

            // A prop is either animated (keyframed 'animation') or static (a single 'startPose',
            // which may also carry a non-unit scale, e.g. SHIP_random_1..3 at 6×).
            bool hasAnim = root.TryGetValue("animation", out var animObj)
                           && animObj is List<object> a && a.Count > 0;
            var anim = hasAnim ? (List<object>)animObj : null;
            Dictionary<string, object> startPose = null;
            if (!hasAnim)
            {
                if (!(root.TryGetValue("startPose", out var spObj) && spObj is Dictionary<string, object> sp))
                    throw new Exception("prop has neither a keyframed 'animation' nor a static 'startPose'");
                startPose = sp;
            }

            // --- Build: outer (path-driven or static) → Pose (model.json modelPose) → model instance ---
            var outer = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(outer, "Import Prop");
            outer.transform.SetParent(propsRoot.transform, worldPositionStays: false);

            var poseNode = new GameObject("Pose");
            poseNode.transform.SetParent(outer.transform, worldPositionStays: false);
            poseNode.transform.localPosition = model.PosePosition;
            poseNode.transform.localRotation = model.PoseRotation;
            poseNode.transform.localScale    = Vector3.one * model.PoseScale;

            var instance = (GameObject)PrefabUtility.InstantiatePrefab(gltf);
            instance.transform.SetParent(poseNode.transform, worldPositionStays: false);

            AssignMaterials(instance.transform, model.SceneModelName, matMap);
            // Set up lights *before* renaming: a single scene-root gltf node (e.g. OrbLight) is promoted
            // to the instance root, so its node name must still be intact for the by-name lookup.
            SetupLights(instance.transform, model.Lights);
            instance.name = "Model";

            if (hasAnim)
            {
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

                var path = outer.AddComponent<PropFlightPath>();
                path.times = times;
                path.positions = positions;
                path.rotations = rotations;
                path.duration = times[n - 1];
                path.playbackSpeed = playbackSpeed;
                path.loop = true;
            }
            else
            {
                outer.transform.localPosition = ConvPos((List<object>)startPose["translation"]);
                outer.transform.localRotation = ConvRot((List<object>)startPose["rotation"]);
                if (startPose.TryGetValue("scaling", out var sc) && sc is List<object> scl)
                    outer.transform.localScale = ConvScale(scl);
            }

            return outer;
        }

        // ---- model.json: cache + parse (sceneModelName, modelPose, lights) ----

        private static ModelDef GetModelDef(
            string modelName, string modelsJsonFolder, Dictionary<string, ModelDef> cache)
        {
            if (cache.TryGetValue(modelName, out var cached)) return cached;

            string path = Path.Combine(modelsJsonFolder, modelName + ".model.json");
            if (!File.Exists(path))
                throw new Exception($"model.json not found: {path}");

            var json = MiniJson.Parse(File.ReadAllText(path)) as Dictionary<string, object>;
            if (json == null) throw new Exception($"{modelName}.model.json is not a JSON object");

            var def = new ModelDef();
            def.SceneModelName = json.TryGetValue("sceneModelName", out var smn) ? (string)smn : modelName;

            // modelPose → Pose node transform (Falcor → Unity).
            def.PosePosition = Vector3.zero;
            def.PoseRotation = Quaternion.identity;
            def.PoseScale    = 1f;
            if (json.TryGetValue("modelPose", out var mpObj) && mpObj is Dictionary<string, object> mp)
            {
                if (mp.TryGetValue("translation", out var t) && t is List<object> tl)
                    def.PosePosition = ConvPos(tl);
                if (mp.TryGetValue("euler", out var e) && e is List<object> el)
                {
                    // Falcor euler (radians) → quaternion → negate-X convention (ConvRot).
                    var fq = Quaternion.Euler(
                        (float)(double)el[0] * Mathf.Rad2Deg,
                        (float)(double)el[1] * Mathf.Rad2Deg,
                        (float)(double)el[2] * Mathf.Rad2Deg);
                    def.PoseRotation = new Quaternion(fq.x, -fq.y, -fq.z, fq.w);
                }
                if (mp.TryGetValue("scaling", out var s))
                    def.PoseScale = (float)(double)s;
            }

            // lights[] (optional): spot if it has a non-zero outerAngle, else point.
            def.Lights = new List<LightDef>();
            if (json.TryGetValue("lights", out var lObj) && lObj is List<object> lights)
            {
                foreach (var lo in lights)
                {
                    if (lo is not Dictionary<string, object> l) continue;
                    var ld = new LightDef
                    {
                        Name      = l.TryGetValue("name", out var ln) ? (string)ln : "",
                        Intensity = l.TryGetValue("intensity", out var inten) ? (float)(double)inten : 1f,
                        Color     = l.TryGetValue("color", out var c) && c is List<object> cl
                                        ? new Color((float)(double)cl[0], (float)(double)cl[1], (float)(double)cl[2])
                                        : Color.white,
                    };
                    float outer = l.TryGetValue("outerAngle", out var oa) ? (float)(double)oa : 0f;
                    float inner = l.TryGetValue("innerAngle", out var ia) ? (float)(double)ia : 0f;
                    ld.IsSpot       = outer > 0f;
                    ld.OuterHalfDeg = outer;
                    ld.InnerHalfDeg = inner;
                    def.Lights.Add(ld);
                }
            }

            def.GltfAssetPath = FindGltfAssetPath(def.SceneModelName)
                ?? throw new Exception($"no '{def.SceneModelName}.gltf' found under {kModelsRoot}");

            cache[modelName] = def;
            return def;
        }

        private sealed class ModelDef
        {
            public string SceneModelName;
            public string GltfAssetPath;
            public Vector3 PosePosition;
            public Quaternion PoseRotation;
            public float PoseScale;
            public List<LightDef> Lights;
        }

        private sealed class LightDef
        {
            public string Name;
            public Color Color;
            public float Intensity;
            public bool IsSpot;
            public float InnerHalfDeg, OuterHalfDeg;
        }

        // ---- Materials: add RtxptRenderer + assign slots by matching Unity material name ----
        private static void AssignMaterials(Transform root, string prefix, Dictionary<string, RtxptMaterial> matMap)
        {
            foreach (var rend in root.GetComponentsInChildren<Renderer>(includeInactive: true))
            {
                var mats = rend.sharedMaterials ?? Array.Empty<Material>();
                var mf   = rend.GetComponent<MeshFilter>();
                int slotCount = mf != null && mf.sharedMesh != null ? mf.sharedMesh.subMeshCount : mats.Length;
                if (slotCount == 0) continue;

                var rr = rend.GetComponent<RtxptRenderer>();
                if (rr == null) rr = rend.gameObject.AddComponent<RtxptRenderer>();
                while (rr.Slots.Count < slotCount) rr.Slots.Add(null);
                if (rr.Slots.Count > slotCount) rr.Slots.RemoveRange(slotCount, rr.Slots.Count - slotCount);

                for (int s = 0; s < slotCount; s++)
                {
                    Material mat = s < mats.Length ? mats[s] : (mats.Length > 0 ? mats[^1] : null);
                    if (mat == null) continue;
                    if (matMap.TryGetValue($"{prefix}.{mat.name}", out var asset))
                        rr.Slots[s] = asset;
                }

                rr.RebuildGroups();
            }
        }

        // ---- Lights: recreate the model.json analytic lights on their named nodes ----
        private static void SetupLights(Transform root, List<LightDef> defs)
        {
            if (defs == null) return;
            foreach (var d in defs)
            {
                var node = FindLightNode(root, d.Name);
                if (node == null)
                {
                    Debug.LogWarning($"[RtxptPropImporter] light node '{d.Name}' not found under {root.name}");
                    continue;
                }
                // Reuse any glTFast-imported KHR_lights_punctual Light on the node, else add one.
                // Use Unity's overloaded null check, not '??' (which bypasses it and can keep a fake-null).
                var light = node.GetComponent<Light>();
                if (light == null) light = node.gameObject.AddComponent<Light>();
                light.color     = d.Color;
                light.intensity = d.Intensity;
                light.shadows   = LightShadows.None; // path traced

                if (d.IsSpot)
                {
                    light.type           = LightType.Spot;
                    light.spotAngle      = d.OuterHalfDeg * 2f; // model.json stores half-angles
                    light.innerSpotAngle = d.InnerHalfDeg * 2f;
                    // glTF aims a spot down its node's -Z; a Unity Spot shines +Z. Flip the node so the
                    // beam points where the source authored it (also corrects the ship's nose-forward pose).
                    light.transform.localRotation *= Quaternion.Euler(0f, 180f, 0f);
                }
                else
                {
                    light.type = LightType.Point;
                }
            }
        }

        // Locate a light's node by name, preferring an inner node over a same-named instance/wrapper root
        // (e.g. SpotLight's inner light node vs. the file-named wrapper), then falling back to the root
        // itself (a single scene-root node, e.g. OrbLight, promoted to the instance root).
        private static Transform FindLightNode(Transform root, string name)
        {
            foreach (Transform child in root)
            {
                var found = FindDeep(child, name);
                if (found != null) return found;
            }
            return root.name == name ? root : null;
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
                map.TryAdd(Path.GetFileNameWithoutExtension(path), asset);
            }
            return map;
        }

        // Resolve "<sceneModelName>.gltf" anywhere under the Models root → an "Assets/…" path.
        private static string FindGltfAssetPath(string sceneModelName)
        {
            string absRoot = Path.Combine(Application.dataPath, "RTXPTAssets/Models");
            if (!Directory.Exists(absRoot)) return null;
            var hits = Directory.GetFiles(absRoot, sceneModelName + ".gltf", SearchOption.AllDirectories);
            if (hits.Length == 0) return null;
            string abs = hits[0].Replace('\\', '/');
            // Application.dataPath ends with "/Assets"; rebuild as a project-relative asset path.
            return "Assets" + abs.Substring(Application.dataPath.Length);
        }

        private GameObject FindOrCreateRoot()
        {
            var existing = GameObject.Find("/" + kPropsRootName);
            if (existing != null) return existing;
            var go = new GameObject(kPropsRootName);
            Undo.RegisterCreatedObjectUndo(go, "Create Props Root");
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

        // Scale is sign-invariant under the negate-X convention (only positions/handedness flip).
        private static Vector3 ConvScale(List<object> v)
            => new((float)(double)v[0], (float)(double)v[1], (float)(double)v[2]);
    }

    /// <summary>Minimal allocation-tolerant JSON parser (objects, arrays, numbers as double, strings,
    /// bool, null) — enough for RTXPT prop/model files; the project has no Newtonsoft dependency.</summary>
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
