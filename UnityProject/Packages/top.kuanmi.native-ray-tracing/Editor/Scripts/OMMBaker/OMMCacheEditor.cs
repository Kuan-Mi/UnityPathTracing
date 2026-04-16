using System;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    [CustomEditor(typeof(OMMCache))]
    public class OMMCacheEditor : Editor
    {
        // ── Preview size ───────────────────────────────────────────────────
        private const int PreviewSize = 512;

        // ── State ──────────────────────────────────────────────────────────
        private bool _showStats   = true;
        private bool _showPreview = true;
        private bool _showBake    = true;

        private string _bakeStatus  = null;
        private bool   _bakeSuccess = false;

        // UV-space preview rendering
        private PreviewRenderUtility _preview;
        private Material             _previewMatFill; // shader pass 0
        private Material             _previewMatWire; // shader pass 1
        private Mesh                 _uvMesh; // mesh with UV as position
        private ComputeBuffer        _ommIndexBuf;
        private ComputeBuffer        _ommDescBuf;
        private ComputeBuffer        _ommArrayBuf;

        // GPU data validity cache
        private OMMCache _cachedFor;

        // Display options
        private bool _optColorize    = true;
        private bool _optContour     = true;
        private bool _optWireframe   = false;
        private int  _optHighlight   = -1;
        private int  _primitiveStart = 0;
        private int  _primitiveEnd   = -1; // -1 = all

        private string _previewError = null;

        private SerializedProperty _propSourceMesh;
        private SerializedProperty _propSourceTexture;
        private SerializedProperty _propSubmeshIndex;
        private SerializedProperty _propAlphaCutoff;
        private SerializedProperty _propMaxSubdiv;
        private SerializedProperty _propDownsample;
        private SerializedProperty _propDynSubdivScale;
        private SerializedProperty _propOmmFormat;

        private void OnEnable()
        {
            _propSourceMesh      = serializedObject.FindProperty("sourceMesh");
            _propSourceTexture   = serializedObject.FindProperty("sourceTexture");
            _propSubmeshIndex    = serializedObject.FindProperty("submeshIndex");
            _propAlphaCutoff     = serializedObject.FindProperty("alphaCutoff");
            _propMaxSubdiv       = serializedObject.FindProperty("maxSubdivisionLevel");
            _propDownsample      = serializedObject.FindProperty("textureDownsampleFactor");
            _propDynSubdivScale  = serializedObject.FindProperty("dynamicSubdivisionScale");
            _propOmmFormat       = serializedObject.FindProperty("ommFormat");
        }

        public override void OnInspectorGUI()
        {
            var cache = (OMMCache)target;
            serializedObject.Update();

            // ── Bake section ───────────────────────────────────────────────
            _showBake = EditorGUILayout.BeginFoldoutHeaderGroup(_showBake, "Bake");
            if (_showBake)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.PropertyField(_propSourceMesh, new GUIContent("Mesh"));

                // Submesh picker — show a popup when the mesh has multiple submeshes
                if (cache.sourceMesh != null && cache.sourceMesh.subMeshCount > 1)
                {
                    int subCount = cache.sourceMesh.subMeshCount;
                    string[] subLabels = new string[subCount];
                    for (int si = 0; si < subCount; si++) subLabels[si] = $"Submesh {si}";
                    int newIdx = EditorGUILayout.Popup(
                        new GUIContent("Submesh", "Which submesh to bake OMM for."),
                        Mathf.Clamp(_propSubmeshIndex.intValue, 0, subCount - 1),
                        subLabels);
                    if (newIdx != _propSubmeshIndex.intValue)
                    {
                        _propSubmeshIndex.intValue = newIdx;
                        serializedObject.ApplyModifiedProperties();
                    }
                }
                else
                {
                    // Single-submesh mesh: always index 0, show read-only label
                    using (new EditorGUI.DisabledScope(true))
                        EditorGUILayout.LabelField("Submesh", "0 (only)");
                    if (_propSubmeshIndex.intValue != 0)
                    {
                        _propSubmeshIndex.intValue = 0;
                        serializedObject.ApplyModifiedProperties();
                    }
                }

                EditorGUILayout.PropertyField(_propSourceTexture, new GUIContent("Texture"));
                EditorGUILayout.PropertyField(_propAlphaCutoff, new GUIContent("Alpha Cutoff"));
                EditorGUILayout.PropertyField(_propMaxSubdiv, new GUIContent("Max Subdivision Level"));
                EditorGUILayout.PropertyField(_propDynSubdivScale, new GUIContent("Dynamic Subdiv Scale",
                    "Micro-triangle covers ~(scale²) texels. 0 = uniform: uses Max Subdivision Level for all triangles."));
                EditorGUILayout.PropertyField(_propOmmFormat, new GUIContent("OMM Format",
                    "FourState (default): opaque/transparent/unknown-O/unknown-T. TwoState: opaque/transparent only, faster bake."));
                EditorGUILayout.PropertyField(_propDownsample, new GUIContent("Texture Downsample",
                    "Downsample the alpha texture before baking. Higher factors bake faster with less detail."));
                if (cache.textureDownsampleFactor != OMMCache.DownsampleFactor.x1 && cache.sourceTexture != null)
                {
                    int f = (int)cache.textureDownsampleFactor;
                    EditorGUILayout.HelpBox(
                        $"Bake will use {cache.sourceTexture.width / f}×{cache.sourceTexture.height / f} (1/{f} of original {cache.sourceTexture.width}×{cache.sourceTexture.height})",
                        MessageType.None);
                }
                serializedObject.ApplyModifiedProperties();

                EditorGUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                using (new EditorGUI.DisabledScope(
                           cache.sourceMesh == null || cache.sourceTexture == null || cache.alphaCutoff <= 0f))
                {
                    if (GUILayout.Button("Bake", GUILayout.Width(90)))
                    {
                        bool ok = BakeInto(cache, cache.submeshIndex);
                        if (ok)
                        {
                            _bakeStatus  = $"Baked \u2713  ({cache.bakedIndexCount} triangles, {cache.bakedArrayData.Length / 1024.0f:F1} KB)";
                            _bakeSuccess = true;
                            InvalidateGPUBuffers();
                        }
                        else
                        {
                            _bakeStatus  = "Bake failed \u2014 check Console for details.";
                            _bakeSuccess = false;
                        }
                    }
                }

                GUILayout.FlexibleSpace();
                EditorGUILayout.EndHorizontal();

                if (_bakeStatus != null)
                    EditorGUILayout.HelpBox(_bakeStatus, _bakeSuccess ? MessageType.Info : MessageType.Error);

                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndFoldoutHeaderGroup();

            EditorGUILayout.Space(4);

            // ── Validity badge ─────────────────────────────────────────────
            if (cache.IsValid)
            {
                var oldColor = GUI.color;
                GUI.color = new Color(0.3f, 0.9f, 0.4f);
                EditorGUILayout.HelpBox("\u2713  Cache is valid", MessageType.None);
                GUI.color = oldColor;
            }
            else
            {
                EditorGUILayout.HelpBox("\u2717  Cache is empty \u2014 fill in the fields above and click Bake.", MessageType.Warning);
                return;
            }

            EditorGUILayout.Space(4);

            // ── Statistics ─────────────────────────────────────────────────
            _showStats = EditorGUILayout.BeginFoldoutHeaderGroup(_showStats, "Statistics");
            if (_showStats)
            {
                EditorGUI.indentLevel++;
                DrawReadOnly("Array Data", $"{cache.bakedArrayData.Length:N0} bytes  ({cache.bakedArrayData.Length / 1024.0f:F1} KB)");
                DrawReadOnly("Desc Array", $"{cache.bakedDescArray.Length:N0} bytes  ({cache.bakedDescArrayCount} descriptors)");
                DrawReadOnly("Index Buffer", $"{cache.bakedIndexBuffer.Length:N0} bytes  ({cache.bakedIndexCount} indices, stride {cache.bakedIndexStride})");
                DrawReadOnly("Triangles", $"{cache.bakedIndexCount}");
                DrawReadOnly("Histogram entries", $"{cache.HistogramEntryCount}");
                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndFoldoutHeaderGroup();

            EditorGUILayout.Space(4);

            // ── UV-space GPU Preview ───────────────────────────────────────
            _showPreview = EditorGUILayout.BeginFoldoutHeaderGroup(_showPreview, "UV-Space Preview");
            if (_showPreview)
            {
                EditorGUI.indentLevel++;

                if (cache.sourceMesh == null)
                {
                    EditorGUILayout.HelpBox("Assign a Source Mesh to enable UV-space preview.", MessageType.Info);
                }
                else
                {
                    // Options
                    EditorGUILayout.BeginHorizontal();
                    bool newColorize  = EditorGUILayout.ToggleLeft("Colorize", _optColorize, GUILayout.Width(80));
                    bool newContour   = EditorGUILayout.ToggleLeft("Contour", _optContour, GUILayout.Width(80));
                    bool newWireframe = EditorGUILayout.ToggleLeft("Wireframe", _optWireframe, GUILayout.Width(80));
                    EditorGUILayout.EndHorizontal();

                    int maxTri     = (int)cache.bakedIndexCount;
                    int newPrimEnd = EditorGUILayout.IntSlider("Tri End (0=all)", _primitiveEnd < 0 ? 0 : _primitiveEnd, 0, maxTri);

                    bool dirty = newColorize != _optColorize || newContour != _optContour
                                                             || newWireframe != _optWireframe || newPrimEnd != (_primitiveEnd < 0 ? 0 : _primitiveEnd);
                    _optColorize  = newColorize;
                    _optContour   = newContour;
                    _optWireframe = newWireframe;
                    _primitiveEnd = newPrimEnd == 0 ? -1 : newPrimEnd;

                    if (_previewError != null)
                        EditorGUILayout.HelpBox(_previewError, MessageType.Error);

                    // Render and display
                    Texture previewImg = RenderUVPreview(cache, dirty || _cachedFor != cache);
                    _cachedFor = cache;

                    if (previewImg != null)
                    {
                        Rect previewRect = GUILayoutUtility.GetAspectRect(1.0f);
                        previewRect.width  = Mathf.Min(previewRect.width, PreviewSize);
                        previewRect.height = previewRect.width;
                        float cx = (EditorGUIUtility.currentViewWidth - previewRect.width) * 0.5f;
                        previewRect.x = cx;

                        GUI.DrawTexture(previewRect, previewImg, ScaleMode.ScaleToFit, false);
                        EditorGUILayout.Space(previewRect.height);
                    }

                    // Legend
                    DrawLegend();
                }

                EditorGUI.indentLevel--;
            }

            EditorGUILayout.EndFoldoutHeaderGroup();
        }

        // ── GPU UV-space preview ───────────────────────────────────────────

        private Texture RenderUVPreview(OMMCache cache, bool rebuildBuffers)
        {
            _previewError = null;

            // Lazy-init PreviewRenderUtility
            if (_preview == null)
            {
                _preview                           = new PreviewRenderUtility();
                _preview.camera.orthographic       = true;
                _preview.camera.orthographicSize   = 1f;
                _preview.camera.nearClipPlane      = 0.01f;
                _preview.camera.farClipPlane       = 10f;
                _preview.camera.transform.position = new Vector3(0, 0, -1f);
                _preview.camera.transform.LookAt(Vector3.zero);
                _preview.camera.clearFlags      = CameraClearFlags.SolidColor;
                _preview.camera.backgroundColor = new Color(0.12f, 0.12f, 0.12f, 1f);
            }

            // Lazy-load shader materials (one per shader)
            if (_previewMatFill == null)
            {
                var shaderFill = Shader.Find("NativeRender/OMMPreviewFill");
                if (shaderFill == null)
                {
                    _previewError = "Shader 'NativeRender/OMMPreviewFill' not found. Ensure OMMPreviewFill.shader is in Assets/Shaders/.";
                    return null;
                }

                _previewMatFill = new Material(shaderFill) { hideFlags = HideFlags.HideAndDontSave };

                var shaderWire = Shader.Find("NativeRender/OMMPreviewWire");
                if (shaderWire == null)
                {
                    _previewError = "Shader 'NativeRender/OMMPreviewWire' not found. Ensure OMMPreviewWire.shader is in Assets/Shaders/.";
                    return null;
                }

                _previewMatWire = new Material(shaderWire) { hideFlags = HideFlags.HideAndDontSave };
            }

            // Build UV mesh (positions = UV coords mapped to [-1,1])
            if (_uvMesh == null || rebuildBuffers)
                _uvMesh = BuildUVMesh(cache.sourceMesh);

            if (_uvMesh == null)
            {
                _previewError = "Source mesh has no UV channel.";
                return null;
            }

            // Build/refresh GPU buffers when cache data changes
            if (rebuildBuffers || _ommIndexBuf == null)
            {
                ReleaseGPUBuffers();
                if (!BuildGPUBuffers(cache))
                    return null;
            }

            // Set material properties on fill material
            _previewMatFill.SetBuffer("_OmmIndexBuf", _ommIndexBuf);
            _previewMatFill.SetBuffer("_OmmDescBuf", _ommDescBuf);
            _previewMatFill.SetBuffer("_OmmArrayBuf", _ommArrayBuf);
            _previewMatFill.SetTexture("_AlphaTex", cache.sourceTexture != null ? (Texture)cache.sourceTexture : Texture2D.whiteTexture);
            _previewMatFill.SetFloat("_AlphaCutoff", cache.alphaCutoff);
            _previewMatFill.SetInt("_ColorizeModes", _optColorize ? 1 : 0);
            _previewMatFill.SetInt("_DrawContour", _optContour ? 1 : 0);
            _previewMatFill.SetInt("_HighlightOMM", _optHighlight);
            _previewMatFill.SetInt("_PrimitiveOffset", _primitiveStart);

            _preview.BeginPreview(new Rect(0, 0, PreviewSize, PreviewSize), GUIStyle.none);

            // Fill pass
            _preview.DrawMesh(_uvMesh, Matrix4x4.identity, _previewMatFill, 0, (MaterialPropertyBlock)null);

            // Wireframe overlay (dedicated shader — no pass toggling needed)
            if (_optWireframe)
                _preview.DrawMesh(_uvMesh, Matrix4x4.identity, _previewMatWire, 0, (MaterialPropertyBlock)null);

            _preview.camera.Render();
            Texture result = _preview.EndPreview();
            return result;
        }

        // ── Build a mesh whose vertex positions are UV coords in NDC ──────

        private static Mesh BuildUVMesh(Mesh src)
        {
            if (src == null) return null;
            Vector2[] uvs = src.uv;
            if (uvs == null || uvs.Length == 0) return null;

            // Vertex positions = UV mapped to XY, z=0
            Vector3[] verts = new Vector3[uvs.Length];
            for (int i = 0; i < uvs.Length; i++)
                verts[i] = new Vector3(uvs[i].x, uvs[i].y, 0f);

            var m = new Mesh { hideFlags = HideFlags.HideAndDontSave };
            m.name        = "OMMPreviewUVMesh";
            m.indexFormat = src.indexFormat;
            m.vertices    = verts;
            m.uv          = uvs;

            int sc = src.subMeshCount;
            m.subMeshCount = sc;
            for (int s = 0; s < sc; s++)
                m.SetTriangles(src.GetTriangles(s), s);

            m.RecalculateBounds();
            return m;
        }

        // ── Build GPU ComputeBuffers from baked OMM data ───────────────────

        private bool BuildGPUBuffers(OMMCache cache)
        {
            try
            {
                // OMM index buffer: int per triangle (signed; negative = special)
                int   triCount   = (int)cache.bakedIndexCount;
                int[] ommIndices = new int[triCount];
                for (int i = 0; i < triCount; i++)
                    ommIndices[i] = (int)ReadIndexBuffer(cache.bakedIndexBuffer, (uint)i, cache.bakedIndexStride);

                _ommIndexBuf = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);
                _ommIndexBuf.SetData(ommIndices);

                // OMM desc buffer: uint2 per desc  [offset, subdivLevel | (format<<16)]
                int    descCount = (int)cache.bakedDescArrayCount;
                uint[] descData  = new uint[descCount * 2];
                for (int d = 0; d < descCount; d++)
                {
                    int    b      = d * 8;
                    uint   offset = BitConverter.ToUInt32(cache.bakedDescArray, b);
                    ushort subdiv = BitConverter.ToUInt16(cache.bakedDescArray, b + 4);
                    ushort fmt    = BitConverter.ToUInt16(cache.bakedDescArray, b + 6);
                    descData[d * 2 + 0] = offset;
                    descData[d * 2 + 1] = (uint)subdiv | ((uint)fmt << 16);
                }

                _ommDescBuf = new ComputeBuffer(descCount, sizeof(uint) * 2, ComputeBufferType.Structured);
                _ommDescBuf.SetData(descData);

                // OMM array data buffer: uint[] (raw dword-packed bits)
                int    dwordCount  = (cache.bakedArrayData.Length + 3) / 4;
                uint[] arrayDwords = new uint[dwordCount];
                Buffer.BlockCopy(cache.bakedArrayData, 0, arrayDwords, 0, cache.bakedArrayData.Length);
                _ommArrayBuf = new ComputeBuffer(dwordCount, sizeof(uint), ComputeBufferType.Structured);
                _ommArrayBuf.SetData(arrayDwords);

                return true;
            }
            catch (Exception ex)
            {
                _previewError = $"Failed to build GPU buffers: {ex.Message}";
                ReleaseGPUBuffers();
                return false;
            }
        }

        private void ReleaseGPUBuffers()
        {
            _ommIndexBuf?.Release();
            _ommIndexBuf = null;
            _ommDescBuf?.Release();
            _ommDescBuf = null;
            _ommArrayBuf?.Release();
            _ommArrayBuf = null;
        }

        private void InvalidateGPUBuffers()
        {
            ReleaseGPUBuffers();
            _cachedFor = null;
            if (_uvMesh != null)
            {
                DestroyImmediate(_uvMesh);
                _uvMesh = null;
            }
        }

        // ── Read a descriptor-index entry from the OMM index buffer ───────

        private static uint ReadIndexBuffer(byte[] buf, uint triIdx, uint stride)
        {
            uint byteOff = triIdx * stride;
            switch (stride)
            {
                case 1: return buf[byteOff];
                case 2: return BitConverter.ToUInt16(buf, (int)byteOff);
                default: return BitConverter.ToUInt32(buf, (int)byteOff);
            }
        }

        // ── Legend ────────────────────────────────────────────────────────

        private void DrawLegend()
        {
            EditorGUILayout.BeginHorizontal();
            GUILayout.FlexibleSpace();
            DrawSwatch(new Color(0, 0, 1), "Transparent");
            DrawSwatch(new Color(0, 1, 0), "Opaque");
            DrawSwatch(new Color(1, 0, 1), "Unknown (T)");
            DrawSwatch(new Color(1, 1, 0), "Unknown (O)");
            GUILayout.FlexibleSpace();
            EditorGUILayout.EndHorizontal();
        }

        private static void DrawSwatch(Color c, string label)
        {
            Rect r = GUILayoutUtility.GetRect(14, 14, GUILayout.Width(14), GUILayout.Height(14));
            EditorGUI.DrawRect(r, c);
            GUILayout.Label(label, EditorStyles.miniLabel);
            GUILayout.Space(6);
        }

        // ── Helpers ───────────────────────────────────────────────────────

        private static void DrawReadOnly(string label, string value)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(label, GUILayout.Width(EditorGUIUtility.labelWidth));
            EditorGUILayout.SelectableLabel(value, EditorStyles.textField,
                GUILayout.Height(EditorGUIUtility.singleLineHeight));
            EditorGUILayout.EndHorizontal();
        }

        private void OnDisable()
        {
            InvalidateGPUBuffers();
            if (_previewMatFill != null)
            {
                DestroyImmediate(_previewMatFill);
                _previewMatFill = null;
            }

            if (_previewMatWire != null)
            {
                DestroyImmediate(_previewMatWire);
                _previewMatWire = null;
            }

            _preview?.Cleanup();
            _preview = null;
        }

        /// <summary>
        /// Bakes OMM data using the source references stored on <paramref name="cache"/>
        /// and writes the result back into that same asset.
        /// Returns true on success, false on failure.
        /// </summary>
        public static bool BakeInto(OMMCache cache, int submeshIndex = 0)
        {
            if (cache == null) { Debug.LogError("[OMMBaker] BakeInto: cache is null"); return false; }

            Mesh      mesh                  = cache.sourceMesh;
            Texture2D texture               = cache.sourceTexture;
            float     alphaCutoff           = cache.alphaCutoff;
            byte      maxSubdivisionLevel   = cache.maxSubdivisionLevel;
            int       downsample            = (int)cache.textureDownsampleFactor;
            float     dynSubdivScale        = cache.dynamicSubdivisionScale;
            byte      ommFormat             = (byte)cache.ommFormat;

            if (mesh == null)
            {
                Debug.LogError("[OMMBaker] BakeInto: sourceMesh is null");
                return false;
            }

            if (submeshIndex < 0 || submeshIndex >= mesh.subMeshCount)
            {
                Debug.LogError($"[OMMBaker] BakeInto: submeshIndex {submeshIndex} is out of range (mesh has {mesh.subMeshCount} submeshes)");
                return false;
            }

            if (texture == null)
            {
                Debug.LogError("[OMMBaker] BakeInto: sourceTexture is null");
                return false;
            }

            if (alphaCutoff <= 0f)
            {
                Debug.LogWarning("[OMMBaker] BakeInto: alphaCutoff <= 0");
                return false;
            }


            // -- Read alpha pixels, optionally downsampling -----------------
            int bakeTexW = Mathf.Max(1, texture.width  / downsample);
            int bakeTexH = Mathf.Max(1, texture.height / downsample);

            Color32[] pixels32;
            if (texture.isReadable && downsample == 1)
            {
                pixels32 = texture.GetPixels32(0);
            }
            else
            {
                // Blit into a RenderTexture at the target (possibly downsampled) resolution.
                // Graphics.Blit performs bilinear downsampling automatically.
                RenderTexture tmp = RenderTexture.GetTemporary(
                    bakeTexW, bakeTexH, 0,
                    RenderTextureFormat.ARGB32,
                    RenderTextureReadWrite.Linear);

                Graphics.Blit(texture, tmp);
                RenderTexture previous = RenderTexture.active;
                RenderTexture.active = tmp;

                Texture2D readableTex = new Texture2D(bakeTexW, bakeTexH, TextureFormat.RGBA32, false);
                readableTex.ReadPixels(new Rect(0, 0, bakeTexW, bakeTexH), 0, 0);
                readableTex.Apply();
                pixels32 = readableTex.GetPixels32();

                RenderTexture.active = previous;
                RenderTexture.ReleaseTemporary(tmp);
                DestroyImmediate(readableTex);
            }
            // -----------------------------------------------------------------

            Texture2D tex2d       = texture;
            Vector2[] meshUV      = mesh.uv;
            uint      indexStride = mesh.indexFormat == IndexFormat.UInt16 ? 2u : 4u;

            if (meshUV == null || meshUV.Length == 0)
            {
                Debug.LogError($"[OMMBaker] Mesh '{mesh.name}' has no UV channel");
                return false;
            }

            string label = $"{mesh.name}[{submeshIndex}]";

            // -- Extract indices for the target submesh only -----------------
            var   subDesc   = mesh.GetSubMesh(submeshIndex);
            int[] allTris   = mesh.triangles;
            // subDesc.indexStart is in units of indices, subDesc.indexCount is the count
            int  subStart   = subDesc.indexStart;
            int  subCount2  = subDesc.indexCount;
            uint indexCount = (uint)subCount2;

            // -- Alpha bytes (R8_UNORM) ----------------------------------------
            byte[] alphaBytes = new byte[pixels32.Length];
            for (int i = 0; i < pixels32.Length; i++)
                alphaBytes[i] = pixels32[i].a;

            // -- Per-vertex flat UVs ------------------------------------------
            // The OMM SDK indexes into texCoords using the index buffer values,
            // so this must be a per-vertex array (size = meshUV.Length), NOT
            // a pre-expanded per-triangle-vertex array (size = indexCount).
            int     vertexCount = meshUV.Length;
            float[] uvsFlat     = new float[vertexCount * 2];
            for (int vi = 0; vi < vertexCount; vi++)
            {
                uvsFlat[vi * 2 + 0] = meshUV[vi].x;
                uvsFlat[vi * 2 + 1] = meshUV[vi].y;
            }

            // -- Raw index bytes (submesh only) --------------------------------
            byte[] idxBytes = new byte[indexCount * indexStride];
            if (indexStride == 2)
            {
                for (int ti = 0; ti < (int)indexCount; ti++)
                {
                    ushort v = (ushort)allTris[subStart + ti];
                    idxBytes[ti * 2 + 0] = (byte)(v & 0xFF);
                    idxBytes[ti * 2 + 1] = (byte)(v >> 8);
                }
            }
            else
            {
                for (int ti = 0; ti < (int)indexCount; ti++)
                {
                    uint v = (uint)allTris[subStart + ti];
                    idxBytes[ti * 4 + 0] = (byte)(v & 0xFF);
                    idxBytes[ti * 4 + 1] = (byte)((v >> 8) & 0xFF);
                    idxBytes[ti * 4 + 2] = (byte)((v >> 16) & 0xFF);
                    idxBytes[ti * 4 + 3] = (byte)(v >> 24);
                }
            }
            // -----------------------------------------------------------------

            Debug.Log($"[OMMBaker] '{label}': calling NR_BakeOMMCPU " +
                      $"(tex={bakeTexW}x{bakeTexH}{(downsample > 1 ? $" (1/{downsample} of {tex2d.width}x{tex2d.height})" : "")}, indices={indexCount}, cutoff={alphaCutoff:F3})");

            int bakeOk;
            unsafe
            {
                fixed (byte* pAlpha = alphaBytes)
                fixed (float* pUVs = uvsFlat)
                fixed (byte* pIdx = idxBytes)
                {
                    bakeOk = OMMBakerPlugin.NR_BakeOMMCPU(
                        (IntPtr)pAlpha, (uint)bakeTexW, (uint)bakeTexH,
                        (IntPtr)pUVs,
                        (IntPtr)pIdx, indexCount, indexStride,
                        alphaCutoff, maxSubdivisionLevel,
                        dynSubdivScale, ommFormat);
                }
            }

            if (bakeOk == 0)
            {
                Debug.LogError($"[OMMBaker] '{label}': NR_BakeOMMCPU failed — check native log");
                return false;
            }

            // -- Collect baked result via single struct call ------------------
            if (OMMBakerPlugin.NR_GetBakeResult(out var rd) == 0)
            {
                Debug.LogError($"[OMMBaker] '{label}': NR_GetBakeResult returned no valid result");
                return false;
            }

            byte[] bakedArrayData   = new byte[rd.arrayDataSize];
            byte[] bakedDescArray   = new byte[rd.descArrayByteCount];
            byte[] bakedIndexBuffer = new byte[rd.indexCount * rd.indexStride];
            uint[] histogramFlat    = new uint[rd.histogramCount * 3];

            if (rd.arrayDataSize > 0) Marshal.Copy(rd.arrayData, bakedArrayData, 0, (int)rd.arrayDataSize);
            if (rd.descArrayByteCount > 0) Marshal.Copy(rd.descArray, bakedDescArray, 0, (int)rd.descArrayByteCount);
            if (bakedIndexBuffer.Length > 0) Marshal.Copy(rd.indexBuffer, bakedIndexBuffer, 0, bakedIndexBuffer.Length);
            if (rd.histogramCount > 0)
            {
                int[] histTmp = new int[rd.histogramCount * 3];
                Marshal.Copy(rd.histogramFlat, histTmp, 0, histTmp.Length);
                Buffer.BlockCopy(histTmp, 0, histogramFlat, 0, histTmp.Length * sizeof(int));
            }

            uint descArrayCount   = rd.descArrayCount;
            uint bakedIndexCount  = rd.indexCount;
            uint bakedIndexStride = rd.indexStride;
            uint histCount        = rd.histogramCount;

            OMMBakerPlugin.NR_FreeBakeResult();

            Debug.Log($"[OMMBaker] '{label}': bake complete — " +
                      $"arrayData={rd.arrayDataSize}B, descs={descArrayCount}, " +
                      $"indices={bakedIndexCount}, histEntries={histCount}");

            // -- Write result into the existing cache asset ------------------
            cache.submeshIndex        = submeshIndex;
            cache.bakedArrayData      = bakedArrayData;
            cache.bakedDescArray      = bakedDescArray;
            cache.bakedDescArrayCount = descArrayCount;
            cache.bakedIndexBuffer    = bakedIndexBuffer;
            cache.bakedIndexCount     = bakedIndexCount;
            cache.bakedIndexStride    = bakedIndexStride;
            cache.histogramFlat       = histogramFlat;

            EditorUtility.SetDirty(cache);
            AssetDatabase.SaveAssetIfDirty(cache);

            Debug.Log($"[OMMBaker] Saved → '{AssetDatabase.GetAssetPath(cache)}'");
            return true;
        }
    }
}