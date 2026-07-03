using System;
using System.Collections.Generic;

namespace NativeRender
{
    /// <summary>D3D12 register class of a reflected binding (t / u / b / s registers).</summary>
    internal enum BindingRegClass : byte
    {
        SRV,     // t registers (Texture / StructuredBuffer / ByteAddressBuffer / TLAS)
        UAV,     // u registers
        CBV,     // b registers
        Sampler  // s registers
    }

    /// <summary>
    /// One resource binding parsed from the import-time reflection JSON
    /// (produced by NR_SC_ReflectCS / NR_SC_ReflectLib on the compiler plugin
    /// and cached on the shader asset). Used ONLY on the C# side — for
    /// auto-generating a <see cref="NativeBindingLayout"/>, resolving HLSL
    /// variable names to layout slots, and validating hand-authored layouts.
    /// The render plugin itself never reflects (nvrhi model).
    /// </summary>
    internal struct ReflectedBinding
    {
        public string          Name;
        public BindingRegClass RegClass;
        public bool            IsTlas;    // RaytracingAccelerationStructure (t register)
        public uint            Reg;
        public uint            Space;
        /// <summary>BindCount: 1 = single, N = bounded array, 0 = unbounded array.
        /// Reflection JSONs cached before the "count" field existed parse as 1.</summary>
        public uint            Count;
        /// <summary>CBV byte size ("size" field); 0 for non-CBV bindings.</summary>
        public uint            SizeBytes;
    }

    /// <summary>
    /// Minimal parser for the reflection JSON shape emitted by the shader
    /// compiler plugin:
    /// <c>{ "bindings": [ { "name": "...", "type": "SRV|UAV|CBV|Sampler|TLAS",
    /// "space": N, "reg": N, "count": N, ... , "size": N }, ... ] }</c>.
    /// </summary>
    internal static class ShaderReflectionInfo
    {
        public static List<ReflectedBinding> Parse(string json)
        {
            var result = new List<ReflectedBinding>();
            ParseInto(json, result);
            return result;
        }

        /// <summary>
        /// Appends the bindings of <paramref name="json"/> into <paramref name="dst"/>,
        /// de-duplicating by (name, reg, space) — used to merge the reflection of a
        /// multi-blob RT pipeline (primary + hit-group shaders).
        /// </summary>
        public static void ParseInto(string json, List<ReflectedBinding> dst)
        {
            if (string.IsNullOrEmpty(json)) return;

            int bindingsIdx = json.IndexOf("\"bindings\"", StringComparison.Ordinal);
            if (bindingsIdx < 0) return;
            int arrayStart = json.IndexOf('[', bindingsIdx);
            if (arrayStart < 0) return;
            int arrayEnd = json.IndexOf(']', arrayStart);
            if (arrayEnd < 0) arrayEnd = json.Length;

            int pos = arrayStart + 1;
            while (pos < arrayEnd)
            {
                int objStart = json.IndexOf('{', pos);
                if (objStart < 0 || objStart >= arrayEnd) break;
                int objEnd = json.IndexOf('}', objStart);
                if (objEnd < 0) break;

                string obj  = json.Substring(objStart + 1, objEnd - objStart - 1);
                string name = ExtractString(obj, "name");
                string type = ExtractString(obj, "type");
                if (!string.IsNullOrEmpty(name) && TryClassOf(type, out var cls, out bool isTlas))
                {
                    var b = new ReflectedBinding
                    {
                        Name      = name,
                        RegClass  = cls,
                        IsTlas    = isTlas,
                        Reg       = ExtractUInt(obj, "reg",   0),
                        Space     = ExtractUInt(obj, "space", 0),
                        Count     = ExtractUInt(obj, "count", 1),
                        SizeBytes = ExtractUInt(obj, "size",  0),
                    };

                    bool duplicate = false;
                    for (int i = 0; i < dst.Count; i++)
                    {
                        if (dst[i].Name == b.Name && dst[i].Reg == b.Reg && dst[i].Space == b.Space)
                        {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) dst.Add(b);
                }

                pos = objEnd + 1;
            }
        }

        private static bool TryClassOf(string type, out BindingRegClass cls, out bool isTlas)
        {
            isTlas = false;
            switch (type)
            {
                case "SRV":     cls = BindingRegClass.SRV;     return true;
                case "TLAS":    cls = BindingRegClass.SRV;     isTlas = true; return true;
                case "UAV":     cls = BindingRegClass.UAV;     return true;
                case "CBV":     cls = BindingRegClass.CBV;     return true;
                case "Sampler": cls = BindingRegClass.Sampler; return true;
                default:        cls = BindingRegClass.SRV;     return false;
            }
        }

        private static string ExtractString(string obj, string key)
        {
            string search = "\"" + key + "\"";
            int ki = obj.IndexOf(search, StringComparison.Ordinal);
            if (ki < 0) return null;
            int colon = obj.IndexOf(':', ki + search.Length);
            if (colon < 0) return null;
            int q1 = obj.IndexOf('"', colon + 1);
            if (q1 < 0) return null;
            int q2 = obj.IndexOf('"', q1 + 1);
            if (q2 < 0) return null;
            return obj.Substring(q1 + 1, q2 - q1 - 1);
        }

        private static uint ExtractUInt(string obj, string key, uint fallback)
        {
            string search = "\"" + key + "\"";
            int ki = obj.IndexOf(search, StringComparison.Ordinal);
            if (ki < 0) return fallback;
            int colon = obj.IndexOf(':', ki + search.Length);
            if (colon < 0) return fallback;
            int p = colon + 1;
            while (p < obj.Length && (obj[p] == ' ' || obj[p] == '\t')) p++;
            uint value = 0;
            bool any = false;
            while (p < obj.Length && obj[p] >= '0' && obj[p] <= '9')
            {
                value = value * 10u + (uint)(obj[p] - '0');
                any = true;
                p++;
            }
            return any ? value : fallback;
        }
    }
}
