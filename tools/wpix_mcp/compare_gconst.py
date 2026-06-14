"""Extract and compare g_Const (ConstantBuffer<SampleConstants>) of the raytracing passes
in the two Other/ captures.

g_Const is bound through the DXR global root signature (the *compute* root-binding path),
which describe_dispatch/extract don't track (they only see ->Dispatch, not ->DispatchRays).
So this reads the binding straight from the exported command list: for each DispatchRays it
finds the active SetComputeRootConstantBufferView whose root-signature register is b0 (g_Const),
then reads those bytes from the full-capture export (the per-event region recapture renumbers /
drops the upload buffer, but the whole-capture export keeps stable ids + the captured bytes).

The SampleConstants layout is pulled live from the RTXPT shader headers and overlaid on the
bytes; SampleConstants is hand-padded to 16-byte rows so its tight-packed layout matches the
cbuffer layout, which is what wpix_core's struct parser assumes.

Run from tools/wpix_mcp:  python compare_gconst.py [rtxpt.wpix] [unity.wpix] [--shaders DIR]
"""
import argparse, os, re, sys
import wpix_core as w
from wpix_core import parse_struct_layout, _field_value, _val_str, _render_table

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")

DEF_SHADERS = os.path.join("..", "..", "RenderingPlugin", "External", "RTXPT", "Rtxpt", "Shaders")


def struct_sources(shaders):
    """(file, struct) in dependency order — dependencies must precede the structs that use them."""
    return [
        (os.path.join(shaders, "PathTracer", "PathTracerShared.h"), "PathTracerCameraData"),
        (os.path.join(shaders, "PathTracer", "PathTracerShared.h"), "PathTracerConstants"),
        (os.path.join(shaders, "PathTracer", "Lighting", "EnvMap.hlsli"), "EnvMapSceneParams"),
        (os.path.join(shaders, "PathTracer", "Lighting", "EnvMap.hlsli"), "EnvMapImportanceSamplingParams"),
        (os.path.join(shaders, "PathTracer", "PathTracerDebug.hlsli"), "DebugConstants"),
        (os.path.join(shaders, "SampleConstantBuffer.h"), "SimpleViewConstants"),
        (os.path.join(shaders, "SampleConstantBuffer.h"), "SampleConstants"),
    ]


def grab_struct(path, name):
    """Brace-matched text of `struct name { ... };` from a source file."""
    t = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"struct\s+%s\s*\{" % re.escape(name), t)
    if not m:
        raise RuntimeError(f"struct {name} not found in {path}")
    depth, j = 0, m.end() - 1
    while j < len(t):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return t[m.start():j + 1] + ";\n"


def sample_constants_def(shaders=DEF_SHADERS):
    return "\n".join(grab_struct(p, n) for p, n in struct_sources(shaders))


def gconst_binding(wpix):
    """For each DispatchRays in `wpix`, the g_Const root CBV (resource_id, offset).
    g_Const = the bound root CBV whose global-root-signature register is b0."""
    ed = w.load_export(w.export_full(wpix))
    import glob
    text = ""
    for f in glob.glob(os.path.join(ed.dir, "CommandLists_*.cpp")):
        text += open(f, encoding="utf-8", errors="replace").read()
    out = []
    cur_sig = None; root_cbv = {}; last_gid = None
    for ln in text.split("\n"):
        m = re.search(r"SetComputeRootSignature\(GetRootSignature\((\d+)\)", ln)
        if m: cur_sig = int(m.group(1))
        m = re.search(r"SetComputeRootConstantBufferView\((\d+),\s*GetGpuva\((\d+),\s*(\d+)\)", ln)
        if m: root_cbv[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
        m = re.search(r"//\s*GlobalId\s*=\s*(\d+)", ln)
        if m: last_gid = int(m.group(1))
        if "->DispatchRays(" in ln:
            sig = ed.root_sigs.get(cur_sig, {})
            # find the root-CBV param whose register == 0 (b0 = g_Const)
            g = None
            for param, (res, off) in root_cbv.items():
                reg = sig.get(param, {}).get("reg")
                if reg and reg[0] == 0:
                    g = (param, res, off, reg)
            if g is None and root_cbv:  # fallback: the only/last bound root CBV
                param = sorted(root_cbv)[-1]
                res, off = root_cbv[param]
                g = (param, res, off, sig.get(param, {}).get("reg"))
            out.append({"global_id": last_gid, "root_sig": cur_sig,
                        "param": g[0], "resource_id": g[1], "offset": g[2], "reg": g[3]})
    return ed, out


def read_gconst(ed, rid, off, nbytes):
    blob = w.decompress_blob(ed.dir, rid, ed)
    return blob[off:off + nbytes]


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY, shaders=DEF_SHADERS):
    layout, size = parse_struct_layout(sample_constants_def(shaders), "SampleConstants")
    print(f"sizeof(SampleConstants) = {size} bytes ({len(layout)} scalar fields)\n")

    captures = {}
    for tag, wpix in [("RTXPT", rtxpt), ("UNITY", unity)]:
        ed, binds = gconst_binding(wpix)
        # all ray passes in a capture bind the same g_Const buffer+offset; report + verify.
        markers = {e["global_id"]: e["marker"].split("/")[-1]
                   for e in w.find_shader_events(wpix) if e["name"] == "DispatchRays"}
        print(f"== {tag} g_Const bindings ==")
        sigset = set()
        for b in binds:
            print(f"   DispatchRays gid={b['global_id']} ({markers.get(b['global_id'],'?')}): "
                  f"rootSig={b['root_sig']} param={b['param']} reg=b{b['reg'][0] if b['reg'] else '?'} "
                  f"-> resource {b['resource_id']} @ offset {b['offset']}")
            sigset.add((b["resource_id"], b["offset"]))
        rid, off = next(iter(sigset))
        if len(sigset) > 1:
            print(f"   NOTE: ray passes bind DIFFERENT g_Const buffers: {sigset}")
        captures[tag] = read_gconst(ed, rid, off, size)
        print()

    a, b = captures["RTXPT"], captures["UNITY"]
    print(f"Bytes read: RTXPT={len(a)}  UNITY={len(b)}\n")

    rows = []
    ndiff = 0
    for f in layout:
        va = _field_value(a, f); vb = _field_value(b, f)
        differ = va != vb
        # treat tiny float drift as "~=" but still list it
        if differ:
            ndiff += 1
            rows.append([f["offset"], f["name"], f["type"], _val_str(va), _val_str(vb)])
    print(f"Differing fields: {ndiff} / {len(layout)}\n")
    if rows:
        print(_render_table(["offset", "field", "type", "RTXPT", "UNITY"], rows))
    else:
        print("g_Const is byte-identical between the two captures.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    p.add_argument("--shaders", default=DEF_SHADERS, help="RTXPT Shaders dir for struct layouts")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity, args.shaders))
