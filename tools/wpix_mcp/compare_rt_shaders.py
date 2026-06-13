"""Compare the raytracing (DXR) shaders executed in the two Other/ captures.

The ray passes run via SetPipelineState1 + DispatchRays on DXR *state objects*, each of
which bundles one or more DXIL library blobs (raygen/miss/closesthit/anyhit). describe_shader
only handles compute PSOs, so this walks the exported CreateStateObject_*() functions, replays
the global resources.bin read order to find each DXIL library's byte offset, decompresses it,
and reads its container HASH (the value PIX shows) + target + debug name + exports/hit groups.

Run from tools/wpix_mcp:  python compare_rt_shaders.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, os, re, sys
import wpix_core as w
from wpix_core import _READ_RE, _CALL_RE, container_compile_info, _xpress_decompress_at

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")


def _has_read_map(funcs):
    """name -> bool: does this function (transitively) perform a g_resourceReader->Read."""
    cache = {}
    def chk(name, stack):
        if name in cache:
            return cache[name]
        if name in stack or name not in funcs:
            return False
        stack.add(name)
        res = False
        for ln in funcs[name]:
            if _READ_RE.search(ln):
                res = True; break
            cm = _CALL_RE.match(ln)
            if cm and cm.group(1) in funcs and cm.group(1) != name and chk(cm.group(1), stack):
                res = True; break
        stack.discard(name)
        cache[name] = res
        return res
    for nm in list(funcs):
        chk(nm, set())
    return cache


def state_object_libs(ed):
    """{state_object_id: [ {offset, size} per DXIL library, in subobject order ]}.

    Replays wpix_core's blob-offset simulation (the global read order from
    CreateAppResources_000) but records the offset of every Read that occurs inside a
    CreateStateObject_<id>() function."""
    funcs = ed._funcs
    has_read = _has_read_map(funcs)
    so_re = re.compile(r"CreateStateObject_(\d+)")
    offset = [0]
    out = {}
    def walk(name):
        m = so_re.fullmatch(name)
        cur_so = int(m.group(1)) if m else None
        for ln in funcs.get(name, []):
            rm = _READ_RE.search(ln)
            if rm:
                if cur_so is not None:
                    out.setdefault(cur_so, []).append({"offset": offset[0],
                                                       "size": int(rm.group(1))})
                offset[0] += int(rm.group(1))
                continue
            cm = _CALL_RE.match(ln)
            if cm:
                cn = cm.group(1)
                if cn in funcs and has_read.get(cn) and cn != name:
                    walk(cn)
    if "CreateAppResources_000" in funcs:
        walk("CreateAppResources_000")
    return out


def state_object_meta(ed):
    """{sid: {'libs':[{exports:[...]}], 'hitgroups':[...]}} parsed from CreatePSOs text."""
    text = ""
    import glob
    for f in glob.glob(os.path.join(ed.dir, "CreatePSOs*.cpp")):
        text += open(f, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"void CreateStateObject_(\d+)\(\)(.*?)\n\}", text, re.S):
        sid, body = int(m.group(1)), m.group(2)
        libs = []
        # each lib: a Read followed (a few lines later) by its DXIL_LIBRARY_DESC export list
        for dm in re.finditer(r"D3D12_DXIL_LIBRARY_DESC\s+dxilLibDesc_\d+_\d+\s*=\s*"
                              r"\{\s*dxilLib_\d+_\d+,\s*(\d+),\s*(\w+)", body):
            nexp, exparr = int(dm.group(1)), dm.group(2)
            exports = []
            if nexp and exparr != "nullptr":
                em = re.search(re.escape(exparr) + r"\[\]\s*=\s*\{(.*?)\};", body, re.S)
                if em:
                    exports = re.findall(r'LR?"\(([^)]*)\)"', em.group(1))[:nexp * 2:2] \
                              or re.findall(r'LR?"\(([^)]*)\)"', em.group(1))
            libs.append({"n_exports": nexp, "exports": exports})
        hitgroups = re.findall(r'D3D12_HIT_GROUP_DESC\s+\w+\s*=\s*\{\s*LR?"\(([^)]*)\)"', body)
        out[sid] = {"libs": libs, "hitgroups": hitgroups}
    return out


def dispatchrays_state_objects(ed):
    """[{global_id, state_object}] for each DispatchRays, in command order."""
    import glob
    text = ""
    for f in glob.glob(os.path.join(ed.dir, "CommandLists_*.cpp")):
        text += open(f, encoding="utf-8", errors="replace").read()
    out = []
    cur_so = None; last_gid = None
    for ln in text.split("\n"):
        m = re.search(r"SetPipelineState1\(GetStateObject\((\d+)\)", ln)
        if m: cur_so = int(m.group(1))
        m = re.search(r"//\s*GlobalId\s*=\s*(\d+)", ln)
        if m: last_gid = int(m.group(1))
        if "->DispatchRays(" in ln:
            out.append({"global_id": last_gid, "state_object": cur_so})
    return out


def rt_report(wpix):
    ed = w.load_export(w.export_full(wpix))
    libs = state_object_libs(ed)
    meta = state_object_meta(ed)
    drays = dispatchrays_state_objects(ed)
    # marker names for the DispatchRays global ids
    markers = {e["global_id"]: e["marker"]
               for e in w.find_shader_events(wpix) if e["name"] == "DispatchRays"}
    passes = []
    for dr in drays:
        sid = dr["state_object"]
        entry = {"global_id": dr["global_id"], "marker": markers.get(dr["global_id"], ""),
                 "state_object": sid, "libs": []}
        for i, lib in enumerate(libs.get(sid, [])):
            blob = _xpress_decompress_at(ed.dir, lib["offset"], lib["size"])
            info = container_compile_info(blob)
            m = meta.get(sid, {}).get("libs", [])
            entry["libs"].append({
                "size": lib["size"],
                "hash": info.get("shader_hash"),
                "target": info.get("target"),
                "debug_name": info.get("debug_name"),
                "exports": (m[i]["exports"] if i < len(m) else []),
            })
        entry["hitgroups"] = meta.get(sid, {}).get("hitgroups", [])
        passes.append(entry)
    return passes


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    print("Reading RTXPT raytracing pipelines ...")
    a = rt_report(rtxpt)
    print("Reading Unity raytracing pipelines ...\n")
    b = rt_report(unity)

    def dump(tag, passes):
        print(f"==================== {tag} ====================")
        for p in passes:
            print(f"DispatchRays gid={p['global_id']}  marker='{p['marker']}'  "
                  f"stateObject={p['state_object']}  ({len(p['libs'])} DXIL libraries)")
            for L in p["libs"]:
                exp = ", ".join(L["exports"]) if L["exports"] else "<all (no export filter)>"
                print(f"   hash={L['hash']}  target={L['target']}  bytes={L['size']}")
                print(f"      debug_name={L['debug_name']}")
                print(f"      exports={exp}")
            if p["hitgroups"]:
                print(f"   hit groups: {', '.join(p['hitgroups'])}")
            print()

    dump("RTXPT", a)
    dump("UNITY", b)

    ha = {L["hash"] for p in a for L in p["libs"]}
    hb = {L["hash"] for p in b for L in p["libs"]}
    print("==================== HASH COMPARISON ====================")
    print(f"RTXPT DXIL-library hashes ({len(ha)}): {sorted(ha)}")
    print(f"Unity DXIL-library hashes ({len(hb)}): {sorted(hb)}")
    print(f"Shared: {sorted(ha & hb) or 'NONE'}")
    print()
    if ha & hb:
        print("VERDICT: some raytracing DXIL libraries are byte-identical (shared hash above).")
    else:
        print("VERDICT: NO raytracing DXIL library is byte-identical between the two captures —")
        print("the ray pipelines are compiled/structured differently (lib count, exports, hit groups).")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
