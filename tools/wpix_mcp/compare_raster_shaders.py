"""Compare the rasterization (graphics) shaders executed in the two Other/ captures.

Raster passes run via SetPipelineState(graphics PSO) + DrawInstanced/DrawIndexedInstanced.
A graphics PSO bundles its VS+PS DXIL containers into a single blob read from the global
resource reader. describe_shader only handles compute PSOs, so this walks the exported
CreateGraphicsPipelineState_*() functions, replays the global resources.bin read order to
find each PSO blob's byte offset, decompresses it, slices the VS and PS sub-containers
(sizes taken from the psoDesc.VS/.PS fields) and reads each container's HASH + target.
Draws are matched to their active graphics PSO from the command list, with marker names.

Run from tools/wpix_mcp:  python compare_raster_shaders.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, glob, os, re, sys
import wpix_core as w
from wpix_core import _READ_RE, _CALL_RE, container_compile_info, _xpress_decompress_at

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")

GFX_FUNC_RE = re.compile(r"CreateGraphicsPipelineState_(\d+)")
DRAW_RE = re.compile(r"->Draw(?:Indexed)?Instanced\(")


def graphics_pso_sizes(ed):
    """{pso_id: [(stage, size), ...]} for the DXIL stages bundled in each graphics PSO,
    in blob order (VS first, then PS / others), from the psoDesc.<stage> = {ptr, size} lines."""
    text = "".join(open(f, encoding="utf-8", errors="replace").read()
                    for f in glob.glob(os.path.join(ed.dir, "CreatePSOs*.cpp")))
    out = {}
    for m in re.finditer(r"void CreateGraphicsPipelineState_(\d+)\(\)(.*?)\n\}", text, re.S):
        pid, body = int(m.group(1)), m.group(2)
        stages = []
        for sm in re.finditer(r"psoDesc\.(VS|PS|DS|HS|GS)\s*=\s*\{[^,}]+,\s*(\d+)\s*\}", body):
            stages.append((sm.group(1), int(sm.group(2))))
        out[pid] = stages
    return out


def graphics_pso_blobs(ed):
    """{pso_id: (byte_offset, compressed_size)} of the combined VS+PS blob, found by replaying
    the global resource read order (same simulation wpix_core uses for compute PSOs)."""
    funcs = ed._funcs
    has_read = {}

    def chk(name, stack):
        if name in has_read:
            return has_read[name]
        if name in stack or name not in funcs:
            return False
        stack.add(name)
        res = False
        for ln in funcs[name]:
            if _READ_RE.search(ln):
                res = True
                break
            cm = _CALL_RE.match(ln)
            if cm and cm.group(1) in funcs and cm.group(1) != name and chk(cm.group(1), stack):
                res = True
                break
        stack.discard(name)
        has_read[name] = res
        return res

    for nm in list(funcs):
        chk(nm, set())

    offset = [0]
    blobs = {}

    def walk(name):
        gm = GFX_FUNC_RE.fullmatch(name)
        cur = int(gm.group(1)) if gm else None
        for ln in funcs.get(name, []):
            rm = _READ_RE.search(ln)
            if rm:
                if cur is not None and cur not in blobs:
                    blobs[cur] = (offset[0], int(rm.group(1)))
                offset[0] += int(rm.group(1))
                continue
            cm = _CALL_RE.match(ln)
            if cm and cm.group(1) in funcs and has_read.get(cm.group(1)) and cm.group(1) != name:
                walk(cm.group(1))

    if "CreateAppResources_000" in funcs:
        walk("CreateAppResources_000")
    return blobs


def draw_pso_ids(ed, gfx_ids):
    """[{global_id, pso}] for each Draw, with the active graphics PSO (compute PSO sets ignored)."""
    text = "".join(open(f, encoding="utf-8", errors="replace").read()
                    for f in glob.glob(os.path.join(ed.dir, "CommandLists_*.cpp")))
    out = []
    cur_pso = None
    last_gid = None
    for ln in text.split("\n"):
        m = re.search(r"SetPipelineState\(GetPipelineState\((\d+)\)", ln)
        if m:
            pid = int(m.group(1))
            cur_pso = pid if pid in gfx_ids else cur_pso  # keep last graphics PSO bound
        m = re.search(r"//\s*GlobalId\s*=\s*(\d+)", ln)
        if m:
            last_gid = int(m.group(1))
        if DRAW_RE.search(ln):
            out.append({"global_id": last_gid, "pso": cur_pso})
    return out


def raster_report(wpix):
    ed = w.load_export(w.export_full(wpix))
    sizes = graphics_pso_sizes(ed)
    blobs = graphics_pso_blobs(ed)
    draws = draw_pso_ids(ed, set(sizes))
    markers = {e["global_id"]: e["marker"]
               for e in w.find_shader_events(wpix)
               if e["name"] in ("DrawInstanced", "DrawIndexedInstanced")}

    # hash each graphics PSO's stages once
    pso_stages = {}
    for pid, stages in sizes.items():
        if pid not in blobs:
            continue
        off, csize = blobs[pid]
        data = _xpress_decompress_at(ed.dir, off, csize)
        pos = 0
        rows = []
        for stage, sz in stages:
            info = container_compile_info(data[pos:pos + sz])
            rows.append({"stage": stage, "size": sz,
                         "hash": info.get("shader_hash"), "target": info.get("target")})
            pos += sz
        pso_stages[pid] = rows

    passes = []
    for d in draws:
        passes.append({"global_id": d["global_id"], "pso": d["pso"],
                       "marker": markers.get(d["global_id"], ""),
                       "stages": pso_stages.get(d["pso"], [])})
    return passes


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    print("Reading RTXPT graphics pipelines ...")
    a = raster_report(rtxpt)
    print("Reading Unity graphics pipelines ...\n")
    b = raster_report(unity)

    def dump(tag, passes):
        print(f"==================== {tag} ====================")
        for p in passes:
            leaf = p["marker"].rsplit("/", 1)[-1] if p["marker"] else "?"
            print(f"Draw gid={p['global_id']}  pso={p['pso']}  marker='{leaf}'")
            for s in p["stages"]:
                print(f"   {s['stage']}  hash={s['hash']}  target={s['target']}  bytes={s['size']}")
            if not p["stages"]:
                print("   <no graphics PSO resolved>")
        print()

    dump("RTXPT", a)
    dump("UNITY", b)

    # None = a tiny driver/blit PSO whose VS/PS isn't a parseable DXIL container.
    ha = {s["hash"] for p in a for s in p["stages"] if s["hash"]}
    hb = {s["hash"] for p in b for s in p["stages"] if s["hash"]}
    print("==================== HASH COMPARISON ====================")
    print(f"RTXPT raster shader hashes ({len(ha)}): {sorted(ha)}")
    print(f"Unity raster shader hashes ({len(hb)}): {sorted(hb)}")
    print(f"Shared:     {sorted(ha & hb) or 'NONE'}")
    print(f"RTXPT-only: {sorted(ha - hb)}")
    print(f"Unity-only: {sorted(hb - ha)}")
    print("(hashes are the DXIL program hash — invariant to embedded vs side-car debug,")
    print(" so a match means bit-identical shader regardless of container size.)")
    print()
    if ha & hb:
        print(f"VERDICT: {len(ha & hb)} rasterization shader(s) are bit-identical across captures; "
              "the rest are engine-specific raster passes (blits, bloom, UI) and differ as expected.")
    else:
        print("VERDICT: NO rasterization shader is shared — the raster passes (blits, tonemap, UI) "
              "come from different engines and are expected to differ.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
