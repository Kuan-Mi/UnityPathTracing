"""Dump the DXC compile defines of every DXIL library bound at the two PathTrace
DispatchRays in both kitchen captures, and diff RTXPT vs Unity per pass.

RTXPT libs carry embedded debug info (-Qembed_debug build); Unity libs resolve via
the <hash>.pdb side-cars in UnityProject/ShaderPDB.

Run from tools/wpix_mcp:  python compare_rt_macros.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, os, subprocess, sys
import wpix_core as w
from wpix_core import (_xpress_decompress_at, container_compile_info, _parse_compile_args,
                       _compile_args_from_pdb, _default_pdb_dir, find_dxc, _cache_root)
import compare_rt_shaders as cr

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt-kitchen.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity-kitchen.wpix")


def lib_args(blob, dxc, pdb_dir):
    info = container_compile_info(blob)
    h = info.get("shader_hash")
    tmp = os.path.join(_cache_root(), f"rtlib_{h}.bin")
    with open(tmp, "wb") as fh:
        fh.write(blob)
    proc = subprocess.run([dxc, "-dumpbin", tmp], capture_output=True, text=True)
    args = _parse_compile_args(proc.stdout) if proc.returncode == 0 else {}
    src = "embedded"
    if not args:
        args, _, _ = _compile_args_from_pdb(h, pdb_dir, dxc)
        src = "external_pdb" if args else "none"
    return h, info.get("debug_name", ""), (args or {}), src


def report(wpix):
    ed = w.load_export(w.export_full(wpix))
    libs = cr.state_object_libs(ed)
    drays = cr.dispatchrays_state_objects(ed)
    markers = {e["global_id"]: e["marker"].rsplit("/", 1)[-1]
               for e in w.find_shader_events(wpix) if e["name"] == "DispatchRays"}
    dxc = find_dxc()
    pdb_dir = _default_pdb_dir()
    out = {}  # pass-leaf -> [lib dicts]
    for dr in drays:
        lf = markers.get(dr["global_id"], "?")
        if lf in out:
            continue
        rows = []
        for lib in libs.get(dr["state_object"], []):
            blob = _xpress_decompress_at(ed.dir, lib["offset"], lib["size"])
            h, dbg, args, src = lib_args(blob, dxc, pdb_dir)
            rows.append({"hash": h, "debug": os.path.basename(dbg), "src": src,
                         "entry": args.get("entry"), "target": args.get("target"),
                         "defines": sorted(args.get("defines") or []),
                         "flags": sorted(set(args.get("flags") or []))})
        out[lf] = rows
    return out


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    a, b = report(rtxpt), report(unity)
    for lf in sorted(set(a) | set(b)):
        print(f"\n######## {lf} ########")
        for tag, rows in (("RTXPT", a.get(lf, [])), ("UNITY", b.get(lf, []))):
            for r in rows:
                print(f"[{tag}] {r['debug'] or r['hash']}  ({r['src']}, {r['target']})")
                print(f"        defines: {r['defines']}")
                fl = [f for f in r["flags"] if f not in ("-Qembed_debug",)]
                print(f"        flags:   {fl}")
        # set-level define diff across all libs of the pass
        da = set(d for r in a.get(lf, []) for d in r["defines"])
        db = set(d for r in b.get(lf, []) for d in r["defines"])
        if da != db:
            print(f"   >>> defines only in RTXPT libs: {sorted(da - db)}")
            print(f"   >>> defines only in UNITY libs: {sorted(db - da)}")
        else:
            print("   >>> union of defines identical across both sides")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
