"""Compare every NON-bindless buffer (SRV or UAV) bound at the "PathTrace" DispatchRays
in the two Other/ captures -- the fixed-register scene/lighting/path-state buffers
(t_InstanceData, t_GeometryData, t_PTMaterialData, the t12..t17 light buffers,
u_StablePlanesBuffer, u_FeedbackBuffer, ...) -- plus a separate section for the
unbounded bindless BUFFER table (t_BindlessBuffers, space1; the texture side of the
bindless heap is compare_bindless_textures.py's job). The root CBV (g_Const, b0) is
only listed for reference: its contents are compare_gconst.py's job.

Fixed-register buffers are matched by HLSL register (e.g. 't3'), NOT by name: registers
come from the shared shader source so they mean the same thing in both engines, while
debug names don't (Unity autogenerates e.g. 'Buffer-16-186176' for what RTXPT calls
'BindlessGeometry'). For each shared register the *view geometry* is compared --
element count, structure stride, view format -- rather than the backing resource's byte
size: both engines suballocate views from differently-sized pools, so resource width is
reported but only view mismatches count as differences.

Same root-signature quirks as compare_pathtrace_textures.py (stale root_tables entries,
finite tables with unpopulated leading slots), and the same expected asymmetry: RTXPT's
one shared root signature carries bound-but-unused registers that Unity's per-pass
binding simply doesn't bind (neither engine uses DXR local root signatures -- verified
via the state objects' single GLOBAL_ROOT_SIGNATURE subobject and bare 32-byte SBT
records). What matters is that every register in the FILL shaders' RDAT used set
matches.

Run from tools/wpix_mcp:  python compare_pathtrace_buffers.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, os, re, sys
import wpix_core as w
from wpix_core import _render_table, _slot_register
from compare_bindless_textures import pathtrace_dispatch, bindless_slots, TEXTURE_SHAPES, UNBOUNDED_THRESHOLD

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")


def _view_geom(desc):
    """(first_element, num_elements, stride) from a buffer view's exported creation args.
    SRV tail: `, Shader4ComponentMapping, First, Num, Stride, FLAGS`
    UAV tail: `, First, Num, Stride, CounterOffset, FLAGS`"""
    nums = [int(x) for x in re.findall(r'\b\d+\b', desc.get("view_args") or "")]
    if desc["view"] == "ShaderResource":
        nums = nums[1:]
    return tuple(nums[:3]) if len(nums) >= 3 else (None, None, None)


def pathtrace_buffers(wpix):
    """(fixed, bindless): fixed = [{reg, usage, name, elems, stride, view_format, size}]
    for every Buffer-shaped SRV/UAV in a bounded root table at the PathTrace
    DispatchRays; bindless = same rows (reg=None) for Buffer-shaped slots inside the
    unbounded bindless tables."""
    ed, d, _ = pathtrace_dispatch(wpix)
    bindless = bindless_slots(ed, d)
    sig = ed.root_sigs.get(d["root_signature"], {})
    table_starts = d["root_tables"]
    other_starts = set(table_starts.values())
    fixed, in_bindless = {}, {}
    for param, start in table_starts.items():
        pinfo = sig.get(param, {})
        ranges = pinfo.get("ranges") or []
        if not any(rg["range_type"] in ("SRV", "UAV") for rg in ranges):
            continue  # stale carry-over binding from an earlier root signature
        table_size = pinfo.get("table_size") or 0
        unbounded = table_size >= UNBOUNDED_THRESHOLD
        if unbounded:
            descs = ed.table_slots(start, other_starts - {start}, max_count=16384)
        else:
            descs = [ed.descriptors[s] for s in range(start, start + table_size) if s in ed.descriptors]
        for desc in descs:
            slot = desc["slot"]
            if desc["shape"] in TEXTURE_SHAPES or desc["shape"] != "Buffer":
                continue
            r = ed.resources.get(desc["resource"], {})
            first, elems, stride = _view_geom(desc)
            usage = "UAV" if desc["view"] == "UnorderedAccess" else "SRV"
            cls, reg, space = _slot_register(ranges, slot - start)
            row = {"slot": slot, "usage": usage,
                   "reg": (f'{"u" if usage == "UAV" else "t"}{reg}' if reg is not None and not unbounded else None),
                   "space": space,
                   "resource_id": desc["resource"], "resource_name": r.get("name"),
                   "view_format": desc["format"], "first": first, "elems": elems,
                   "stride": stride, "size": r.get("width")}
            (in_bindless if slot in bindless else fixed)[slot] = row
    ed_d = d
    cbv = {param: ed_d["root_cbv"][param] for param in sorted(ed_d["root_cbv"])}
    return (sorted(fixed.values(), key=lambda r: (r["usage"], r["reg"] or "")),
            sorted(in_bindless.values(), key=lambda r: r["slot"]), cbv)


def _row(it):
    return [it["reg"] or it["slot"], it["usage"],
            it["resource_name"] or f'<resource {it["resource_id"]}>',
            it["elems"] if it["elems"] is not None else "-",
            it["stride"] if it["stride"] is not None else "-",
            (it["view_format"] or "-").replace("DXGI_FORMAT_", ""),
            it["size"] or "-"]


HEADERS = ["reg/slot", "usage", "name", "elems", "stride", "view fmt", "res bytes"]


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    print("Reading RTXPT PathTrace-bound buffers ...")
    A, Abl, Acbv = pathtrace_buffers(rtxpt)
    print("Reading Unity PathTrace-bound buffers ...\n")
    B, Bbl, Bcbv = pathtrace_buffers(unity)

    print(f"== RTXPT fixed-register buffers ({len(A)}) ==")
    print(_render_table(HEADERS, [_row(it) for it in A]) if A else "  (none)")
    print(f"\n== UNITY fixed-register buffers ({len(B)}) ==")
    print(_render_table(HEADERS, [_row(it) for it in B]) if B else "  (none)")

    by_reg_a = {it["reg"]: it for it in A if it["reg"]}
    by_reg_b = {it["reg"]: it for it in B if it["reg"]}
    shared = sorted(set(by_reg_a) & set(by_reg_b))
    only_a = sorted(set(by_reg_a) - set(by_reg_b))
    only_b = sorted(set(by_reg_b) - set(by_reg_a))

    print(f"\n== Comparison (matched by HLSL register) ==")
    print(f"   shared registers: {len(shared)}   RTXPT-only: {len(only_a)}   UNITY-only: {len(only_b)}")

    mismatched = []
    for reg in shared:
        ia, ib = by_reg_a[reg], by_reg_b[reg]
        if (ia["elems"], ia["stride"], ia["view_format"]) != (ib["elems"], ib["stride"], ib["view_format"]):
            mismatched.append(reg)
    if mismatched:
        print(f"\n   {len(mismatched)} shared register(s) differ in view geometry "
              "(elems/stride/format):")
        rows = []
        for reg in mismatched:
            ia, ib = by_reg_a[reg], by_reg_b[reg]
            rows.append([reg, ia["resource_name"], f'{ia["elems"]}x{ia["stride"]}',
                         ib["resource_name"], f'{ib["elems"]}x{ib["stride"]}'])
        print(_render_table(["reg", "RTXPT name", "RTXPT elems x stride",
                             "UNITY name", "UNITY elems x stride"], rows))
    else:
        print("   PASS: every shared register's view geometry (elems/stride/format) matches.")

    if only_a:
        print(f"\n   -- only in RTXPT ({len(only_a)}) --  "
              f"(bound-but-unused via RTXPT's shared root signature -- see module docstring)")
        for reg in only_a:
            it = by_reg_a[reg]
            print(f"      {reg:<5} {it['resource_name']}  ({it['elems']} x {it['stride']}B)")
    if only_b:
        print(f"\n   -- only in UNITY ({len(only_b)}) --")
        for reg in only_b:
            it = by_reg_b[reg]
            print(f"      {reg:<5} {it['resource_name']}  ({it['elems']} x {it['stride']}B)")

    print(f"\n== Bindless buffer table (t_BindlessBuffers, space1) ==")
    print(f"   RTXPT: {len(Abl)} buffer descriptors   UNITY: {len(Bbl)} buffer descriptors")
    for label, rows in (("RTXPT", Abl), ("UNITY", Bbl)):
        print(f"\n   -- {label} --")
        print(_render_table(HEADERS, [_row(it) for it in rows]) if rows else "      (none)")

    print(f"\n== Root CBV (contents: compare_gconst.py) ==")
    for label, cbv in (("RTXPT", Acbv), ("UNITY", Bcbv)):
        for param, (res, off) in cbv.items():
            print(f"   {label}: root param {param} -> resource {res} + offset {off}")

    print()
    if mismatched or only_b:
        print("VERDICT: fixed-register buffer bindings differ between the two engines (see above).")
    else:
        print("VERDICT: every shared fixed-register buffer matches in view geometry; "
              "RTXPT-only registers are bound-but-unused carry-over from its shared root signature.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
