"""Compare the size/format of every NON-bindless texture (SRV or UAV) bound at the
"PathTrace" DispatchRays in the two Other/ captures -- i.e. the fixed-slot G-buffer,
denoiser, feedback and history render targets the path tracer reads/writes directly,
as opposed to the scene's unbounded bindless material texture array (see
compare_bindless_textures.py, which covers that array specifically and is excluded
here by removing every slot bindless_slots() reports).

Deliberately does NOT use wpix_core.binding_table(): for a table whose declared size
is the D3D12 unbounded sentinel (the bindless param), binding_table's "size>0" path
filters the *entire* sorted descriptor set by [start, start+size) with no contiguity
check, so on a table that never legitimately ends it walks straight through gaps into
descriptor slots some later, unrelated dispatch reused for its own resources (e.g.
tonemap/bloom scratch textures show up as "bound at PathTrace").

Two more root-signature quirks this has to work around (found by inspecting a real
capture, not documented anywhere):
  * `d["root_tables"]` can carry STALE entries for root params the target dispatch's
    OWN root signature doesn't even declare as a descriptor table (32-bit-constants /
    root-CBV params) -- these are just whatever a root signature bound EARLIER in the
    same PopulateCommandList_* function last wrote to that root-parameter slot, before
    the signature was switched. Only params the *current* signature declares with an
    SRV/UAV range are real.
  * A table's declared start slot can be a legitimately UNPOPULATED descriptor (some
    of a big shared root signature's many packed ranges simply aren't used by this
    particular dispatch) -- so a strict "must start populated, walk contiguously"
    approach (ed.table_slots(), fine for the always-densely-packed bindless table)
    returns nothing even though real data exists a few slots later. For a table with a
    known FINITE size we instead iterate the declared [start, start+size) range
    directly and just skip unpopulated slots, never mistaking a null slot for the end.

EXPECTED ASYMMETRY: a large "-- only in RTXPT --" list is an artifact of binding
philosophy, not a real gap. Neither engine uses DXR local root signatures (verified:
both state objects contain a single GLOBAL_ROOT_SIGNATURE subobject and their SBT
records are bare 32-byte shader identifiers with no local root arguments). RTXPT sets
one big shared root signature for many passes, so at PathTrace it carries dozens of
bound-but-unused textures (denoiser guides, RR/G-buffer/output targets, dummy black/
white textures) that no FILL shader's RDAT resource list references; Unity binds
per-pass and only binds the registers the FILL shaders actually use. What matters is
that every texture in the shaders' used set appears in BOTH lists and matches.

Run from tools/wpix_mcp:  python compare_pathtrace_textures.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, os, sys
import wpix_core as w
from wpix_core import _render_table, _slot_register
from compare_bindless_textures import pathtrace_dispatch, bindless_slots, TEXTURE_SHAPES, UNBOUNDED_THRESHOLD

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")


def pathtrace_textures(wpix):
    """[{slot, reg, usage, name, format, width, height, depth, mips}], sorted by slot,
    for every Texture-shaped SRV/UAV bound at the PathTrace DispatchRays that is NOT
    part of the unbounded bindless material texture table."""
    ed, d, n_events = pathtrace_dispatch(wpix)
    exclude = bindless_slots(ed, d)
    sig = ed.root_sigs.get(d["root_signature"], {})
    table_starts = d["root_tables"]
    other_starts = set(table_starts.values())
    rows = {}
    for param, start in table_starts.items():
        pinfo = sig.get(param, {})
        ranges = pinfo.get("ranges") or []
        # Only params the CURRENT root signature actually declares as a descriptor
        # table containing an SRV/UAV range are real; 32BIT_CONSTANTS/CBV/SAMPLER
        # params (empty ranges here) are either not textures or stale carry-over
        # bindings from a different root signature used earlier in the command list.
        if not any(rg["range_type"] in ("SRV", "UAV") for rg in ranges):
            continue
        table_size = pinfo.get("table_size") or 0
        unbounded = table_size >= UNBOUNDED_THRESHOLD
        if unbounded:
            descs = ed.table_slots(start, other_starts - {start}, max_count=16384)
        else:
            descs = [ed.descriptors[s] for s in range(start, start + table_size) if s in ed.descriptors]
        for desc in descs:
            slot = desc["slot"]
            if slot in exclude or desc["shape"] not in TEXTURE_SHAPES:
                continue
            r = ed.resources.get(desc["resource"], {})
            _, reg, _ = _slot_register(ranges, slot - start)
            usage = "UAV" if desc["view"] == "UnorderedAccess" else "SRV"
            reg_label = f'{"u" if usage == "UAV" else "t"}{reg}' if reg is not None else "-"
            if slot in rows:
                rows[slot]["usage"] += "+" + usage
            else:
                rows[slot] = {"slot": slot, "reg": reg_label, "usage": usage,
                              "resource_id": desc["resource"], "resource_name": r.get("name"),
                              "res_format": r.get("format"), "width": r.get("width"),
                              "height": r.get("height"), "depth": r.get("array_or_depth"),
                              "mips": r.get("mips")}
    return sorted(rows.values(), key=lambda r: r["slot"]), n_events


def _row(item):
    dims = f'{item["width"]}x{item["height"]}' if item["width"] else "-"
    if item["depth"] and item["depth"] > 1:
        dims += f'x{item["depth"]}'
    return [item["slot"], item["reg"] or "-", item["usage"],
            item["resource_name"] or f'<resource {item["resource_id"]}>',
            item["res_format"] or "-", dims, item["mips"] or "-"]


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    print("Reading RTXPT PathTrace-bound textures ...")
    A, na = pathtrace_textures(rtxpt)
    print("Reading Unity PathTrace-bound textures ...\n")
    B, nb = pathtrace_textures(unity)

    headers = ["slot", "reg", "usage", "name", "format", "dims", "mips"]
    print(f"== RTXPT non-bindless textures ({len(A)}) ==")
    print(_render_table(headers, [_row(it) for it in A]) if A else "  (none)")
    print(f"\n== UNITY non-bindless textures ({len(B)}) ==")
    print(_render_table(headers, [_row(it) for it in B]) if B else "  (none)")

    # order-independent diff by resource debug name (register/slot assignment is
    # engine/root-signature-layout specific, not comparable positionally)
    def norm(name):
        return name.strip().lower()

    by_name_a = {norm(it["resource_name"]): it for it in A if it["resource_name"]}
    by_name_b = {norm(it["resource_name"]): it for it in B if it["resource_name"]}
    names_a, names_b = set(by_name_a), set(by_name_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    shared = sorted(names_a & names_b)

    print(f"\n== Comparison (matched by resource name) ==")
    print(f"   RTXPT: {len(A)} textures ({len(names_a)} named)")
    print(f"   UNITY: {len(B)} textures ({len(names_b)} named)")
    print(f"   shared names: {len(shared)}   RTXPT-only: {len(only_a)}   UNITY-only: {len(only_b)}")

    mismatched = []
    for name in shared:
        ia, ib = by_name_a[name], by_name_b[name]
        if (ia["res_format"], ia["width"], ia["height"], ia["depth"], ia["mips"]) != \
           (ib["res_format"], ib["width"], ib["height"], ib["depth"], ib["mips"]):
            mismatched.append(name)
    if mismatched:
        print(f"\n   {len(mismatched)} shared texture(s) differ in format/dims/mips:")
        rows = []
        for name in mismatched:
            ia, ib = by_name_a[name], by_name_b[name]
            rows.append([name, ia["res_format"] or "-", f'{ia["width"]}x{ia["height"]}',
                        ib["res_format"] or "-", f'{ib["width"]}x{ib["height"]}'])
        print(_render_table(["name", "RTXPT format", "RTXPT dims", "UNITY format", "UNITY dims"], rows))

    if only_a:
        print(f"\n   -- only in RTXPT ({len(only_a)}) --  "
              f"(bound-but-unused via RTXPT's shared root signature -- see module docstring)")
        for name in only_a:
            it = by_name_a[name]
            print(f"      {it['resource_name']}  ({it['res_format']}, {it['width']}x{it['height']}, {it['usage']})")
    if only_b:
        print(f"\n   -- only in UNITY ({len(only_b)}) --")
        for name in only_b:
            it = by_name_b[name]
            print(f"      {it['resource_name']}  ({it['res_format']}, {it['width']}x{it['height']}, {it['usage']})")

    print()
    if only_a or only_b or mismatched:
        print("VERDICT: PathTrace's non-bindless texture bindings differ between the two engines (see lists above).")
    else:
        print("VERDICT: identical set of non-bindless PathTrace textures (by name/format/dims) in both engines.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
