"""List every texture bound in the bindless texture table (t_BindlessTextures[], SRV
register t0 space2 -- see Shaders/Bindings/SceneBindings.hlsli) at the "PathTrace"
DispatchRays in each of the two Other/ captures, and diff the two lists.

t_BindlessTextures is an unbounded descriptor-table range (VK_BINDING(1,1) Texture2D
t_BindlessTextures[] : register(t0, space2), alongside t_BindlessBuffers[] at space1);
material texture indices (see compare_materials.py's TEX_FIELDS) index directly into it.

The two engines lay this out differently: Unity gives space1/space2 each their own root
parameter, so the two SRV ranges are unambiguous. RTXPT packs both into a *single* root
parameter with two overlapping unbounded ranges (both `OffsetInDescriptorsFromTableStart`
= 0), so the declared HLSL register space can't disambiguate a given heap slot -- buffers
and textures are simply interleaved in the same physical bindless heap region, and only
the actual bound SRV *view shape* (Tex2D/TexCube/... vs Buffer) tells them apart. So this
walks every unbounded-SRV root table bound at the PathTrace DispatchRays, and classifies
each populated descriptor slot as a "bindless texture" purely by its view shape -- this
works uniformly for both engines' root-signature layouts.

Texture indices are engine-specific bindless *heap slots* (not comparable positionally
across engines), so the two lists are diffed order-independently by resource debug name.

Run from tools/wpix_mcp:  python compare_bindless_textures.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, os, sys
import wpix_core as w
from wpix_core import _render_table

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")

# D3D12 uses 0xFFFFFFFF (UINT_MAX) as the unbounded-range sentinel for NumDescriptors.
UNBOUNDED_THRESHOLD = 1_000_000
TEXTURE_SHAPES = {"Tex2D", "Tex2DArray", "TexCube", "TexCubeArray", "Tex3D", "Tex1D", "Tex1DArray"}


def pathtrace_dispatch(wpix):
    """(ExportData, dispatch dict) for the 'PathTrace' (FILL) DispatchRays in `wpix`.

    name_filter="PathTrace" also substring-matches the separate "PathTracePrePass"
    marker (RTXPT and Unity both emit both), so the marker must be checked to END
    with "PathTrace" (not just contain it) -- otherwise the coherent prepass gets
    picked instead of FILL. RTXPT prefixes markers with a PIX legacy-API notice
    ("<deprecated - use pix3.h instead> PathTrace"), Unity nests them under the
    render-graph path ("ExecuteRenderGraph/PathTrace/PathTrace") -- both still end
    with the exact "PathTrace" token, while "...PathTracePrePass" does not."""
    ed = w.load_export(w.export_full(wpix))
    events = [e for e in w.find_shader_events(wpix, name_filter="PathTrace")
              if e["name"] == "DispatchRays" and e["marker"].endswith("PathTrace")]
    if not events:
        raise RuntimeError(f"no 'PathTrace' DispatchRays found in {wpix}")
    gid = events[0]["global_id"]
    d = next((x for x in ed.dispatches if x.get("global_id") == gid), None)
    if d is None:
        raise RuntimeError(f"DispatchRays gid={gid} not found in exported command list")
    return ed, d, len(events)


def bindless_textures(wpix):
    """[{slot, name, format, width, height, mips}], sorted by heap slot, for every
    Texture-shaped SRV in an unbounded bindless descriptor table bound at the
    PathTrace DispatchRays (buffers in the same/sibling table are excluded by shape)."""
    ed, d, n_events = pathtrace_dispatch(wpix)
    sig = ed.root_sigs.get(d["root_signature"], {})
    table_starts = d["root_tables"]
    other_starts = set(table_starts.values())
    rows = {}
    for param, start in table_starts.items():
        ranges = sig.get(param, {}).get("ranges") or []
        if not any(rg["range_type"] == "SRV" and rg["num"] >= UNBOUNDED_THRESHOLD for rg in ranges):
            continue
        for desc in ed.table_slots(start, other_starts - {start}, max_count=16384):
            if desc["shape"] not in TEXTURE_SHAPES:
                continue
            rows[desc["slot"]] = desc
    out = []
    for slot, desc in sorted(rows.items()):
        r = ed.resources.get(desc["resource"], {})
        out.append({"slot": slot, "resource_id": desc["resource"],
                    "resource_name": r.get("name"), "res_format": r.get("format"),
                    "width": r.get("width"), "height": r.get("height"), "mips": r.get("mips")})
    return out, n_events


def _row(item):
    dims = f'{item["width"]}x{item["height"]}' if item["width"] else "-"
    return [item["slot"], item["resource_name"] or f'<resource {item["resource_id"]}>',
            item["res_format"] or "-", dims, item["mips"] or "-"]


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    print("Reading RTXPT bindless texture table ...")
    A, na = bindless_textures(rtxpt)
    print("Reading Unity bindless texture table ...\n")
    B, nb = bindless_textures(unity)
    if na > 1 or nb > 1:
        print(f"NOTE: multiple 'PathTrace' DispatchRays found (RTXPT={na}, UNITY={nb}); "
              "using the first one of each.\n")

    headers = ["slot", "name", "format", "dims", "mips"]
    print(f"== RTXPT bindless textures ({len(A)}) ==")
    print(_render_table(headers, [_row(it) for it in A]) if A else "  (none)")
    print(f"\n== UNITY bindless textures ({len(B)}) ==")
    print(_render_table(headers, [_row(it) for it in B]) if B else "  (none)")

    # order-independent diff by resource debug name (heap slots are not comparable
    # across engines: RTXPT uses descriptor-heap offsets, Unity small sequential indices).
    # RTXPT names carry the full source .dds path (e.g. 'F:/RTXPT/Assets/.../foo_diff.dds'),
    # Unity keeps just the asset's base name ('foo_diff') -- normalize to compare.
    def norm(name):
        base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if base.lower().endswith(".dds"):
            base = base[:-4]
        return base.lower()

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
        if (ia["res_format"], ia["width"], ia["height"], ia["mips"]) != \
           (ib["res_format"], ib["width"], ib["height"], ib["mips"]):
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
        print(f"\n   -- only in RTXPT ({len(only_a)}) --")
        for name in only_a:
            print(f"      {name}")
    if only_b:
        print(f"\n   -- only in UNITY ({len(only_b)}) --")
        for name in only_b:
            print(f"      {name}")

    print()
    if only_a or only_b or mismatched:
        print("VERDICT: bindless texture sets differ between the two engines (see lists above).")
    else:
        print("VERDICT: identical set of bindless textures (by name/format/dims) in both engines; "
              "heap slot order differs (engine-specific bindless indices).")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
