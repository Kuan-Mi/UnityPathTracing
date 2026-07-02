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


def bindless_slots(ed, d):
    """Set of descriptor-heap slots covered by any unbounded SRV root table bound at
    dispatch `d` (both the texture and buffer arrays -- shape, not space, tells them
    apart; see module docstring). Shared with compare_pathtrace_textures.py, which
    needs this set to EXCLUDE the bindless heap from the "regular" bound textures."""
    sig = ed.root_sigs.get(d["root_signature"], {})
    table_starts = d["root_tables"]
    other_starts = set(table_starts.values())
    slots = set()
    for param, start in table_starts.items():
        ranges = sig.get(param, {}).get("ranges") or []
        if not any(rg["range_type"] == "SRV" and rg["num"] >= UNBOUNDED_THRESHOLD for rg in ranges):
            continue
        for desc in ed.table_slots(start, other_starts - {start}, max_count=16384):
            slots.add(desc["slot"])
    return slots


def bindless_textures(wpix):
    """[{slot, name, format, width, height, mips}], sorted by heap slot, for every
    Texture-shaped SRV in an unbounded bindless descriptor table bound at the
    PathTrace DispatchRays (buffers in the same/sibling table are excluded by shape)."""
    ed, d, n_events = pathtrace_dispatch(wpix)
    rows = {}
    for slot in bindless_slots(ed, d):
        desc = ed.descriptors.get(slot, {})
        if desc.get("shape") not in TEXTURE_SHAPES:
            continue
        rows[slot] = desc
    out = []
    for slot, desc in sorted(rows.items()):
        r = ed.resources.get(desc["resource"], {})
        out.append({"slot": slot, "resource_id": desc["resource"],
                    "resource_name": r.get("name"), "res_format": r.get("format"),
                    "width": r.get("width"), "height": r.get("height"), "mips": r.get("mips")})
    return out, n_events


def expected_mips(w, h):
    """Mip count for a full chain down to 1x1 (D3D12 default): floor(log2(max(w,h)))+1.
    For power-of-two square textures (all of these are) that's just max(w,h).bit_length()."""
    if not w or not h:
        return None
    return max(w, h).bit_length()


def shape_key(item):
    return (item["res_format"], item["width"], item["height"])


def shape_report(A, B):
    """Since RTXPT keeps the full source-asset name and Unity only has generic
    'image_N' debug names (see main()'s name-based diff -> 0 shared names), the only
    way to cross-check mips/dims between engines is by grouping on (format, width,
    height) shape and comparing per-shape mip counts + population counts instead of
    per-texture identity."""
    from collections import Counter
    count_a = Counter(shape_key(it) for it in A)
    count_b = Counter(shape_key(it) for it in B)
    mips_a = {shape_key(it): it["mips"] for it in A}
    mips_b = {shape_key(it): it["mips"] for it in B}

    print("\n== Shape-based comparison (format+dims, since debug names don't overlap) ==")
    keys = sorted(set(count_a) | set(count_b), key=lambda k: (k[0] or "", k[1] or 0, k[2] or 0))
    formula_bad, mip_mismatch, count_mismatch = [], [], []
    for k in keys:
        fmt, wd, ht = k
        exp = expected_mips(wd, ht)
        ca, cb = count_a.get(k, 0), count_b.get(k, 0)
        ma, mb = mips_a.get(k), mips_b.get(k)
        if ca and ma != exp:
            formula_bad.append(("RTXPT", k, ma, exp))
        if cb and mb != exp:
            formula_bad.append(("UNITY", k, mb, exp))
        if ca and cb and ma != mb:
            mip_mismatch.append((k, ma, mb))
        if ca != cb:
            count_mismatch.append((k, ca, cb))

    print(f"   unique (format,dims) shapes: RTXPT={len(count_a)}  UNITY={len(count_b)}")

    if formula_bad:
        print(f"\n   {len(formula_bad)} entries violate the full-mip-chain formula "
              "(mips != floor(log2(max(w,h)))+1):")
        for eng, k, actual, exp in formula_bad:
            print(f"      {eng} {k[0]} {k[1]}x{k[2]}: mips={actual} expected={exp}")
    else:
        print("   PASS: every texture in both engines has a full mip chain "
              "(mips == floor(log2(max(w,h)))+1).")

    if mip_mismatch:
        print(f"\n   {len(mip_mismatch)} shared (format,dims) shape(s) have differing mip counts:")
        for k, ma, mb in mip_mismatch:
            print(f"      {k[0]} {k[1]}x{k[2]}: RTXPT mips={ma}  UNITY mips={mb}")
    else:
        print("   PASS: mip counts agree for every shared (format,dims) shape.")

    if count_mismatch:
        print(f"\n   {len(count_mismatch)} shape(s) (i.e. dims) have a different "
              "population between engines:")
        for k, ca, cb in count_mismatch:
            print(f"      {k[0]} {k[1]}x{k[2]}: RTXPT count={ca}  UNITY count={cb}")
    else:
        print("   PASS: dims are distributed identically -- same number of textures "
              "per (format,dims) shape in both engines.")

    return not (formula_bad or mip_mismatch or count_mismatch)


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

    shape_ok = shape_report(A, B)

    print()
    if only_a or only_b or mismatched:
        print("VERDICT (by name): bindless texture sets differ between the two engines (see lists above).")
    else:
        print("VERDICT (by name): identical set of bindless textures (by name/format/dims) in both engines; "
              "heap slot order differs (engine-specific bindless indices).")
    print(f"VERDICT (by shape): mips/dims are {'CONSISTENT' if shape_ok else 'INCONSISTENT'} "
          "between engines (see shape-based comparison above).")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
