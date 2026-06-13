"""Compare the PTMaterialData material list (t_PTMaterialData, register t5) of the
raytracing passes in the two Other/ captures.

The material array is the StructuredBuffer<PTMaterialData> bound at SRV register t5 of the
DXR global root signature (RTXPT names it "PTMaterialDataStorage"; Unity allocates an
unnamed buffer). describe_dispatch only tracks ->Dispatch, not ->DispatchRays, so this finds
the buffer by walking the exported command list: for each DispatchRays it resolves the active
SetComputeRootDescriptorTable whose root-signature range covers register t5(space0), maps that
to a descriptor-heap slot, and reads the bound resource id. The bytes come from the whole-
capture export (stable ids + captured contents).

PTMaterialData layout is pulled live from the RTXPT shader header and overlaid on each 128-byte
record. Two diffs are produced:
  1. positional  -- material[i] vs material[i] (per-field diff counts + fully-identical count)
  2. order-independent -- match materials across captures by a physical signature, since the
     two engines do NOT store materials in the same index order (the geometry->material index
     mapping is engine-internal). Reports unmatched-on-one-side materials + a transmission summary.

Texture index fields are engine-specific bindless handles (RTXPT: descriptor-heap offsets;
Unity: small sequential indices) and are reported separately, never used in the signature.

Run from tools/wpix_mcp:  python compare_materials.py [rtxpt.wpix] [unity.wpix] [--shaders DIR]
"""
import argparse, collections, glob, os, re, sys
import wpix_core as w

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt-bistro.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity-bistro.wpix")
DEF_SHADERS = os.path.join("..", "..", "RenderingPlugin", "External", "RTXPT", "Rtxpt", "Shaders")

MAT_HEADER = os.path.join("PathTracer", "Materials", "MaterialPT.h")

# engine-specific bindless handles + explicit padding -- never compared by value
TEX_FIELDS = {
    "BaseOrDiffuseTextureIndex", "MetalRoughOrSpecularTextureIndex", "EmissiveTextureIndex",
    "NormalTextureIndex", "OcclusionTextureIndex", "TransmissionTextureIndex",
    "_padding0", "_padding1",
}
# Flags bit (PTMaterialFlags_*) for the texture-use bits and ThinSurface
FLAG_THIN_SURFACE = 0x200
FLAG_TEXTURE_USE_MASK = 0x1 | 0x4 | 0x8 | 0x10 | 0x20 | 0x80  # specGloss + the *UseTexture bits


def material_layout(shaders):
    path = os.path.join(shaders, MAT_HEADER)
    return w.parse_struct_layout(open(path, encoding="utf-8", errors="replace").read(),
                                 "PTMaterialData")


def material_buffer_id(ed, wpix):
    """Resolve the resource id bound at SRV t5 (space0) for the PathTrace DispatchRays.
    Falls back to the resource named 'PTMaterialDataStorage'."""
    text = ""
    for f in glob.glob(os.path.join(ed.dir, "CommandLists_*.cpp")):
        text += open(f, encoding="utf-8", errors="replace").read()
    cur_sig = None
    tables = {}     # root param -> base descriptor-heap slot
    found = None
    for ln in text.split("\n"):
        m = re.search(r"SetComputeRootSignature\(GetRootSignature\((\d+)\)", ln)
        if m:
            cur_sig = int(m.group(1))
        m = re.search(r"SetComputeRootDescriptorTable\((\d+),\s*GetGpuDescriptor\([^,]+,\s*(\d+)\)", ln)
        if m:
            tables[int(m.group(1))] = int(m.group(2))
        if "->DispatchRays(" in ln:
            sig = ed.root_sigs.get(cur_sig, {})
            for param, base in tables.items():
                for rg in sig.get(param, {}).get("ranges") or []:
                    if rg["range_type"] == "SRV" and rg["space"] == 0 and \
                            rg["base"] <= 5 < rg["base"] + rg["num"]:
                        slot = base + rg["offset"] + (5 - rg["base"])
                        desc = ed.descriptors.get(slot, {})
                        if desc.get("resource") is not None:
                            found = desc["resource"]
    if found is None:
        found = ed.find_resource_loose("PTMaterialDataStorage")
    return found


def read_materials(wpix, layout, size):
    """[ {field: value} ] for each populated material in the capture's t5 buffer."""
    ed = w.load_export(w.export_full(wpix))
    rid = material_buffer_id(ed, wpix)
    if rid is None:
        raise RuntimeError(f"could not resolve t_PTMaterialData (t5) buffer in {wpix}")
    blob = w.decompress_blob(ed.dir, rid, ed)
    n_slots = len(blob) // size
    mats = []
    last_nonzero = -1
    for i in range(n_slots):
        sub = blob[i * size:(i + 1) * size]
        if any(sub):
            last_nonzero = i
        mats.append({f["name"]: w._field_value(sub, f) for f in layout})
    return rid, len(blob), mats[:last_nonzero + 1]


def _round(x, n=4):
    if isinstance(x, list):
        return tuple(round(v, n) if v is not None else None for v in x)
    return round(x, n) if x is not None else None


def phys_signature(m):
    """Engine-neutral physical fingerprint: every non-texture, non-emissive field, with the
    texture-use flag bits masked off (RTXPT drives color via a texture where Unity may bake a
    constant). EmissiveColor is excluded here so that an emissive *intensity* port mismatch
    doesn't hide an otherwise-matching material -- it is checked separately below."""
    return (
        _round(m["BaseOrDiffuseColor"]), _round(m["SpecularColor"]),
        _round(m["ShadowNoLFadeout"]), _round(m["Opacity"]), _round(m["Roughness"]),
        _round(m["Metalness"]), _round(m["NormalTextureScale"]), _round(m["AlphaCutoff"]),
        _round(m["TransmissionFactor"]), _round(m["IoR"]), _round(m["ThicknessFactor"]),
        _round(m["DiffuseTransmissionFactor"]), _round(m["Volume.AttenuationColor"]),
        _round(m["Volume.AttenuationDistance"]), m["Flags"] & ~FLAG_TEXTURE_USE_MASK,
    )


def _emissive_normalized(c):
    """EmissiveColor with absolute intensity removed (hue only): divide by max channel."""
    mx = max(c) if c and max(c) > 0 else 0
    return tuple(round(v / mx, 3) for v in c) if mx > 0 else (0.0, 0.0, 0.0)


def transmission_summary(mats):
    thin = sum(1 for m in mats if m["Flags"] & FLAG_THIN_SURFACE)
    trans = sum(1 for m in mats if m["TransmissionFactor"] > 0)
    dtrans = sum(1 for m in mats if m["DiffuseTransmissionFactor"] > 0)
    return thin, trans, dtrans


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY, shaders=DEF_SHADERS):
    layout, size = material_layout(shaders)
    print(f"sizeof(PTMaterialData) = {size} bytes ({len(layout)} scalar fields)\n")

    print("Reading RTXPT material buffer ...")
    rid_a, bytes_a, A = read_materials(rtxpt, layout, size)
    print("Reading Unity material buffer ...")
    rid_b, bytes_b, B = read_materials(unity, layout, size)
    print(f"\nRTXPT t5 buffer = resource {rid_a} ({bytes_a} bytes), {len(A)} populated materials")
    print(f"UNITY t5 buffer = resource {rid_b} ({bytes_b} bytes), {len(B)} populated materials\n")

    # ---- 1. positional per-field diff -------------------------------------------------
    n = min(len(A), len(B))
    percol = collections.Counter()
    identical = 0
    for i in range(n):
        any_diff = False
        for f in layout:
            k = f["name"]
            if A[i][k] != B[i][k]:
                percol[k] += 1
                any_diff = True
        if not any_diff:
            identical += 1
    print("== Positional diff (material[i] vs material[i]) ==")
    print(f"   fully-identical materials: {identical} / {n}")
    print(f"   {'field':36} diffs")
    print("   " + "-" * 46)
    for f in layout:
        c = percol[f["name"]]
        tag = "  (texture/bindless - expected)" if f["name"] in TEX_FIELDS and c else ""
        print(f"   {f['name']:36} {c:4}{tag}")

    # ---- 2. order-independent match by physical signature (emissive excluded) ---------
    SA = collections.Counter(phys_signature(m) for m in A)
    SB = collections.Counter(phys_signature(m) for m in B)
    only_a, only_b = SA - SB, SB - SA
    matched = sum((SA & SB).values())
    print("\n== Order-independent match (by physical signature; texture indices & emissive ignored) ==")
    print(f"   materials whose physical signature is present in BOTH: {matched} / {len(A)}")
    print(f"   only in RTXPT: {sum(only_a.values())}   only in UNITY: {sum(only_b.values())}")
    if only_a or only_b:
        sig_fields = ["BaseOrDiffuseColor", "SpecularColor", "Roughness", "Metalness",
                      "TransmissionFactor", "DiffuseTransmissionFactor", "IoR", "Flags"]
        def show(side, mats, counter):
            for i, m in enumerate(mats):
                if counter.get(phys_signature(m), 0) > 0:
                    counter[phys_signature(m)] -= 1
                    vals = "  ".join(f"{k}={w._val_str(m[k])}" for k in sig_fields)
                    print(f"   [{side} #{i}] {vals}")
        print("   --- unmatched RTXPT materials (genuinely different physical values) ---")
        show("R", A, dict(only_a))
        print("   --- unmatched UNITY materials ---")
        show("U", B, dict(only_b))

    # ---- 2b. emissive intensity: do the matched materials carry the same emission? -----
    # Match emissive materials across captures by their *hue* (intensity-normalized color),
    # then compare absolute intensity (max channel).  A systematic Unity<RTXPT ratio means the
    # emissive intensity multiplier was dropped in the port.
    def emissives(mats):
        out = collections.defaultdict(list)
        for i, m in enumerate(mats):
            c = m["EmissiveColor"]
            if c and max(c) > 0:
                out[_emissive_normalized(c)].append((i, max(c)))
        return out
    EA, EB = emissives(A), emissives(B)
    na = sum(len(v) for v in EA.values())
    nb = sum(len(v) for v in EB.values())
    print("\n== Emissive materials (EmissiveColor != 0) ==")
    print(f"   RTXPT: {na}   UNITY: {nb}")
    same_hue = sorted(set(EA) & set(EB))
    intensity_rows = []
    for hue in same_hue:
        # pair brightest-to-brightest within the same hue bucket
        ia = sorted((mx for _, mx in EA[hue]), reverse=True)
        ib = sorted((mx for _, mx in EB[hue]), reverse=True)
        for ra, ub in zip(ia, ib):
            if abs(ra - ub) > 1e-3:
                intensity_rows.append((hue, ra, ub, (ra / ub if ub else float("inf"))))
    if intensity_rows:
        print(f"   {len(intensity_rows)} emissive material(s) share hue but differ in INTENSITY:")
        print(f"   {'hue (normalized)':28} {'RTXPT max':>10} {'UNITY max':>10} {'R/U':>8}")
        for hue, ra, ub, ratio in intensity_rows:
            print(f"   {str(hue):28} {ra:10.4g} {ub:10.4g} {ratio:8.1f}")
        print("   >>> Unity emissive intensity is stripped (hue preserved): the emissive")
        print("       intensity multiplier is not being applied in the material port.")
    elif na or nb:
        print("   emissive intensities match.")

    # ---- 3. transmission summary (the glass / light-blocking signal) ------------------
    ta, tb = transmission_summary(A), transmission_summary(B)
    print("\n== Transmission summary (ThinSurface | TransmissionFactor>0 | DiffuseTransmission>0) ==")
    print(f"   RTXPT: {ta[0]} | {ta[1]} | {ta[2]}")
    print(f"   UNITY: {tb[0]} | {tb[1]} | {tb[2]}")
    # multiset of transmission-only signatures (catches glass that moved index but kept value)
    tsig = lambda m: (_round(m["TransmissionFactor"]), _round(m["DiffuseTransmissionFactor"]),
                      _round(m["IoR"]), _round(m["ThicknessFactor"]),
                      bool(m["Flags"] & FLAG_THIN_SURFACE), _round(m["Opacity"]),
                      _round(m["Volume.AttenuationColor"]), _round(m["Volume.AttenuationDistance"]))
    TA = collections.Counter(tsig(m) for m in A)
    TB = collections.Counter(tsig(m) for m in B)
    print(f"   transmission-signature multiset identical: {TA == TB}")
    if TA != TB:
        print(f"      only in RTXPT: {dict(TA - TB)}")
        print(f"      only in UNITY: {dict(TB - TA)}")

    print()
    order_note = ("the material set matches but is stored in a DIFFERENT index order "
                  "(material ordering is not fixed across the two engines)")
    if matched == len(A) == len(B) and not intensity_rows:
        print(f"VERDICT: identical material set; {order_note}.")
    elif matched == len(A) == len(B):
        print(f"VERDICT: all {matched} materials match on surface/transmission params; "
              f"{order_note}. BUT {len(intensity_rows)} emissive material(s) lost their "
              "intensity in the Unity port (see emissive table above).")
    else:
        print(f"VERDICT: material sets mostly overlap ({matched} matched); "
              f"{sum(only_a.values())} RTXPT-only / {sum(only_b.values())} Unity-only differ "
              f"in physical values; {order_note}.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    p.add_argument("--shaders", default=DEF_SHADERS, help="RTXPT Shaders dir for PTMaterialData layout")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity, args.shaders))
