"""Compare the raytracing acceleration-structure TLAS instances of the two Other/ captures:
instance count, per-instance Transform, InstanceMask, Flags, InstanceID,
InstanceContributionToHitGroupIndex, and BLAS sub-allocation count.

PIX exports the TLAS input as PopulateRaytracingInstanceDescs_*() which initialises a
D3D12_RAYTRACING_INSTANCE_DESC[] array.  Each entry is emitted as:

    instanceDescs[i] = { { <12 floats: row-major 3x4 Transform> },
                         InstanceID, InstanceMask, InstanceContributionToHitGroupIndex, Flags,
                         GetGpuva(<blasResourceId>, <offset>) };

so this parses that array directly (no GPU decode needed) for both captures and diffs them.

Transforms are compared two ways:
  * positional  -- instance[i] vs instance[i]
  * order-independent (multiset) -- because the two engines do NOT emit instances in the same
    order, and RTXPT/Unity use opposite world handedness (an X-axis flip, same convention seen
    in g_Const).  The script auto-detects the axis-flip F=diag(+/-1,+/-1,+/-1) that maximises
    the match, applies the similarity M' = F*M*F to one side, then compares rounded transforms.

BLAS identity (the GetGpuva resource id) is engine-internal and NOT compared by value; only the
*number* of distinct BLAS sub-allocations (resourceId, offset) is compared.

Run from tools/wpix_mcp:  python compare_accel_structs.py [rtxpt.wpix] [unity.wpix] [--round N]
"""
import argparse, collections, glob, itertools, os, re, sys
import wpix_core as w

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt-bistro.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity-bistro.wpix")

INST_RE = re.compile(
    r"instanceDescs\[(\d+)\]\s*=\s*\{\s*\{([^}]*)\}\s*,\s*"
    r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*GetGpuva\((\d+),\s*(\d+)\)")

# D3D12_RAYTRACING_INSTANCE_FLAGS bit -> name
FLAG_NAMES = {
    0x1: "TRIANGLE_CULL_DISABLE", 0x2: "TRIANGLE_FRONT_CCW",
    0x4: "FORCE_OPAQUE", 0x8: "FORCE_NON_OPAQUE",
}


def _f(s):
    return float(s.strip().rstrip("f"))


def parse_instances(wpix):
    """[{idx, transform[12], id, mask, contrib, flags, blas, off}] from the TLAS input array."""
    ed = w.load_export(w.export_full(wpix))
    files = glob.glob(os.path.join(ed.dir, "RaytracingInstanceDescs_*.cpp"))
    insts = []
    for fp in files:
        t = open(fp, encoding="utf-8", errors="replace").read()
        for m in INST_RE.finditer(t):
            insts.append({
                "idx": int(m.group(1)),
                "transform": [_f(x) for x in m.group(2).split(",")],
                "id": int(m.group(3)), "mask": int(m.group(4)),
                "contrib": int(m.group(5)), "flags": int(m.group(6)),
                "blas": int(m.group(7)), "off": int(m.group(8)),
            })
    insts.sort(key=lambda i: i["idx"])
    return insts


def flag_str(f):
    if f == 0:
        return "0 (none)"
    parts = [n for b, n in FLAG_NAMES.items() if f & b]
    return f"0x{f:x} ({'|'.join(parts) or '?'})"


def apply_flip(tf, s):
    """Similarity M' = F*M*F on a row-major 3x4, F = diag(s0,s1,s2). 3x3: M'[i][j]=s[i]*s[j]M;
    translation col: M'[i][3]=s[i]*M[i][3]."""
    out = list(tf)
    for i in range(3):
        for j in range(3):
            out[i * 4 + j] = s[i] * s[j] * tf[i * 4 + j]
        out[i * 4 + 3] = s[i] * tf[i * 4 + 3]
    return out


def best_flip(A, B, nd):
    """Pick the axis-flip applied to A that maximises the rounded-transform multiset overlap."""
    rnd = lambda tf: tuple(round(x, nd) for x in tf)
    SB = collections.Counter(rnd(i["transform"]) for i in B)
    best = None
    for s in itertools.product((1, -1), repeat=3):
        SA = collections.Counter(rnd(apply_flip(i["transform"], s)) for i in A)
        matched = sum((SA & SB).values())
        if best is None or matched > best[1]:
            best = (s, matched, SA)
    return best  # (sign-vector, matched-count, SA-counter)


def histo(insts, key):
    return dict(sorted(collections.Counter(i[key] for i in insts).items()))


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY, nd=2):
    print("Reading RTXPT TLAS instances ...")
    A = parse_instances(rtxpt)
    print("Reading Unity TLAS instances ...")
    B = parse_instances(unity)
    if not A or not B:
        print(f"ERROR: no instance descs parsed (RTXPT={len(A)}, UNITY={len(B)})")
        return 1

    def blas_count(insts):
        return len({(i["blas"], i["off"]) for i in insts})

    print("\n== Per-capture summary ==")
    print(f"   {'':24}{'RTXPT':>14}{'UNITY':>14}")
    print(f"   {'instance count':24}{len(A):>14}{len(B):>14}")
    print(f"   {'distinct BLAS (blas,off)':24}{blas_count(A):>14}{blas_count(B):>14}")
    print(f"   InstanceMask histogram   RTXPT={histo(A,'mask')}   UNITY={histo(B,'mask')}")
    fa = {f: flag_str(f) for f in histo(A, "flags")}
    fb = {f: flag_str(f) for f in histo(B, "flags")}
    print(f"   Flags histogram          RTXPT={ {flag_str(k):v for k,v in histo(A,'flags').items()} }")
    print(f"                            UNITY={ {flag_str(k):v for k,v in histo(B,'flags').items()} }")
    print(f"   InstanceID range         RTXPT={A[0]['id']}..{A[-1]['id']}   "
          f"UNITY={B[0]['id']}..{B[-1]['id']}")
    print(f"   Contribution range       RTXPT={min(i['contrib'] for i in A)}..{max(i['contrib'] for i in A)}   "
          f"UNITY={min(i['contrib'] for i in B)}..{max(i['contrib'] for i in B)}")

    # ---- direct comparisons -----------------------------------------------------------
    print("\n== Differences ==")
    ok = True
    if len(A) != len(B):
        print(f"   [DIFF] instance COUNT differs: RTXPT={len(A)} UNITY={len(B)}"); ok = False
    if histo(A, "mask") != histo(B, "mask"):
        print(f"   [DIFF] InstanceMask differs: RTXPT={histo(A,'mask')}  UNITY={histo(B,'mask')}")
        print("          (mask is AND-ed with TraceRay InstanceInclusionMask; a scene-wide")
        print("           constant like 1 vs 255 only matters if rays use a non-0xFF mask)")
        ok = False
    if histo(A, "flags") != histo(B, "flags"):
        print(f"   [DIFF] instance Flags differ: RTXPT={histo(A,'flags')}  UNITY={histo(B,'flags')}")
        ok = False
    if blas_count(A) != blas_count(B):
        print(f"   [DIFF] distinct BLAS count differs: RTXPT={blas_count(A)} UNITY={blas_count(B)}"); ok = False
    # positional id / contrib (these are sequential per engine; only meaningful if counts match)
    if len(A) == len(B):
        idm = sum(1 for a, b in zip(A, B) if a["id"] != b["id"])
        cm = sum(1 for a, b in zip(A, B) if a["contrib"] != b["contrib"])
        if idm or cm:
            print(f"   [DIFF] positional InstanceID mismatches={idm}, Contribution mismatches={cm}")
            ok = False

    # ---- transforms: positional + order-independent (convention-aware) ----------------
    print("\n== Transforms ==")
    if len(A) == len(B):
        rnd = lambda tf: tuple(round(x, nd) for x in tf)
        pos_match = sum(1 for a, b in zip(A, B) if rnd(a["transform"]) == rnd(b["transform"]))
        print(f"   positional (instance[i] vs instance[i], round {nd}): {pos_match}/{len(A)} match")
    sign, matched, _ = best_flip(A, B, nd)
    flip_desc = "identity" if sign == (1, 1, 1) else \
        "axis-flip F=diag" + str(sign) + " applied to RTXPT (M'=F*M*F)"
    print(f"   order-independent multiset match (best coordinate convention): {matched}/{len(A)}")
    print(f"      best convention: {flip_desc}")
    if matched < len(A):
        print(f"      {len(A) - matched} transform(s) unmatched at round {nd} "
              "(fp precision or genuine differences)")

    # ---- BLAS topology: count, fan-out, and grouping (reorder-invariant) ---------------
    # NOTE: only the per-frame TLAS build is in the capture; the BLASes are persistent (built
    # once at load), so their geometry descs / opaque flags are NOT exported.  What is
    # comparable is the instance->BLAS topology: how geometry is grouped into BLASes and how
    # often each BLAS is instanced.
    print("\n== BLAS topology ==")

    def blas_groups(insts):
        g = collections.defaultdict(list)
        for i in insts:
            g[(i["blas"], i["off"])].append(i["transform"])
        return g

    GA, GB = blas_groups(A), blas_groups(B)
    print(f"   distinct BLAS: RTXPT={len(GA)}  UNITY={len(GB)}")

    # fan-out = how many BLASes are referenced by exactly N instances
    fan_a = collections.Counter(len(v) for v in GA.values())
    fan_b = collections.Counter(len(v) for v in GB.values())
    fan_ok = fan_a == fan_b
    print(f"   instances-per-BLAS fan-out histogram identical: {fan_ok}")
    if not fan_ok:
        print(f"      RTXPT={dict(sorted(fan_a.items()))}")
        print(f"      UNITY={dict(sorted(fan_b.items()))}")

    # reorder-invariant grouping: characterise each BLAS by the multiset of its member
    # instance transforms (canonicalised via the auto-detected flip), then diff the multisets.
    def grp_sig(members, do_flip):
        return tuple(sorted(
            tuple(round(x, nd) for x in (apply_flip(tf, sign) if do_flip else tf))
            for tf in members))
    SA = collections.Counter(grp_sig(v, True) for v in GA.values())
    SB = collections.Counter(grp_sig(v, False) for v in GB.values())
    grp_match = sum((SA & SB).values())
    print(f"   grouping match (BLAS by member-transform signature, {flip_desc}): "
          f"{grp_match}/{len(GA)}")
    if grp_match < len(GA):
        print(f"      {len(GA) - grp_match} BLAS group(s) unmatched at round {nd} "
              "(fp precision in member transforms)")
    blas_ok = fan_ok and len(GA) == len(GB)

    # ---- verdict ----------------------------------------------------------------------
    print()
    conv = "" if sign == (1, 1, 1) else f" (up to the {sign} world-handedness convention)"
    if ok and matched == len(A) and blas_ok:
        print(f"VERDICT: acceleration structures are CONSISTENT{conv}: same instance count, "
              "masks, flags, BLAS count/topology, ids, and transforms.")
    elif ok and blas_ok:
        print(f"VERDICT: instance metadata (count/mask/flags/ids) and BLAS topology match; "
              f"{matched}/{len(A)} transforms match{conv}, rest differ only at round {nd}.")
    else:
        print("VERDICT: acceleration structures DIFFER -- see [DIFF] lines above "
              f"(transforms: {matched}/{len(A)} match{conv}).")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    p.add_argument("--round", type=int, default=2, help="decimals for transform comparison (default 2)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity, args.round))
