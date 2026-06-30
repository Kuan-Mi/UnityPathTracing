"""Compare the DXIL shader hashes of every compute Dispatch executed in the two
captures under Other/ (Rtxpt.wpix = original RTXPT, Unity.wpix = RtxptFeature
replica). Prints a per-marker table and a verdict on whether each capture ran the
exact same shader binaries.

Run from tools/wpix_mcp:  python compare_compute_shaders.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, os, sys
import wpix_core as w

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")


def shader_table(wpix):
    """Ordered list of {marker, global_id, hash, target} for each compute Dispatch."""
    rows = []
    for e in w.find_shader_events(wpix, dispatches_only=True):
        gid = e["global_id"]
        try:
            info = w.describe_shader(wpix, gid, disassemble=False)
            h = info.get("shader_hash")
            tgt = info.get("target")
        except Exception as ex:
            h, tgt = f"<error: {type(ex).__name__}>", None
        rows.append({"marker": e["marker"], "global_id": gid, "hash": h, "target": tgt})
    return rows


def leaf(marker):
    return marker.rsplit("/", 1)[-1] if marker else marker


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    print("Reading RTXPT capture ...")
    a = shader_table(rtxpt)
    print("Reading Unity capture ...")
    b = shader_table(unity)

    # Group hashes by the dispatch's leaf marker name (the shader pass).
    from collections import defaultdict
    A, B = defaultdict(list), defaultdict(list)
    for r in a:
        A[leaf(r["marker"])].append(r["hash"])
    for r in b:
        B[leaf(r["marker"])].append(r["hash"])

    markers = sorted(set(A) | set(B))
    wmark = max((len(m) for m in markers), default=10)

    print()
    print(f"{'pass (leaf marker)':<{wmark}}  {'n':>3}  match  RTXPT hash / Unity hash")
    print("-" * (wmark + 60))
    all_match = True
    only_one_side = []
    for m in markers:
        ha, hb = A.get(m, []), B.get(m, [])
        if not ha or not hb:
            only_one_side.append(m)
            all_match = False
            side = "RTXPT-only" if ha else "Unity-only"
            hs = (ha or hb)
            print(f"{m:<{wmark}}  {len(hs):>3}  {side:<9}  {set(hs)}")
            continue
        sa, sb = set(ha), set(hb)
        same = sa == sb
        all_match &= same
        mark = "  OK " if same else " DIFF"
        if same:
            print(f"{m:<{wmark}}  {len(ha):>3}  {mark}  {next(iter(sa))}")
        else:
            print(f"{m:<{wmark}}  {len(ha):>3}  {mark}  RTXPT={sa}")
            print(f"{'':<{wmark}}       {'':>5}  UNITY={sb}")

    print()
    # Overall: are the *sets* of shader hashes executed identical?
    set_a = {r["hash"] for r in a}
    set_b = {r["hash"] for r in b}
    print(f"Distinct shader hashes — RTXPT: {len(set_a)}, Unity: {len(set_b)}")
    print(f"Shared:        {len(set_a & set_b)}")
    print(f"RTXPT-only:    {sorted(set_a - set_b)}")
    print(f"Unity-only:    {sorted(set_b - set_a)}")
    print()
    if all_match and not only_one_side:
        print("VERDICT: every pass ran identical shader binaries in both captures.")
    else:
        print("VERDICT: shader hashes differ — see DIFF / *-only rows above.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
