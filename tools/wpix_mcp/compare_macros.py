"""Compare compile macros (-D defines) and key flags of every compute Dispatch shared
between the two kitchen captures, plus full detail for hash-mismatched passes.

Run from tools/wpix_mcp:  python compare_macros.py [rtxpt.wpix] [unity.wpix]
"""
import argparse, os, sys
import wpix_core as w

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt-kitchen.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity-kitchen.wpix")


def leaf(marker):
    return marker.rsplit("/", 1)[-1] if marker else marker


def shader_infos(wpix):
    """leaf marker -> describe_shader info (first dispatch only per marker)."""
    out = {}
    for e in w.find_shader_events(wpix, dispatches_only=True):
        lf = leaf(e["marker"])
        if lf in out:
            continue
        try:
            out[lf] = w.describe_shader(wpix, e["global_id"], disassemble=True)
        except Exception as ex:
            out[lf] = {"error": f"{type(ex).__name__}: {ex}"}
    return out


def fmt_defs(defs):
    return sorted(defs or [])


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY):
    print("Reading RTXPT capture (this exports/caches the capture on first run)...")
    a = shader_infos(rtxpt)
    print("Reading Unity capture ...")
    b = shader_infos(unity)

    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print(f"\nShared passes: {len(shared)}   RTXPT-only: {only_a}   Unity-only: {only_b}\n")

    for lf in shared:
        ia, ib = a[lf], b[lf]
        if "error" in ia or "error" in ib:
            print(f"== {lf}: ERROR  rtxpt={ia.get('error','-')}  unity={ib.get('error','-')}")
            continue
        same_hash = ia.get("shader_hash") == ib.get("shader_hash")
        da, db = fmt_defs(ia.get("defines")), fmt_defs(ib.get("defines"))
        same_def = da == db
        ta, tb = ia.get("target"), ib.get("target")
        ea, eb = ia.get("entry"), ib.get("entry")
        status = "OK " if (same_hash and same_def and ta == tb and ea == eb) else "DIFF"
        print(f"== {lf}: {status}  hash {'same' if same_hash else 'DIFFERS'}  "
              f"args R={ia.get('args_source','?')}/U={ib.get('args_source','?')}")
        if ta != tb: print(f"   target: RTXPT={ta}  UNITY={tb}")
        if ea != eb: print(f"   entry:  RTXPT={ea}  UNITY={eb}")
        if not same_def:
            ra = [d for d in da if d not in db]
            ru = [d for d in db if d not in da]
            if ra: print(f"   defines only in RTXPT: {ra}")
            if ru: print(f"   defines only in UNITY: {ru}")
        fa = sorted(set(ia.get("flags") or []))
        fb = sorted(set(ib.get("flags") or []))
        if fa != fb:
            print(f"   flags only in RTXPT: {[f for f in fa if f not in fb]}")
            print(f"   flags only in UNITY: {[f for f in fb if f not in fa]}")
        if not same_hash:
            print(f"   RTXPT hash={ia.get('shader_hash')}  defines={da}")
            print(f"   UNITY hash={ib.get('shader_hash')}  defines={db}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity))
