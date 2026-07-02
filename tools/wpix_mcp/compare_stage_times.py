"""Compare GPU time spent per render-pass stage between the two Other/ captures.

Uses wpix_core.stage_times(group_by="leaf") to sum each capture's PIX GPU duration
counter (default "TOP to EOP Duration (ns)") per top-level marker/stage -- the same
data pixtool save-event-list --counters exposes, aggregated the way PIX's own GPU
timing view groups it.

The two engines name markers differently: RTXPT emits legacy single-segment markers
(via the deprecated pix3.h API) prefixed with a fixed PIX compatibility notice, e.g.
"<deprecated - use pix3.h instead> PathTrace"; Unity nests markers under its render
graph path, e.g. ".../ExecuteRenderGraph/PathTrace/PathTrace", whose leaf segment is
the clean "PathTrace". This strips RTXPT's fixed prefix before matching so the same
logical stage lines up across both captures.

Run from tools/wpix_mcp:  python compare_stage_times.py [rtxpt.wpix] [unity.wpix]

`pixtool save-event-list --counters=...` replays the capture on the actual GPU to
produce its duration counters -- these are live hardware measurements, not values
frozen into the .wpix at capture time, so replaying the *same* capture repeatedly
gives independent samples (driver/thermal jitter etc). Pass --runs N to replay each
side's capture N times (bypassing the timing cache each time) and average the
per-stage duration_ms across those replays, instead of a single cached reading:

    python compare_stage_times.py --runs 5
"""
import argparse, os, re, sys, time
import wpix_core as w
from wpix_core import _render_table

# unique per process invocation, so repeated --runs CLI calls don't collide with a
# prior run's cache_bust keys and silently serve stale cached CSVs
_INVOCATION_ID = f"{os.getpid()}-{time.time_ns()}"

DEF_RTXPT = os.path.join("..", "..", "Other", "Rtxpt.wpix")
DEF_UNITY = os.path.join("..", "..", "Other", "Unity.wpix")

# RTXPT's legacy PIXBeginEvent markers all carry this fixed compatibility notice.
_DEPRECATED_PREFIX_RE = re.compile(r"^<deprecated[^>]*>\s*")


def norm_stage(name):
    return _DEPRECATED_PREFIX_RE.sub("", name).strip()


def read_stages(wpix, cache_bust=None):
    r = w.stage_times(wpix, group_by="leaf", work_only=True, cache_bust=cache_bust)
    by_name = {}
    for s in r["stages"]:
        by_name[norm_stage(s["stage"])] = s
    return r, by_name


def average_runs(wpix, n, label):
    """Replay `wpix` on the GPU `n` times (each with a distinct cache_bust, so PIX
    actually re-runs save-event-list instead of returning a cached CSV) and average
    per-stage duration_ms / total_duration_ms across those independent samples.
    A stage's run_count records how many of the n replays actually contained it,
    since stages can vary run to run."""
    sum_ms, sum_events, run_count = {}, {}, {}
    total_ms_sum = 0.0
    for i in range(n):
        print(f"   {label} replay {i + 1}/{n} ...")
        r, by_name = read_stages(wpix, cache_bust=f"{_INVOCATION_ID}-{i}")
        total_ms_sum += r["total_duration_ms"]
        for name, s in by_name.items():
            sum_ms[name] = sum_ms.get(name, 0.0) + s["duration_ms"]
            sum_events[name] = sum_events.get(name, 0) + s["event_count"]
            run_count[name] = run_count.get(name, 0) + 1

    by_name = {
        name: {
            "stage": name,
            "duration_ms": sum_ms[name] / run_count[name],
            "event_count": round(sum_events[name] / run_count[name]),
            "run_count": run_count[name],
        }
        for name in sum_ms
    }
    return {"total_duration_ms": total_ms_sum / n, "runs": n}, by_name


def _row(s, total_ms):
    pct = (100.0 * s["duration_ms"] / total_ms) if total_ms else 0.0
    row = [s["stage"], f'{s["duration_ms"]:.3f}', s["event_count"], f'{pct:.1f}%']
    if "run_count" in s:
        row.append(f'{s["run_count"]}')
    return row


def main(rtxpt=DEF_RTXPT, unity=DEF_UNITY, runs=1):
    averaged = runs > 1

    if averaged:
        print(f"Replaying RTXPT capture {runs}x and averaging ...")
        ra, A = average_runs(rtxpt, runs, "RTXPT")
        print(f"Replaying Unity capture {runs}x and averaging ...\n")
        rb, B = average_runs(unity, runs, "UNITY")
    else:
        print("Reading RTXPT stage timings ...")
        ra, A = read_stages(rtxpt)
        print("Reading Unity stage timings ...\n")
        rb, B = read_stages(unity)

    headers = ["stage", "ms", "events", "% of total"] + (["runs"] if averaged else [])
    print(f"== RTXPT stages ({len(A)}, total {ra['total_duration_ms']:.3f} ms) ==")
    rows = sorted(A.values(), key=lambda s: s["duration_ms"], reverse=True)
    print(_render_table(headers, [_row(s, ra["total_duration_ms"]) for s in rows]))

    print(f"\n== UNITY stages ({len(B)}, total {rb['total_duration_ms']:.3f} ms) ==")
    rows = sorted(B.values(), key=lambda s: s["duration_ms"], reverse=True)
    print(_render_table(headers, [_row(s, rb["total_duration_ms"]) for s in rows]))

    names_a, names_b = set(A), set(B)
    shared = sorted(names_a & names_b, key=lambda n: B[n]["duration_ms"] - A[n]["duration_ms"], reverse=True)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    print(f"\n== Per-stage comparison (matched by normalized leaf marker) ==")
    print(f"   shared stages: {len(shared)}   RTXPT-only: {len(only_a)}   UNITY-only: {len(only_b)}\n")
    cmp_rows = []
    for name in shared:
        a, b = A[name]["duration_ms"], B[name]["duration_ms"]
        ratio = (b / a) if a else float("inf")
        cmp_rows.append([name, f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}", f"{ratio:.2f}x"])
    print(_render_table(["stage", "RTXPT ms", "UNITY ms", "delta (U-R)", "U/R"], cmp_rows))

    if only_a:
        print(f"\n   -- only in RTXPT ({len(only_a)}) --")
        for name in only_a:
            print(f"      {name}: {A[name]['duration_ms']:.3f} ms")
    if only_b:
        print(f"\n   -- only in UNITY ({len(only_b)}) --")
        for name in only_b:
            print(f"      {name}: {B[name]['duration_ms']:.3f} ms")

    total_a, total_b = ra["total_duration_ms"], rb["total_duration_ms"]
    print(f"\n== Totals ==")
    print(f"   RTXPT: {total_a:.3f} ms   UNITY: {total_b:.3f} ms   "
          f"UNITY/RTXPT = {(total_b / total_a) if total_a else float('inf'):.2f}x")

    # biggest single contributors to the total gap, shared stages only
    gap_rows = sorted(shared, key=lambda n: B[n]["duration_ms"] - A[n]["duration_ms"], reverse=True)[:5]
    if gap_rows:
        print("\n   Largest contributors to the UNITY-RTXPT gap (shared stages):")
        for name in gap_rows:
            d = B[name]["duration_ms"] - A[name]["duration_ms"]
            print(f"      {name}: {d:+.3f} ms")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rtxpt", nargs="?", default=DEF_RTXPT, help="RTXPT capture (.wpix)")
    p.add_argument("unity", nargs="?", default=DEF_UNITY, help="Unity capture (.wpix)")
    p.add_argument("--runs", type=int, default=1, metavar="N",
                    help="replay each capture N times on the GPU (bypassing the timing "
                         "cache) and average the per-stage duration_ms across the replays")
    args = p.parse_args()
    sys.exit(main(args.rtxpt, args.unity, args.runs))
