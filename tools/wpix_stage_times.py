"""Print per-stage GPU durations for one or more .wpix captures via pixtool.

For each capture this runs
    pixtool open-capture X.wpix save-event-list tmp.csv --counters=*EOP* --counters=Execution*
and reads stage durations straight off the marker rows: PIX marker rows (empty
Global ID) carry timings aggregated over their whole subtree, so no summing of
child dispatches is needed. "EOP to EOP Duration" is the serialized GPU cost and
sums to total frame time; "TOP to EOP" measures execution-start to end-of-pipe
and inflates under overlapping work, so it is shown only for reference.

Stage level: children of ExecuteRenderGraph when present (Unity render-graph
captures), else top-level events (RTXPT-style captures).

Durations come from pixtool replaying the capture on the local GPU, not from
capture time -- numbers are comparable across captures analyzed on one machine.

For the dedicated RTXPT-vs-Unity comparison (marker-name matching, multi-run
averaging) use tools/wpix_mcp/compare_stage_times.py instead.

Usage:  python tools/wpix_stage_times.py Other/Player_20s.wpix [more.wpix ...]
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

PIXTOOL = r"C:\Program Files\Microsoft PIX\2603.25\pixtool.exe"

# Raw D3D12 API calls that can appear between markers at stage level; they are
# listed with a trailing * so real (marker) stages stand out.
API_CALLS = {
    'Signal', 'Wait', 'Reset', 'Close', 'ResourceBarrier', 'CopyBufferRegion',
    'EndQuery', 'BeginQuery', 'ResolveQueryData', 'SetDescriptorHeaps',
    'ClearUnorderedAccessViewFloat', 'ClearUnorderedAccessViewUint',
    'ClearRenderTargetView', 'ClearDepthStencilView', 'CopyTextureRegion',
    'CopyResource', 'ExecuteIndirect', 'DiscardResource', 'ResolveSubresource',
    'BuildRaytracingAccelerationStructure', 'DispatchRays', 'Dispatch',
    'DrawInstanced', 'DrawIndexedInstanced',
}


def export_event_list(pixtool, wpix, out_csv):
    cmd = [pixtool, 'open-capture', wpix, 'save-event-list', out_csv,
           '--counters=*EOP*', '--counters=Execution*']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_csv):
        sys.exit(f"pixtool failed on {wpix} (exit {r.returncode}):\n{r.stdout}{r.stderr}")


def load_events(path):
    rows = {}
    with open(path, newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)  # Queue ID, Parent, Name, Global ID, then the 4 timing columns
        for line in reader:
            if not line:
                continue
            rows[int(line[0])] = {
                'id': int(line[0]),
                'parent': int(line[1]),
                'name': line[2],
                'is_marker': line[3] == '' and line[2] not in API_CALLS,
                'eop_dur': int(line[5]) if line[5] else None,
                'top_eop': int(line[7]) if line[7] else None,
            }
    return rows


def stage_root(rows):
    for r in rows.values():
        if r['name'].startswith('ExecuteRenderGraph'):
            return r['id']
    return -1


def report(wpix, rows, api_calls):
    root = stage_root(rows)
    where = 'under ExecuteRenderGraph' if root != -1 else 'top level'
    stages = [r for r in sorted(rows.values(), key=lambda r: r['id'])
              if r['parent'] == root and r['eop_dur'] is not None
              and (api_calls or r['is_marker'] or r['eop_dur'] > 0)]

    print(f"\n== {wpix} ({where}; * = raw API call) ==")
    print(f"{'Stage':<55} {'EOP-EOP (us)':>14} {'TOP-EOP (us)':>14}")
    total = 0
    for s in stages:
        name = s['name'][:53] + ('' if s['is_marker'] else ' *')
        print(f"{name:<55} {s['eop_dur'] / 1000:>14.1f} {(s['top_eop'] or 0) / 1000:>14.1f}")
        total += s['eop_dur']
    print(f"{'TOTAL (sum of EOP-EOP)':<55} {total / 1000:>14.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('captures', nargs='+', help='.wpix capture file(s)')
    p.add_argument('--pixtool', default=PIXTOOL, help='path to pixtool.exe')
    p.add_argument('--api-calls', action='store_true',
                   help='also list zero-duration raw API calls at stage level')
    p.add_argument('--keep-csv', metavar='DIR',
                   help='keep the raw per-event CSVs (full marker tree, per-dispatch '
                        'timings) in DIR for finer-grained slicing')
    args = p.parse_args()

    for wpix in args.captures:
        if not os.path.exists(wpix):
            sys.exit(f"capture not found: {wpix}")
        if args.keep_csv:
            os.makedirs(args.keep_csv, exist_ok=True)
            out_csv = os.path.join(args.keep_csv, os.path.splitext(os.path.basename(wpix))[0] + '_events.csv')
        else:
            # never write next to the capture: pixtool holds the dir locked while open
            out_csv = os.path.join(tempfile.gettempdir(), 'wpix_stage_times_events.csv')
        export_event_list(args.pixtool, wpix, out_csv)
        report(wpix, load_events(out_csv), args.api_calls)


if __name__ == '__main__':
    main()
