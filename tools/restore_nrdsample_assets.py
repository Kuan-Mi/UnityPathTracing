"""
restore_nrdsample_assets.py
---------------------------
Restore the git-ignored payload files under UnityProject/Assets/NRD-Sample/Assets
(*.png / *.jpg / *.glb / *.bin / *.dds / *.exr, see UnityProject/.gitignore) from
the NRD-Sample _Data checkout, so the project can be opened on another computer.

The repo keeps every Unity .meta sidecar; this script walks them, finds metas
whose payload file is missing, and restores the payload (same approach as
restore_rtxpt_assets.py). The Unity tree flattens the NRD-Sample _Data layout
(e.g. Assets/Bistro <- _Data/Scenes/Bistro, Assets/Textures <- _Data/Textures),
so each payload is looked up under a small list of source bases (SRC_SEARCH_BASES):

  1. same relative path exists under a source base       -> copy as-is
  2. missing .png but <stem>.dds exists under a base      -> decode DDS, save PNG
     (uses load_dds/to_png_image from dds_to_png.py, the same conversion the
      RTXPTAssets originals were produced with; it handles the BC4/BC5 1x1
      textures download_bistro.bat patches with fix_dds_small_bc.ps1, so no
      separate fix step is needed)
  3. a known generated payload (Bistro Image*.png)        -> written verbatim
     (the materials' KHR_materials_specular specularColorTexture; 1x1 white
      "specular tint = white" maps the exporter emitted, not in the Packman data)
  4. otherwise                                            -> reported as MISSING
     (Cube/cube.bin is NOT in the Packman data and must live in git itself;
      see --help epilog)

The Bistro/ShaderBalls data is the same payload download_bistro.bat fetches via
Packman; when --src is omitted this script resolves that source automatically
(auto-running Packman when needed).

Dependencies: Pillow (pip install Pillow) -- only needed when DDS conversion
actually has to run.

Usage (from the repo root):
    python restore_nrdsample_assets.py                     # auto: Packman pull
    python restore_nrdsample_assets.py --src D:\\_Data      # use an existing _Data
    python restore_nrdsample_assets.py --dry-run           # show the plan only
    python restore_nrdsample_assets.py --overwrite         # refresh existing files

When --src is omitted, the source is resolved automatically:
    1. RenderingPlugin/External/NRD-Sample/_Data if it already exists (a previous
       Packman pull), else
    2. Packman is run against NRD-Sample/Dependencies.xml to download the
       nri_data package (same call download_bistro.bat makes). The NRD-Sample
       submodule -- including its bundled Packman -- must be present.
"""

import argparse
import base64
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # this script lives in <repo>/tools

# NRD-Sample layout (mirrors the paths download_bistro.bat builds from %~dp0).
NRD_SAMPLE_DIR = os.path.join(REPO_ROOT, "RenderingPlugin", "External", "NRD-Sample")
PACKMAN_CMD = os.path.join(NRD_SAMPLE_DIR, "External", "Packman", "packman.cmd")
DEPS_XML = os.path.join(NRD_SAMPLE_DIR, "Dependencies.xml")
DATA_DIR = os.path.join(NRD_SAMPLE_DIR, "_Data")  # the Packman download root, used as --src
NRI_DATA_VERSION = "2.3"

# The Unity Assets tree flattens the _Data layout: scene folders (Bistro,
# ShaderBalls, ...) live under _Data/Scenes, while the shared Textures/ folder
# lives at the _Data root. Each payload's relative path is resolved against
# these source bases, in order, until a match is found.
SRC_SEARCH_BASES = ("Scenes", "")

# The Bistro materials reference Image.png / Image-1.png as their
# KHR_materials_specular specularColorTexture, but those are not part of the
# Packman data -- they are 1x1 white "specular tint = white" maps the exporter
# emitted. Synthesize them from this embedded constant instead of relying on git.
# (Both files are the identical 119-byte 1x1 white PNG.)
WHITE_1X1_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAAXNSR0IArs4c6QAAAARnQU1B"
    "AACxjwv8YQUAAAAJcEhZcwAADsAAAA7AAWrWiQkAAAAMSURBVBhXY/j//z8ABf4C/qc1gYQA"
    "AAAASUVORK5CYII=")
GENERATED = {
    "Bistro/Image.png": WHITE_1X1_PNG,
    "Bistro/Image-1.png": WHITE_1X1_PNG,
}

# Hand-authored payloads that are NOT derivable from the NRD-Sample data and are
# expected to be committed to git directly (git add -f). Listed so the summary
# can tell "expected in git" apart from a genuinely broken setup. Paths are
# relative to --dest (Assets/NRD-Sample/Assets).
HAND_AUTHORED = {
    "Bistro/Cube/cube.bin",
}


def find_jobs(dest, src, overwrite):
    """Yield (action, rel, src_path, dst_path); action in copy/convert/missing."""
    for dp, _, fs in os.walk(dest):
        for fn in fs:
            if not fn.endswith(".meta"):
                continue
            dst_path = os.path.join(dp, fn[:-len(".meta")])
            if os.path.isdir(dst_path):           # folder meta
                continue
            if os.path.exists(dst_path) and not overwrite:
                continue
            rel = os.path.relpath(dst_path, dest).replace("\\", "/")
            if rel in GENERATED:
                yield "generate", rel, None, dst_path
                continue
            action, payload = resolve_payload(src, rel)
            yield action, rel, payload, dst_path


def resolve_payload(src, rel):
    """Locate rel under one of SRC_SEARCH_BASES; return (action, src_path)."""
    rel_os = rel.replace("/", os.sep)
    for base in SRC_SEARCH_BASES:
        cand = os.path.join(src, base, rel_os) if base else os.path.join(src, rel_os)
        if os.path.isfile(cand):
            return "copy", cand
        stem, ext = os.path.splitext(cand)
        if ext.lower() == ".png" and os.path.isfile(stem + ".dds"):
            return "convert", stem + ".dds"
    return "missing", None


def ensure_packman_pull(dry_run, version):
    """Download the NRD-Sample data via Packman (reuse the _Data root if present)."""
    sentinel = os.path.join(DATA_DIR, "Scenes", "Bistro")
    if os.path.isdir(sentinel):
        print(f"Using existing NRD-Sample data: {DATA_DIR}")
        return DATA_DIR
    if dry_run:
        print(f"(dry run) would run Packman to download NRD-Sample data -> {DATA_DIR}")
        return DATA_DIR
    if not os.path.isfile(PACKMAN_CMD):
        print(f"error: Packman not found: {PACKMAN_CMD}")
        print("       init the NRD-Sample submodule under RenderingPlugin/External first.")
        sys.exit(1)
    print("Downloading NRD-Sample scene data via Packman")
    print(f"     -> {DATA_DIR}  (large; nri_data_version={version})")
    cmd = ["cmd", "/c", PACKMAN_CMD, "pull", DEPS_XML,
           "-p", "windows-x86_64", "-t", f"nri_data_version={version}"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("error: 'cmd' not found; this script's auto-download is Windows-only.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"error: Packman pull failed (exit {e.returncode}).")
        sys.exit(1)
    if not os.path.isdir(sentinel):
        print(f"error: NRD-Sample data not found after Packman pull: {sentinel}")
        sys.exit(1)
    return DATA_DIR


def resolve_src(args):
    """Return the _Data source root, running Packman if --src was omitted."""
    if args.src is not None:
        return args.src
    return ensure_packman_pull(args.dry_run, args.nri_data_version)


def main():
    ap = argparse.ArgumentParser(
        description="Restore git-ignored Assets/NRD-Sample/Assets payloads from the NRD-Sample data.",
        epilog="MISSING files marked '(expected in git)' are hand-authored and "
               "must be committed with: git add -f <file>  on the source machine.")
    ap.add_argument("--src", default=None,
                    help="NRD-Sample _Data root (the folder holding Scenes/ and "
                         "Textures/). If omitted, Packman downloads it via NRD-Sample.")
    ap.add_argument("--nri-data-version", default=NRI_DATA_VERSION,
                    help=f"nri_data package version for the Packman pull (default: {NRI_DATA_VERSION})")
    ap.add_argument("--dest",
                    default=os.path.join(REPO_ROOT, "UnityProject", "Assets", "NRD-Sample", "Assets"),
                    help="Unity NRD-Sample assets folder "
                         "(default: <repo>/UnityProject/Assets/NRD-Sample/Assets)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-copy/re-convert even if the payload already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                    help="parallel worker threads (default: CPU count)")
    args = ap.parse_args()

    args.src = resolve_src(args)
    if not os.path.isdir(args.src):
        if args.dry_run:
            print(f"(dry run) src not present yet: {args.src} -- skipping plan.")
            return
        print(f"error: --src not found: {args.src}")
        sys.exit(1)
    if not os.path.isdir(args.dest):
        print(f"error: --dest not found: {args.dest} (run from the repo root?)")
        sys.exit(1)

    jobs = list(find_jobs(args.dest, args.src, args.overwrite))
    if not jobs:
        print("Nothing to do: every .meta already has its payload.")
        return

    load_dds = to_png_image = None
    if any(a == "convert" for a, *_ in jobs) and not args.dry_run:
        sys.path.insert(0, SCRIPT_DIR)
        try:
            from dds_to_png import load_dds, to_png_image
        except ImportError as e:
            print(f"error: DDS conversion needs dds_to_png.py + Pillow ({e})")
            sys.exit(1)

    missing = [rel for action, rel, *_ in jobs if action == "missing"]
    work = [j for j in jobs if j[0] != "missing"]

    def run_one(job):
        action, rel, src_path, dst_path = job
        if args.dry_run:
            return action, f"  {action.upper():7s} {rel}"
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if action == "generate":
                with open(dst_path, "wb") as f:
                    f.write(GENERATED[rel])
                return action, f"  GEN     {rel}  (1x1 white png)"
            if action == "copy":
                shutil.copy2(src_path, dst_path)
                return action, f"  COPY    {rel}"
            im = to_png_image(load_dds(src_path))
            im.save(dst_path)
            return action, f"  CONVERT {rel}  ({im.mode} {im.size[0]}x{im.size[1]})"
        except Exception as e:
            return "error", f"  ERROR   {rel}  {type(e).__name__}: {e}"

    copied = converted = generated = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for action, line in pool.map(run_one, work):
            print(line)
            copied += action == "copy"
            converted += action == "convert"
            generated += action == "generate"
            failed += action == "error"

    print(f"\nDone{' (dry run)' if args.dry_run else ''}: "
          f"{copied} copied, {converted} converted from DDS, "
          f"{generated} generated, {failed} failed, {len(missing)} missing.")
    for rel in missing:
        tag = "  (expected in git -- commit it with: git add -f)" if rel in HAND_AUTHORED else ""
        print(f"  MISSING {rel}{tag}")
    if failed or any(rel not in HAND_AUTHORED for rel in missing):
        sys.exit(2)


if __name__ == "__main__":
    main()
