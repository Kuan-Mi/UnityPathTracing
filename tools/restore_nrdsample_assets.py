"""
restore_nrdsample_assets.py
---------------------------
Restore the git-ignored Bistro scene payloads under UnityProject/Assets/Gltf
(BistroInterior.bin, BistroExterior.bin, Textures/*, see UnityProject/.gitignore)
from the NRD-Sample _Data checkout, so the project can be opened on another
computer.

The Bistro data is the same payload download_bistro.bat fetches via Packman;
this script resolves that source (auto-running Packman when needed) and restores
the files into Assets/Gltf:

  1. BistroInterior.bin / BistroExterior.bin           -> copy as-is
  2. Textures/<name>.dds                               -> decode DDS, save PNG
     (uses load_dds/to_png_image from dds_to_png.py, the same conversion the
      RTXPTAssets originals were produced with; it handles the BC4/BC5 1x1
      textures download_bistro.bat patches with fix_dds_small_bc.ps1, so no
      separate fix step is needed)
  3. Textures/<name>.png|.jpg already in the source     -> copy as-is
  4. otherwise                                          -> reported as MISSING

Dependencies: Pillow (pip install Pillow) -- only needed when DDS conversion
actually has to run.

Usage (from the repo root):
    python restore_nrdsample_assets.py                     # auto: Packman pull
    python restore_nrdsample_assets.py --src D:\\Bistro     # use an existing copy
    python restore_nrdsample_assets.py --dry-run           # show the plan only
    python restore_nrdsample_assets.py --overwrite         # refresh existing files

When --src is omitted, the source is resolved automatically:
    1. RenderingPlugin/External/NRD-Sample/_Data/Scenes/Bistro if it already
       exists (a previous Packman pull), else
    2. Packman is run against NRD-Sample/Dependencies.xml to download the
       nri_data package (same call download_bistro.bat makes). The NRD-Sample
       submodule -- including its bundled Packman -- must be present.
"""

import argparse
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
DATA_DIR = os.path.join(NRD_SAMPLE_DIR, "_Data")
DEFAULT_SRC = os.path.join(DATA_DIR, "Scenes", "Bistro")
NRI_DATA_VERSION = "2.3"

# The two scene buffers live at the Bistro root; everything else we want lives
# under Textures/ (the only two things download_bistro.bat copies).
BIN_FILES = ("BistroInterior.bin", "BistroExterior.bin")
TEXTURES_DIR = "Textures"

# Files that are NOT derivable from the Bistro data and are expected to be
# committed to git directly. Currently empty: every payload is restorable.
HAND_AUTHORED = set()


def find_jobs(dest, src, overwrite):
    """Yield (action, rel, src_path, dst_path); action in copy/convert/missing."""
    # the two scene buffers at the Bistro root
    for name in BIN_FILES:
        dst_path = os.path.join(dest, name)
        if os.path.exists(dst_path) and not overwrite:
            continue
        src_path = os.path.join(src, name)
        if os.path.isfile(src_path):
            yield "copy", name, src_path, dst_path
        else:
            yield "missing", name, None, dst_path

    # every texture under Textures/ (.dds -> PNG, other images copied as-is)
    tex_src = os.path.join(src, TEXTURES_DIR)
    for dp, _, fs in os.walk(tex_src):
        for fn in fs:
            stem, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext == ".dds":
                action, out_name, payload = "convert", stem + ".png", os.path.join(dp, fn)
            elif ext in (".png", ".jpg", ".jpeg"):
                action, out_name, payload = "copy", fn, os.path.join(dp, fn)
            else:
                continue  # skip .meta and any non-image sidecars in the source
            rel = os.path.relpath(os.path.join(dp, out_name), src).replace("\\", "/")
            dst_path = os.path.join(dest, rel.replace("/", os.sep))
            if os.path.exists(dst_path) and not overwrite:
                continue
            yield action, rel, payload, dst_path


def ensure_packman_pull(dry_run, version):
    """Download the Bistro data via the NRD-Sample Packman (reuse if present)."""
    if os.path.isdir(DEFAULT_SRC):
        print(f"Using existing NRD-Sample Bistro data: {DEFAULT_SRC}")
        return DEFAULT_SRC
    if dry_run:
        print(f"(dry run) would run Packman to download Bistro data -> {DEFAULT_SRC}")
        return DEFAULT_SRC
    if not os.path.isfile(PACKMAN_CMD):
        print(f"error: Packman not found: {PACKMAN_CMD}")
        print("       init the NRD-Sample submodule under RenderingPlugin/External first.")
        sys.exit(1)
    print("Downloading Bistro scene data via Packman")
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
    if not os.path.isdir(DEFAULT_SRC):
        print(f"error: Bistro data not found after Packman pull: {DEFAULT_SRC}")
        sys.exit(1)
    return DEFAULT_SRC


def resolve_src(args):
    """Return the Bistro source root, running Packman if --src was omitted."""
    if args.src is not None:
        return args.src
    return ensure_packman_pull(args.dry_run, args.nri_data_version)


def main():
    ap = argparse.ArgumentParser(
        description="Restore git-ignored Assets/Gltf Bistro payloads from the NRD-Sample data.",
        epilog="MISSING files marked '(expected in git)' are hand-authored and "
               "must be committed with: git add -f <file>  on the source machine.")
    ap.add_argument("--src", default=None,
                    help="Bistro data root (the folder holding BistroInterior.bin and "
                         "Textures/). If omitted, Packman downloads it via NRD-Sample.")
    ap.add_argument("--nri-data-version", default=NRI_DATA_VERSION,
                    help=f"nri_data package version for the Packman pull (default: {NRI_DATA_VERSION})")
    ap.add_argument("--dest",
                    default=os.path.join(REPO_ROOT, "UnityProject", "Assets", "Gltf"),
                    help="Unity Gltf folder (default: <repo>/UnityProject/Assets/Gltf)")
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
        print("Nothing to do: every Bistro payload is already present.")
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
            if action == "copy":
                shutil.copy2(src_path, dst_path)
                return action, f"  COPY    {rel}"
            im = to_png_image(load_dds(src_path))
            im.save(dst_path)
            return action, f"  CONVERT {rel}  ({im.mode} {im.size[0]}x{im.size[1]})"
        except Exception as e:
            return "error", f"  ERROR   {rel}  {type(e).__name__}: {e}"

    copied = converted = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for action, line in pool.map(run_one, work):
            print(line)
            copied += action == "copy"
            converted += action == "convert"
            failed += action == "error"

    print(f"\nDone{' (dry run)' if args.dry_run else ''}: "
          f"{copied} copied, {converted} converted from DDS, "
          f"{failed} failed, {len(missing)} missing.")
    for rel in missing:
        tag = "  (expected in git -- commit it with: git add -f)" if rel in HAND_AUTHORED else ""
        print(f"  MISSING {rel}{tag}")
    if failed or any(rel not in HAND_AUTHORED for rel in missing):
        sys.exit(2)


if __name__ == "__main__":
    main()
