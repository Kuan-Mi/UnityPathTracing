"""
restore_rtxpt_assets.py
-----------------------
Restore the git-ignored payload files under UnityProject/Assets/RTXPTAssets
(*.png / *.jpg / *.glb / *.bin, see UnityProject/.gitignore) from an RTXPT
asset checkout (e.g. F:\\RTXPT\\Assets), so the project can be opened on
another computer.

The repo keeps every Unity .meta sidecar; this script walks them, finds metas
whose payload file is missing, and restores the payload:

  1. same relative path exists in --src            -> copy as-is
  2. missing .png but <stem>.dds exists in --src   -> decode DDS, save PNG
     (uses load_dds/to_png_image from dds_to_png.py, same conversion the
     originals were produced with)
  3. otherwise                                     -> reported as MISSING
     (a handful of hand-authored textures are expected to live in git itself;
      if they show up here they were not committed -- see --help epilog)

Dependencies: Pillow (pip install Pillow) -- only needed when DDS conversion
actually has to run.

Usage (from the repo root, on the machine that has the RTXPT assets):
    python restore_rtxpt_assets.py                          # auto: clone RTXPT-Assets
    python restore_rtxpt_assets.py --src D:\\RTXPT\\Assets
    python restore_rtxpt_assets.py --dry-run                # show the plan only
    python restore_rtxpt_assets.py --overwrite              # refresh existing files

When --src is omitted, the source is resolved automatically:
    1. F:\\RTXPT\\Assets if it exists (legacy local checkout), else
    2. the NVIDIA-RTX/RTXPT-Assets repo is cloned into <repo>/tools/.rtxpt-assets
       (an existing clone there is reused). Git + Git LFS must be installed.
"""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # this script lives in <repo>/tools

# When --src is omitted and no legacy local checkout is found, clone this repo.
# Its root maps directly to the RTXPT "Assets" layout (EnvironmentMaps/, Models/,
# Materials/, ...), so the clone directory is used as --src as-is.
ASSETS_REPO_URL = "https://github.com/NVIDIA-RTX/RTXPT-Assets.git"
LEGACY_SRC = r"F:\RTXPT\Assets"
DEFAULT_CLONE_DIR = os.path.join(SCRIPT_DIR, ".rtxpt-assets")

# Hand-authored files that are NOT derivable from the RTXPT assets and are
# expected to be committed to git directly (git add -f). Listed so the
# summary can tell "expected in git" apart from a genuinely broken setup.
# Currently empty: every payload is restorable from the RTXPT assets.
HAND_AUTHORED = set()


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
            src_path = os.path.join(src, rel)
            if os.path.isfile(src_path):
                yield "copy", rel, src_path, dst_path
                continue
            stem, ext = os.path.splitext(src_path)
            if ext.lower() == ".png" and os.path.isfile(stem + ".dds"):
                yield "convert", rel, stem + ".dds", dst_path
                continue
            yield "missing", rel, None, dst_path


def ensure_assets_clone(clone_dir, dry_run):
    """Clone NVIDIA-RTX/RTXPT-Assets into clone_dir (reuse if already there)."""
    if os.path.isdir(os.path.join(clone_dir, ".git")):
        print(f"Using existing RTXPT-Assets clone: {clone_dir}")
        return clone_dir
    if dry_run:
        print(f"(dry run) would clone {ASSETS_REPO_URL} -> {clone_dir}")
        return clone_dir
    print(f"Cloning {ASSETS_REPO_URL}")
    print(f"     -> {clone_dir}  (large; Git LFS must be installed)")
    try:
        subprocess.run(["git", "clone", "--depth", "1", ASSETS_REPO_URL, clone_dir],
                       check=True)
    except FileNotFoundError:
        print("error: 'git' not found on PATH; cannot auto-clone RTXPT-Assets.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"error: git clone failed (exit {e.returncode}).")
        sys.exit(1)
    return clone_dir


def resolve_src(args):
    """Return the asset source root, cloning RTXPT-Assets if --src was omitted."""
    if args.src is not None:
        return args.src
    if os.path.isdir(LEGACY_SRC):
        print(f"Using legacy local assets: {LEGACY_SRC}")
        return LEGACY_SRC
    return ensure_assets_clone(args.clone_dir, args.dry_run)


def main():
    ap = argparse.ArgumentParser(
        description="Restore git-ignored RTXPTAssets payloads from an RTXPT asset checkout.",
        epilog="MISSING files marked '(expected in git)' are hand-authored and "
               "must be committed with: git add -f <file>  on the source machine.")
    ap.add_argument("--src", default=None,
                    help=r"RTXPT assets root. If omitted: use F:\RTXPT\Assets when "
                         "present, otherwise clone NVIDIA-RTX/RTXPT-Assets.")
    ap.add_argument("--clone-dir", default=DEFAULT_CLONE_DIR,
                    help="where to clone RTXPT-Assets when --src is omitted "
                         "(default: <repo>/tools/.rtxpt-assets)")
    ap.add_argument("--dest",
                    default=os.path.join(REPO_ROOT, "UnityProject", "Assets", "RTXPTAssets"),
                    help="Unity RTXPTAssets folder (default: <repo>/UnityProject/Assets/RTXPTAssets)")
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
