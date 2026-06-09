"""
dds_to_png.py
-------------
将指定文件夹（默认递归）下的所有 DDS 贴图转换为 PNG。
原始 .dds 文件保留不动，PNG 与其同名同目录生成。

依赖: Pillow (pip install Pillow)

用法:
    python dds_to_png.py                       # 转换默认目录 UnityProject/Assets
    python dds_to_png.py <folder>              # 转换指定目录
    python dds_to_png.py <folder> --overwrite  # 覆盖已存在的 PNG
    python dds_to_png.py <folder> --no-recursive

Pillow 自带 BCn 解码器可处理本工程中绝大多数格式
(DXT1/BC1, DXT5/BC3, ATI2/BC5, BC7)。两类例外由本脚本手动兜底:
  1. DX10 头的块压缩格式 (如 DXGI BC3=77) -> 重写成等价的 legacy-FourCC
     DDS 再交给 Pillow 解码。
  2. DX10 头的非压缩 R8 / R8G8 -> 直接读原始像素 (R8G8 法线图重建 B 通道)。
"""

import argparse
import io
import math
import os
import struct
import sys

from PIL import Image

# ---------------------------------------------------------------------------
# DDS header constants
# ---------------------------------------------------------------------------
DDS_MAGIC   = b'DDS '
DDPF_FOURCC = 0x4
OFF_HEIGHT  = 12
OFF_WIDTH   = 16
OFF_PF_FLAGS = 80
OFF_PF_FOURCC = 84
DX10_DATA_OFF = 148   # 4 magic + 124 header + 20 DX10 chunk
LEGACY_DATA_OFF = 128  # 4 magic + 124 header

# DXGI_FORMAT -> legacy FourCC for block-compressed formats Pillow understands
# via its legacy path. (UNORM / UNORM_SRGB / TYPELESS variants all share layout.)
DXGI_TO_FOURCC = {
    70: b'DXT1', 71: b'DXT1', 72: b'DXT1',          # BC1
    73: b'DXT3', 74: b'DXT3', 75: b'DXT3',          # BC2
    76: b'DXT5', 77: b'DXT5', 78: b'DXT5',          # BC3
    79: b'ATI1', 80: b'ATI1',                       # BC4
    82: b'ATI2', 83: b'ATI2', 84: b'ATI2',          # BC5
}
DXGI_R8_UNORM   = 61
DXGI_R8G8_UNORM = 49


def _read_dims(d):
    h = struct.unpack_from('<I', d, OFF_HEIGHT)[0]
    w = struct.unpack_from('<I', d, OFF_WIDTH)[0]
    return w, h


def _rewrap_dx10_bcn_as_legacy(d, fourcc):
    """Build a legacy-FourCC DDS (no DX10 chunk) so Pillow's BCn path accepts it."""
    header = bytearray(d[4:LEGACY_DATA_OFF])     # copy the 124-byte DDS_HEADER
    # Force pixelformat to FOURCC + the legacy code (offsets relative to header start).
    struct.pack_into('<I', header, OFF_PF_FLAGS - 4, DDPF_FOURCC)
    header[OFF_PF_FOURCC - 4: OFF_PF_FOURCC - 4 + 4] = fourcc
    return DDS_MAGIC + bytes(header) + d[DX10_DATA_OFF:]


def _decode_uncompressed_rg(d, dxgi):
    """Read a DX10 R8 / R8G8 surface (used by the 4x4 fallback normals/masks)."""
    w, h = _read_dims(d)
    chan = 2 if dxgi == DXGI_R8G8_UNORM else 1
    raw = d[DX10_DATA_OFF: DX10_DATA_OFF + w * h * chan]
    if len(raw) < w * h * chan:
        return None
    if chan == 1:
        return Image.frombytes('L', (w, h), raw)
    # R8G8: treat as a tangent-space normal, reconstruct B = sqrt(1 - x^2 - y^2).
    out = bytearray(w * h * 3)
    for i in range(w * h):
        r = raw[2 * i]
        g = raw[2 * i + 1]
        nx = r / 255.0 * 2.0 - 1.0
        ny = g / 255.0 * 2.0 - 1.0
        nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
        out[3 * i]     = r
        out[3 * i + 1] = g
        out[3 * i + 2] = int(round((nz * 0.5 + 0.5) * 255.0))
    return Image.frombytes('RGB', (w, h), bytes(out))


def load_dds(path):
    """Return a PIL.Image for a DDS file, with fallbacks for formats Pillow rejects."""
    try:
        im = Image.open(path)
        im.load()
        return im
    except NotImplementedError:
        pass  # fall through to manual handling

    with open(path, 'rb') as f:
        d = f.read()
    if d[:4] != DDS_MAGIC or d[OFF_PF_FOURCC: OFF_PF_FOURCC + 4] != b'DX10':
        raise NotImplementedError('unsupported DDS Pillow could not decode')

    dxgi = struct.unpack_from('<I', d, LEGACY_DATA_OFF)[0]
    if dxgi in (DXGI_R8_UNORM, DXGI_R8G8_UNORM):
        im = _decode_uncompressed_rg(d, dxgi)
        if im is not None:
            return im
    fourcc = DXGI_TO_FOURCC.get(dxgi)
    if fourcc:
        im = Image.open(io.BytesIO(_rewrap_dx10_bcn_as_legacy(d, fourcc)))
        im.load()
        return im
    raise NotImplementedError(f'unhandled DXGI format {dxgi}')


def to_png_image(im):
    """Drop a fully-opaque alpha channel so we don't bloat single-channel/RGB textures."""
    if im.mode == 'RGBA':
        alpha = im.getchannel('A')
        if alpha.getextrema() == (255, 255):
            return im.convert('RGB')
        return im
    if im.mode in ('RGB', 'L', 'LA'):
        return im
    return im.convert('RGBA')


def main():
    ap = argparse.ArgumentParser(description='Convert DDS textures in a folder to PNG (originals kept).')
    ap.add_argument('folder', nargs='?', default='UnityProject/Assets',
                    help='folder to scan (default: UnityProject/Assets)')
    ap.add_argument('--overwrite', action='store_true', help='overwrite existing .png files')
    ap.add_argument('--no-recursive', dest='recursive', action='store_false',
                    help='do not descend into subfolders')
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print(f'错误: 目录不存在: {args.folder}')
        sys.exit(1)

    walker = os.walk(args.folder) if args.recursive else [(args.folder, [], os.listdir(args.folder))]
    converted = skipped = failed = 0
    for dp, _, fs in walker:
        for fn in fs:
            if not fn.lower().endswith('.dds'):
                continue
            src = os.path.join(dp, fn)
            dst = os.path.splitext(src)[0] + '.png'
            if os.path.exists(dst) and not args.overwrite:
                skipped += 1
                continue
            try:
                im = to_png_image(load_dds(src))
                im.save(dst)
                converted += 1
                print(f'  OK  {os.path.relpath(src, args.folder)}  -> {im.mode} {im.size}')
            except Exception as e:
                failed += 1
                print(f'  ERR {os.path.relpath(src, args.folder)}  {type(e).__name__}: {e}')

    print(f'\n完成: 转换 {converted}, 跳过(已存在) {skipped}, 失败 {failed}.')
    if failed:
        sys.exit(2)


if __name__ == '__main__':
    main()
