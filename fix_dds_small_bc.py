"""
fix_dds_small_bc.py
-------------------
将目录下所有 BC4/BC5 格式的 DDS 贴图中，宽或高小于4的文件（如1x1），
解码出实际像素值，重新生成一张 4x4 的无压缩 DDS（R8/R8G8 格式）。

无压缩格式 Unity 可以直接读取，importer 会按材质设置重新压缩。

目录已硬编码，直接运行即可。
"""

import struct
import os
import sys

TARGET_DIR = r'.\UnityProject\Assets\Gltf\Textures'

DDS_MAGIC  = b'DDS '
DDPF_FOURCC = 0x4

# DDS_HEADER field offsets (after 4-byte magic)
OFF_HEIGHT     = 12
OFF_WIDTH      = 16
OFF_PITCH      = 20
OFF_FLAGS      = 8
OFF_MIPCOUNT   = 28
OFF_PF_FLAGS   = 80
OFF_PF_FOURCC  = 84

# DXGI_FORMAT values
DXGI_R8_UNORM    = 61
DXGI_R8G8_UNORM  = 49


# ---------------------------------------------------------------------------
# BC4/BC5 alpha-block decoder
# ---------------------------------------------------------------------------
def decode_alpha_block(block8: bytes) -> list:
    """Decode one 8-byte BC4 alpha block → list of 16 uint8 values."""
    a0, a1 = block8[0], block8[1]
    # 6 bytes = 48 bits of indices (16 x 3 bits)
    bits = int.from_bytes(block8[2:8], 'little')

    if a0 > a1:
        palette = [a0, a1] + [
            ((6 - i) * a0 + (1 + i) * a1 + 3) // 7 for i in range(6)
        ]
    else:
        palette = [a0, a1] + [
            ((4 - i) * a0 + (1 + i) * a1 + 2) // 5 for i in range(4)
        ] + [0, 255]

    return [palette[(bits >> (3 * k)) & 7] for k in range(16)]


def decode_bc4_block(block: bytes) -> list:
    """Returns 16 (r,) tuples."""
    r_vals = decode_alpha_block(block[0:8])
    return [(v,) for v in r_vals]


def decode_bc5_block(block: bytes) -> list:
    """Returns 16 (r, g) tuples."""
    r_vals = decode_alpha_block(block[0:8])
    g_vals = decode_alpha_block(block[8:16])
    return list(zip(r_vals, g_vals))


# ---------------------------------------------------------------------------
# Build a valid uncompressed DDS in memory
# ---------------------------------------------------------------------------
def make_uncompressed_dds(pixels: list, width: int, height: int, channels: int) -> bytes:
    """
    Build a DDS file with DX10 header.
    channels=1 → DXGI_R8_UNORM, channels=2 → DXGI_R8G8_UNORM
    pixels: list of length width*height, each element is a tuple of uint8.
    """
    dxgi_fmt   = DXGI_R8_UNORM if channels == 1 else DXGI_R8G8_UNORM
    pitch      = width * channels
    pixel_data = bytearray()
    for px in pixels:
        for c in px:
            pixel_data.append(c)

    # DDS_HEADER (124 bytes)
    DDSD_CAPS        = 0x1
    DDSD_HEIGHT      = 0x2
    DDSD_WIDTH       = 0x4
    DDSD_PITCH       = 0x8
    DDSD_PIXELFORMAT = 0x1000
    dw_flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT

    # DDS_PIXELFORMAT (32 bytes): dwSize, dwFlags, dwFourCC, dwRGBBitCount, dwRBitMask,
    #                              dwGBitMask, dwBBitMask, dwABitMask
    pf  = struct.pack('<II', 32, DDPF_FOURCC)      # dwSize=32, dwFlags
    pf += b'DX10'                                   # dwFourCC
    pf += struct.pack('<IIIII', 0, 0, 0, 0, 0)     # dwRGBBitCount + 4 masks = 20 bytes

    # DDS_HEADER: dwSize(4)+dwFlags(4)+dwHeight(4)+dwWidth(4)+dwPitchOrLinearSize(4)+
    #             dwDepth(4)+dwMipMapCount(4)+dwReserved1[11](44)+ddspf(32)+
    #             dwCaps(4)+dwCaps2(4)+dwCaps3(4)+dwCaps4(4)+dwReserved2(4) = 124
    header  = struct.pack('<II',    124, dw_flags)          # dwSize, dwFlags
    header += struct.pack('<IIIII', height, width, pitch, 1, 1)  # h, w, pitch, depth, mipCount
    header += b'\x00' * 44                                  # dwReserved1[11]
    header += pf                                            # DDS_PIXELFORMAT (32 bytes)
    header += struct.pack('<IIIII', 0x1000, 0, 0, 0, 0)    # caps1-4, reserved2

    # DDS_HEADER_DXT10 (20 bytes)
    dx10 = struct.pack('<IIIII',
        dxgi_fmt,   # dxgiFormat
        3,          # resourceDimension = D3D10_RESOURCE_DIMENSION_TEXTURE2D
        0,          # miscFlag
        1,          # arraySize
        0,          # miscFlags2
    )

    return DDS_MAGIC + header + dx10 + bytes(pixel_data)


# ---------------------------------------------------------------------------
# Process one file
# ---------------------------------------------------------------------------
def process_file(path: str) -> bool:
    with open(path, 'rb') as f:
        data = f.read()

    if len(data) < 128 or data[:4] != DDS_MAGIC:
        return False

    pf_flags = struct.unpack_from('<I', data, OFF_PF_FLAGS)[0]
    if not (pf_flags & DDPF_FOURCC):
        return False

    fourcc   = data[OFF_PF_FOURCC: OFF_PF_FOURCC + 4]
    width    = struct.unpack_from('<I', data, OFF_WIDTH)[0]
    height   = struct.unpack_from('<I', data, OFF_HEIGHT)[0]

    # Only process files that are actually smaller than 4x4
    if width >= 4 and height >= 4:
        return False

    # Determine format and pixel data offset
    if fourcc in (b'ATI2', b'BC5U', b'BC5S'):
        fmt_name   = 'BC5'
        block_size = 16
        channels   = 2
        decode_fn  = decode_bc5_block
        data_off   = 128
    elif fourcc in (b'ATI1', b'BC4U', b'BC4S'):
        fmt_name   = 'BC4'
        block_size = 8
        channels   = 1
        decode_fn  = decode_bc4_block
        data_off   = 128
    elif fourcc == b'DX10' and len(data) >= 148:
        dxgi = struct.unpack_from('<I', data, 128)[0]
        if dxgi in (82, 83, 84):   # BC5
            fmt_name, block_size, channels, decode_fn = 'BC5', 16, 2, decode_bc5_block
        elif dxgi in (79, 80, 81): # BC4
            fmt_name, block_size, channels, decode_fn = 'BC4', 8,  1, decode_bc4_block
        else:
            return False
        data_off = 148
    else:
        return False

    if len(data) < data_off + block_size:
        return False

    # Decode the single block (covers 4x4, we take all 16 pixel values)
    block  = data[data_off: data_off + block_size]
    pixels = decode_fn(block)  # 16 tuples

    # Build a 4x4 pixel grid
    new_dds = make_uncompressed_dds(pixels, 4, 4, channels)

    with open(path, 'wb') as f:
        f.write(new_dds)

    print(f'  {fmt_name} {width}x{height} -> 4x4 uncompressed: {os.path.basename(path)}')
    return True


# ---------------------------------------------------------------------------
def main():
    root = TARGET_DIR
    if not os.path.isdir(root):
        print(f'错误: 目录不存在: {root}')
        sys.exit(1)

    changed = 0
    scanned = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.dds'):
                scanned += 1
                if process_file(os.path.join(dirpath, fn)):
                    changed += 1

    print(f'\n扫描 {scanned} 个DDS文件，已修改 {changed} 个。')


if __name__ == '__main__':
    main()

