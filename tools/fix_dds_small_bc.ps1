<#
fix_dds_small_bc.ps1
--------------------
将目录下所有 BC4/BC5 格式的 DDS 贴图中，宽或高小于4的文件（如1x1），
解码出实际像素值，重新生成一张 4x4 的无压缩 DDS（R8/R8G8 格式）。
无需 Python 环境，纯 PowerShell 实现。
#>

$TARGET_DIR = Join-Path $PSScriptRoot 'UnityProject\Assets\Gltf\Textures'

# ---------------------------------------------------------------------------
# BC4/BC5 解码
# ---------------------------------------------------------------------------
function Decode-AlphaBlock([byte[]]$b8) {
    $a0 = [int]$b8[0]
    $a1 = [int]$b8[1]

    # 6 bytes → 48 bits (little-endian)
    $bits = [long]0
    for ($i = 0; $i -lt 6; $i++) {
        $bits = $bits -bor ([long]$b8[2 + $i] -shl ($i * 8))
    }

    $palette = [int[]]::new(8)
    $palette[0] = $a0
    $palette[1] = $a1
    if ($a0 -gt $a1) {
        for ($i = 0; $i -lt 6; $i++) {
            $palette[2 + $i] = [int][Math]::Floor(((6 - $i) * $a0 + (1 + $i) * $a1 + 3) / 7.0)
        }
    } else {
        for ($i = 0; $i -lt 4; $i++) {
            $palette[2 + $i] = [int][Math]::Floor(((4 - $i) * $a0 + (1 + $i) * $a1 + 2) / 5.0)
        }
        $palette[6] = 0
        $palette[7] = 255
    }

    $result = [int[]]::new(16)
    for ($k = 0; $k -lt 16; $k++) {
        $result[$k] = $palette[($bits -shr (3 * $k)) -band 7]
    }
    return $result
}

# 返回 flat byte array，每像素 1 字节 (R)
function Decode-BC4Block([byte[]]$block) {
    return Decode-AlphaBlock $block[0..7]
}

# 返回 flat byte array，每像素 2 字节 (RG 交错)
function Decode-BC5Block([byte[]]$block) {
    $r = Decode-AlphaBlock $block[0..7]
    $g = Decode-AlphaBlock $block[8..15]
    $out = [byte[]]::new(32)
    for ($i = 0; $i -lt 16; $i++) {
        $out[$i * 2]     = [byte]$r[$i]
        $out[$i * 2 + 1] = [byte]$g[$i]
    }
    return $out
}

# ---------------------------------------------------------------------------
# 构建无压缩 DDS (DX10 扩展头)
# ---------------------------------------------------------------------------
function Make-UncompressedDDS([byte[]]$pixels, [int]$width, [int]$height, [int]$channels) {
    $dxgiFmt = if ($channels -eq 1) { [uint32]61 } else { [uint32]49 }  # R8 or R8G8
    $pitch   = [uint32]($width * $channels)

    # DDS_PIXELFORMAT (32 bytes)
    $pf = [byte[]]::new(32)
    [System.BitConverter]::GetBytes([uint32]32).CopyTo($pf, 0)   # dwSize
    [System.BitConverter]::GetBytes([uint32]4).CopyTo($pf, 4)    # dwFlags = DDPF_FOURCC
    [System.Text.Encoding]::ASCII.GetBytes('DX10').CopyTo($pf, 8) # dwFourCC
    # remaining 20 bytes = 0

    # DDS_HEADER (124 bytes)
    $dwFlags = [uint32](0x1 -bor 0x2 -bor 0x4 -bor 0x8 -bor 0x1000)
    $hdr = [byte[]]::new(124)
    [System.BitConverter]::GetBytes([uint32]124).CopyTo($hdr, 0)      # dwSize
    [System.BitConverter]::GetBytes($dwFlags).CopyTo($hdr, 4)         # dwFlags
    [System.BitConverter]::GetBytes([uint32]$height).CopyTo($hdr, 8)  # dwHeight
    [System.BitConverter]::GetBytes([uint32]$width).CopyTo($hdr, 12)  # dwWidth
    [System.BitConverter]::GetBytes($pitch).CopyTo($hdr, 16)          # dwPitchOrLinearSize
    [System.BitConverter]::GetBytes([uint32]1).CopyTo($hdr, 20)       # dwDepth
    [System.BitConverter]::GetBytes([uint32]1).CopyTo($hdr, 24)       # dwMipMapCount
    # offset 28..71: dwReserved1[11] = zeros (44 bytes)
    $pf.CopyTo($hdr, 72)                                               # DDS_PIXELFORMAT
    [System.BitConverter]::GetBytes([uint32]0x1000).CopyTo($hdr, 104) # dwCaps
    # offset 108..123: zeros

    # DDS_HEADER_DXT10 (20 bytes)
    $dx10 = [byte[]]::new(20)
    [System.BitConverter]::GetBytes($dxgiFmt).CopyTo($dx10, 0)    # dxgiFormat
    [System.BitConverter]::GetBytes([uint32]3).CopyTo($dx10, 4)   # D3D10_RESOURCE_DIMENSION_TEXTURE2D
    [System.BitConverter]::GetBytes([uint32]0).CopyTo($dx10, 8)   # miscFlag
    [System.BitConverter]::GetBytes([uint32]1).CopyTo($dx10, 12)  # arraySize
    [System.BitConverter]::GetBytes([uint32]0).CopyTo($dx10, 16)  # miscFlags2

    $magic = [byte[]](0x44, 0x44, 0x53, 0x20)
    $out = [System.Collections.Generic.List[byte]]::new(4 + 124 + 20 + $pixels.Length)
    $out.AddRange($magic)
    $out.AddRange($hdr)
    $out.AddRange($dx10)
    $out.AddRange($pixels)
    return $out.ToArray()
}

# ---------------------------------------------------------------------------
# 处理单个文件
# ---------------------------------------------------------------------------
function Process-File([string]$path) {
    $data = [System.IO.File]::ReadAllBytes($path)

    if ($data.Length -lt 128) { return $false }

    # 检查 DDS 魔数
    if ($data[0] -ne 0x44 -or $data[1] -ne 0x44 -or $data[2] -ne 0x53 -or $data[3] -ne 0x20) {
        return $false
    }

    $pfFlags = [System.BitConverter]::ToUInt32($data, 80)
    if (-not ($pfFlags -band 0x4)) { return $false }

    $fourcc = [System.Text.Encoding]::ASCII.GetString($data, 84, 4)
    $width  = [int][System.BitConverter]::ToUInt32($data, 16)
    $height = [int][System.BitConverter]::ToUInt32($data, 12)

    if ($width -ge 4 -and $height -ge 4) { return $false }

    $fmtName   = $null
    $blockSize = 0
    $channels  = 0
    $dataOff   = 0
    $isBC5     = $false

    if ($fourcc -in 'ATI2', 'BC5U', 'BC5S') {
        $fmtName = 'BC5'; $blockSize = 16; $channels = 2; $dataOff = 128; $isBC5 = $true
    } elseif ($fourcc -in 'ATI1', 'BC4U', 'BC4S') {
        $fmtName = 'BC4'; $blockSize = 8;  $channels = 1; $dataOff = 128; $isBC5 = $false
    } elseif ($fourcc -eq 'DX10' -and $data.Length -ge 148) {
        $dxgi = [int][System.BitConverter]::ToUInt32($data, 128)
        if ($dxgi -ge 82 -and $dxgi -le 84) {
            $fmtName = 'BC5'; $blockSize = 16; $channels = 2; $dataOff = 148; $isBC5 = $true
        } elseif ($dxgi -ge 79 -and $dxgi -le 81) {
            $fmtName = 'BC4'; $blockSize = 8;  $channels = 1; $dataOff = 148; $isBC5 = $false
        } else {
            return $false
        }
    } else {
        return $false
    }

    if ($data.Length -lt $dataOff + $blockSize) { return $false }

    $block = $data[$dataOff..($dataOff + $blockSize - 1)]

    if ($isBC5) {
        $pixels = Decode-BC5Block $block
    } else {
        $raw    = Decode-BC4Block $block
        $pixels = [byte[]]::new(16)
        for ($i = 0; $i -lt 16; $i++) { $pixels[$i] = [byte]$raw[$i] }
    }

    $newDds = Make-UncompressedDDS $pixels 4 4 $channels
    [System.IO.File]::WriteAllBytes($path, $newDds)

    Write-Host "  $fmtName ${width}x${height} -> 4x4 uncompressed: $(Split-Path $path -Leaf)"
    return $true
}

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------
if (-not (Test-Path $TARGET_DIR)) {
    Write-Error "Directory not found: $TARGET_DIR"
    exit 1
}

$scanned = 0
$changed = 0

Get-ChildItem -Path $TARGET_DIR -Recurse -Filter '*.dds' | ForEach-Object {
    $scanned++
    if (Process-File $_.FullName) { $changed++ }
}

Write-Host "`nScanned $scanned DDS file(s), fixed $changed."
