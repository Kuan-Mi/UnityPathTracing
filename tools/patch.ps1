Add-Type -AssemblyName System.Windows.Forms

$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "Select file to patch"
$dialog.Filter = "Executable (*.exe)|*.exe|All files (*.*)|*.*"
$dialog.InitialDirectory = [Environment]::GetFolderPath("Desktop")

if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

$ExePath = $dialog.FileName

if (-not (Test-Path $ExePath)) {
    Write-Host "File not found: $ExePath" -ForegroundColor Red
    exit 1
}

$bytes = [System.IO.File]::ReadAllBytes($ExePath)
$pattern = @(0x6A, 0x02, 0x00, 0x00)
$replace = @(0x6B, 0x02, 0x00, 0x00)
$count = 0

for ($i = 0; $i -le $bytes.Length - $pattern.Length; $i++) {
    if ($bytes[$i] -eq $pattern[0] -and $bytes[$i+1] -eq $pattern[1] -and
        $bytes[$i+2] -eq $pattern[2] -and $bytes[$i+3] -eq $pattern[3]) {
        Write-Host "Found match at offset 0x$($i.ToString('X8'))"
        for ($j = 0; $j -lt $replace.Length; $j++) {
            $bytes[$i + $j] = $replace[$j]
        }
        $count++
    }
}

if ($count -eq 0) {
    Write-Host "Pattern 6A 02 00 00 not found." -ForegroundColor Yellow
} else {
    $backupPath = $ExePath + ".bak"
    Copy-Item -Path $ExePath -Destination $backupPath -Force
    Write-Host "Backup saved to: $backupPath" -ForegroundColor Cyan
    [System.IO.File]::WriteAllBytes($ExePath, $bytes)
    Write-Host "Patched $count occurrence(s). (6A 02 -> 6B 02, SDK 618 -> 619)" -ForegroundColor Green
}

# Download Agility SDK 1.619.1 and overwrite the D3D12 folder next to the exe
$exeDir = Split-Path -Parent $ExePath
$d3d12Dir = Join-Path $exeDir "D3D12"

Write-Host ""
Write-Host "Downloading Agility SDK v1.619.1 from NuGet..." -ForegroundColor Cyan

$nugetUrl = "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.619.1"
$tmpZip = Join-Path $env:TEMP "agilitysdk_1.619.1.nupkg"

try {
    Invoke-WebRequest -Uri $nugetUrl -OutFile $tmpZip -UseBasicParsing
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    exit 1
}

$tmpExtract = Join-Path $env:TEMP "agilitysdk_1.619.1"
if (Test-Path $tmpExtract) {
    Remove-Item $tmpExtract -Recurse -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($tmpZip, $tmpExtract)

$sdkBinDir = Join-Path $tmpExtract "build\native\bin\x64"
if (-not (Test-Path $sdkBinDir)) {
    Write-Host "Could not find SDK binaries in package (expected: build\native\bin\x64)" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $d3d12Dir)) {
    New-Item -ItemType Directory -Path $d3d12Dir | Out-Null
}

foreach ($dll in @("D3D12Core.dll", "d3d12SDKLayers.dll")) {
    $dest = Join-Path $d3d12Dir $dll
    if (Test-Path $dest) {
        $bakDest = $dest + ".bak"
        Copy-Item -Path $dest -Destination $bakDest -Force
        Write-Host "Backup saved to: $bakDest" -ForegroundColor Cyan
    }
}

Copy-Item -Path (Join-Path $sdkBinDir "D3D12Core.dll")       -Destination $d3d12Dir -Force
Copy-Item -Path (Join-Path $sdkBinDir "d3d12SDKLayers.dll")  -Destination $d3d12Dir -Force

Remove-Item $tmpZip -Force
Remove-Item $tmpExtract -Recurse -Force

Write-Host "Agility SDK 1.619.1 DLLs copied to: $d3d12Dir" -ForegroundColor Green
