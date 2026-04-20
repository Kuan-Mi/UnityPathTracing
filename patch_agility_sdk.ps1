param(
    [Parameter(Mandatory=$true)]
    [string]$ExePath
)

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
    [System.IO.File]::WriteAllBytes($ExePath, $bytes)
    Write-Host "Patched $count occurrence(s). (6A 02 -> 6B 02, SDK 618 -> 619)" -ForegroundColor Green
}
