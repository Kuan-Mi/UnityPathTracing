<#
.SYNOPSIS
Takes a single-frame PIX GPU capture of the built player at a given time mark.

.DESCRIPTION
pixtool constraints discovered the hard way (2026-07):
 - GPU capture only works for processes pixtool itself launched (attach -> PIXTOOL17).
 - pixtool state does not survive across invocations (2nd-invocation take-capture -> PIXTOOL3),
   so the whole session must be one pixtool command line. pixtool has no delay command.
 - The capture hotkey must be armed with set-gpu-capture-parameters, and the keypress must be
   injected with keybd_event + scancode (WScript SendKeys never reaches the capture layer).
 - A hotkey capture does NOT complete programmatic-capture's wait; keeping pixtool alive with
   --until-exit and collecting via save-all-captures after the game exits recovers it.

.EXAMPLE
powershell -File tools\pix_capture.ps1 -Seconds 20 -Out Other\Player_20s.wpix
#>
param(
    [int]$Seconds = 20,
    [string]$Out = "",
    [string]$Exe = "$PSScriptRoot\..\UnityProject\Build\UnityPathTracing.exe",
    [string]$WorkDir = "",
    [string]$PixTool = "C:\Program Files\Microsoft PIX\2603.25\pixtool.exe"
)

$ErrorActionPreference = 'Continue'
if (-not (Test-Path $Exe)) { Write-Error "Player exe not found: $Exe (run build_player.bat first)"; exit 1 }
if (-not (Test-Path $PixTool)) { Write-Error "pixtool not found: $PixTool"; exit 1 }
if ($WorkDir -eq "") { $WorkDir = Split-Path -Parent $Exe }
if (-not (Test-Path $WorkDir)) { Write-Error "WorkDir not found: $WorkDir"; exit 1 }
if ($Out -eq "") { $Out = Join-Path "$PSScriptRoot\..\Other" ("Player_{0}s_{1}.wpix" -f $Seconds, (Get-Date -Format yyyyMMdd_HHmmss)) }
$processName = [System.IO.Path]::GetFileNameWithoutExtension($Exe)
$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }

$staging = Join-Path $env:TEMP "pix_capture_staging"
if (Test-Path $staging) { Remove-Item "$staging\*" -Force -Confirm:$false } else { New-Item -ItemType Directory $staging | Out-Null }
$pixLog = Join-Path $env:TEMP "pix_capture_session.log"

Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;public static class KB{[DllImport("user32.dll")]public static extern void keybd_event(byte bVk,byte bScan,uint dwFlags,UIntPtr dwExtraInfo);[DllImport("user32.dll")]public static extern IntPtr GetForegroundWindow();[DllImport("user32.dll")]public static extern uint GetWindowThreadProcessId(IntPtr hWnd,out uint processId);[DllImport("user32.dll")]public static extern bool SetForegroundWindow(IntPtr hWnd);[DllImport("user32.dll")]public static extern bool ShowWindowAsync(IntPtr hWnd,int nCmdShow);[DllImport("user32.dll")]public static extern bool IsIconic(IntPtr hWnd);}'
function Send-F11 {
    [KB]::keybd_event(0x7A, 0x57, 0, [UIntPtr]::Zero)  # F11 down (VK 0x7A, scan 0x57)
    Start-Sleep -Milliseconds 100
    [KB]::keybd_event(0x7A, 0x57, 2, [UIntPtr]::Zero)  # F11 up
}

function Get-ForegroundProcessId {
    $hwnd = [KB]::GetForegroundWindow()
    if ($hwnd -eq [IntPtr]::Zero) { return 0 }
    $foregroundPid = [uint32]0
    [void][KB]::GetWindowThreadProcessId($hwnd, [ref]$foregroundPid)
    return [int]$foregroundPid
}

function Focus-ProcessWindow($proc) {
    $deadline = (Get-Date).AddSeconds(10)
    do {
        $proc.Refresh()
        if ($proc.MainWindowHandle -ne [IntPtr]::Zero) {
            if ([KB]::IsIconic($proc.MainWindowHandle)) {
                [void][KB]::ShowWindowAsync($proc.MainWindowHandle, 9) # SW_RESTORE
            }
            [void][KB]::SetForegroundWindow($proc.MainWindowHandle)
            Start-Sleep -Milliseconds 300
            if ((Get-ForegroundProcessId) -eq $proc.Id) { return $true }
        }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)
    return $false
}

function Send-CaptureKeyToProcess($proc) {
    if (-not (Focus-ProcessWindow $proc)) {
        Write-Error "Refusing to send F11: target process '$($proc.ProcessName)' did not become the foreground window."
        return $false
    }
    Send-F11
    return $true
}

function Get-LaunchedProcess {
    $escapedExe = $Exe.Replace('\', '\\')
    $cim = Get-CimInstance Win32_Process -Filter "ExecutablePath = '$escapedExe'" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cim) {
        return Get-Process -Id $cim.ProcessId -ErrorAction SilentlyContinue
    }
    return Get-Process $processName -ErrorAction SilentlyContinue | Select-Object -First 1
}

# One pixtool session: arm hotkey, launch, wait for game exit, then dump captures.
$launchTime = Get-Date
$pixArgs = "set-gpu-capture-parameters --capture-key=F11 launch `"$Exe`" programmatic-capture --until-exit save-all-captures `"$staging`""
$pix = Start-Process $PixTool -ArgumentList $pixArgs -WorkingDirectory $WorkDir -PassThru -NoNewWindow -RedirectStandardOutput $pixLog
Write-Host "pixtool launched player under capture; capturing at t+${Seconds}s ..."

Start-Sleep -Seconds $Seconds
$game = Get-LaunchedProcess
if (-not $game) {
    Write-Error "Player is not running at the ${Seconds}s mark; see $pixLog"
    Stop-Process -Id $pix.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
    exit 1
}
if (-not (Send-CaptureKeyToProcess $game)) {
    Stop-Process -Id $pix.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "Capture key sent at $(Get-Date -Format HH:mm:ss.fff)."

# The capture layer writes the .wpix to %TEMP%; wait for it to appear, retry the key once.
function Get-NewCapture { Get-ChildItem "$env:TEMP\*.wpix" -ErrorAction SilentlyContinue | Where-Object { $_.CreationTime -gt $launchTime } | Sort-Object CreationTime -Descending }
$deadline = (Get-Date).AddSeconds(15)
while (-not (Get-NewCapture) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
if (-not (Get-NewCapture)) {
    Write-Host "No capture appeared after 15s; re-sending capture key ..."
    if (-not (Send-CaptureKeyToProcess $game)) {
        Stop-Process -Id $pix.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
        exit 1
    }
    $deadline = (Get-Date).AddSeconds(15)
    while (-not (Get-NewCapture) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
}
$cap = Get-NewCapture | Select-Object -First 1
if (-not $cap) {
    Write-Error "Capture was never taken; see $pixLog"
    Get-Process $processName -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
    exit 1
}

# Wait until the capture file stops growing (serialization done), then close the game.
Write-Host "Capture file: $($cap.Name); waiting for serialization ..."
do {
    $size = (Get-Item $cap.FullName).Length
    Start-Sleep -Seconds 3
} while ((Get-Item $cap.FullName).Length -ne $size)

Get-Process $processName -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
if (-not $pix.WaitForExit(300000)) {
    Write-Warning "pixtool did not exit after the game closed; using temp capture fallback. See $pixLog"
    Stop-Process -Id $pix.Id -Force -Confirm:$false -ErrorAction SilentlyContinue
}

$saved = Get-ChildItem "$staging\*.wpix" | Sort-Object CreationTime | Select-Object -Last 1
if ($saved) {
    Move-Item $saved.FullName $Out -Force
} else {
    Write-Warning "pixtool saved no capture to staging; moving temp capture directly. See $pixLog"
    Move-Item $cap.FullName $Out -Force
}
Write-Host "Saved $([math]::Round((Get-Item $Out).Length / 1MB)) MB capture to $Out"
