<#
.SYNOPSIS
Builds the Unity player in batch mode via Assets\Editor\CommandLineBuild.cs.

.DESCRIPTION
PowerShell port of build_player.bat. Reads the editor version from
ProjectVersion.txt, then invokes Unity in batch mode. The Unity Editor must be
closed (the project can only be open in one editor at a time).

.PARAMETER BuildPath
Output exe path (relative to the Unity project, or absolute).
Default: Build\UnityPathTracing.exe

.EXAMPLE
powershell -File build_player.ps1
#>
[CmdletBinding()]
param(
    [string]$BuildPath = 'Build/UnityPathTracing.exe'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$project = Join-Path $root 'UnityProject'
$log = Join-Path $env:TEMP 'unity_player_build.log'

# Read the editor version from ProjectVersion.txt (first m_EditorVersion line).
$versionFile = Join-Path $project 'ProjectSettings\ProjectVersion.txt'
$unityVersion = (Select-String -Path $versionFile -Pattern '^m_EditorVersion:\s*(\S+)' |
                 Select-Object -First 1).Matches.Groups[1].Value
if (-not $unityVersion) { Write-Error "[ERROR] Could not read m_EditorVersion from $versionFile."; exit 1 }

$unityExe = "C:\Program Files\Unity\Hub\Editor\$unityVersion\Editor\Unity.exe"
if (-not (Test-Path $unityExe)) {
    Write-Error "[ERROR] Unity $unityVersion not found at `"$unityExe`"."
    exit 1
}

Write-Host '============================================================'
Write-Host " Unity Player Build ($unityVersion)"
Write-Host " Output: $BuildPath"
Write-Host " Log:    $log"
Write-Host '============================================================'
Write-Host ''

# Unity batchmode is unreliable about its process exit code: it frequently
# returns non-zero on quit (leak reporting / shutdown path) even after a fully
# successful build. Trust the build report in the log instead.
& $unityExe -batchmode -quit -projectPath $project `
    -executeMethod CommandLineBuild.BuildWindows -buildPath $BuildPath -logFile $log
$unityExit = $LASTEXITCODE

$result = Select-String -Path $log -Pattern 'Build result:\s*(\w+)' | Select-Object -First 1
if (-not $result -or $result.Matches.Groups[1].Value -ne 'Succeeded') {
    Write-Error "[ERROR] Build failed (Unity exit $unityExit). Last log lines:"
    Get-Content $log -Tail 30
    exit 1
}

Write-Host '[OK] Build succeeded.'
$result.Line.Trim()
if ($unityExit -ne 0) { Write-Host "(Unity returned exit code '$unityExit' on quit; ignored - build report says Succeeded.)" }
