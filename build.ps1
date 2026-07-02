<#
.SYNOPSIS
Builds NativeRenderPlugin (C++/D3D12) via the CMake / VS2022 solution.

.DESCRIPTION
PowerShell port of build.bat. Initializes the required submodules, configures
CMake on first run (or when the generated project is missing), then builds.

.PARAMETER Config
Build configuration: Debug (default) or Release.

.EXAMPLE
powershell -File build.ps1 Release
#>
[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Config = 'Debug'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$buildDir = Join-Path $root 'RenderingPlugin\_Build'

Write-Host '============================================================'
Write-Host " NativeRenderPlugin Build ($Config)"
Write-Host '============================================================'
Write-Host ''

Write-Host 'Initializing git submodules...'
$submodules = @(
    '--recursive RenderingPlugin/External/NRD',
    '--recursive RenderingPlugin/External/NRI',
    'RenderingPlugin/External/donut',
    'RenderingPlugin/External/RTXDI-Library',
    'RenderingPlugin/External/RTXPT',
    'RenderingPlugin/External/NRD-Sample',
    'RenderingPlugin/External/RTXDI',
    'RenderingPlugin/External/NVAPI'
)
foreach ($sm in $submodules) {
    git -C $root submodule update --init $sm.Split(' ')
    if ($LASTEXITCODE -ne 0) { Write-Error '[ERROR] git submodule update failed.'; exit 1 }
}
Write-Host '[OK] Submodules ready.'
Write-Host ''

# Configure CMake only when the cache and the generated project are not both present.
$needConfigure = -not ((Test-Path (Join-Path $buildDir 'CMakeCache.txt')) -and
                       (Test-Path (Join-Path $buildDir 'ALL_BUILD.vcxproj')))
if ($needConfigure) {
    Write-Host 'Configuring CMake...'
    cmake -S (Join-Path $root 'RenderingPlugin') -B $buildDir `
        -G 'Visual Studio 17 2022' -A x64 -T host=x64 -DNR_SKIP_UNITY_COPY=ON
    if ($LASTEXITCODE -ne 0) { Write-Error '[ERROR] CMake configuration failed.'; exit 1 }
}

Write-Host ''
Write-Host 'Building...'
cmake --build $buildDir --config $Config
if ($LASTEXITCODE -ne 0) { Write-Error '[ERROR] Build failed.'; exit 1 }

Write-Host ''
Write-Host '============================================================'
Write-Host ' Build successful!'
Write-Host " Output: RenderingPlugin\_Build\$Config\"
Write-Host '============================================================'
Write-Host ''
