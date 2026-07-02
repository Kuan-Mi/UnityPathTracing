<#
.SYNOPSIS
Builds NativeRenderPlugin and deploys all runtime DLLs into the Unity project.

.DESCRIPTION
PowerShell port of build_and_copy.bat. Runs build.ps1, then copies the plugin
outputs, the Streamline 2.11.1 RELEASE runtime, DXC, OMM and NRD/NRI into the
Unity package and Assets plugin folders.

.PARAMETER Config
Build configuration: Debug (default) or Release. NRD/NRI fall back to their
Debug external build if the requested configuration was never built there.

.EXAMPLE
powershell -File build_and_copy.ps1 Release
#>
[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Config = 'Debug'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$OUT     = Join-Path $root "RenderingPlugin\_Build\$Config"
$NRD_DIR = Join-Path $root "RenderingPlugin\_ExternalBuild\NRD_build\$Config"
$NRI_DIR = Join-Path $root "RenderingPlugin\_ExternalBuild\NRI_build\$Config"
$SL_BIN  = Join-Path $root 'Other\streamline-sdk-v2.11.1\bin\x64'
$DXC     = Join-Path $root 'RenderingPlugin\_deps\dxc-nuget\build\native\bin\x64'
$OMM     = Join-Path $root 'RenderingPlugin\_deps\omm-src\bin'

$UNITY_PLUGINS        = Join-Path $root 'UnityProject\Packages\top.kuanmi.native-rendering\Plugins\x86_64'
$UNITY_ASSETS_PLUGINS = Join-Path $root 'UnityProject\Assets\Plugins\x86_64'

Write-Host '============================================================'
Write-Host " NativeRenderPlugin Build and Copy ($Config)"
Write-Host '============================================================'
Write-Host ''

New-Item -ItemType Directory -Force $UNITY_PLUGINS, $UNITY_ASSETS_PLUGINS | Out-Null

# Refuse to build if Unity holds the deployed DLL locked (it can't be overwritten).
$dllPath = Join-Path $UNITY_PLUGINS 'NativeRenderPlugin.dll'
if (Test-Path $dllPath) {
    try { $fs = [IO.File]::Open($dllPath, 'Open', 'Write', 'None'); $fs.Close() }
    catch {
        Write-Error @'
[WARN] NativeRenderPlugin.dll is currently locked by Unity.
       Close the Unity Editor before building, then run this script again.
'@
        exit 1
    }
}

# Build.
& (Join-Path $root 'build.ps1') $Config
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host 'Copying DLLs to Unity...'

# Copy a file, warning (not failing) on a missing optional PDB.
function Copy-Dll {
    param([string]$Src, [string]$Dst, [switch]$Optional)
    if (Test-Path $Src) { Copy-Item $Src $Dst -Force }
    elseif (-not $Optional) { Write-Warning "missing: $Src" }
}

# --- Project outputs -> package (PDB copies are best-effort) ---
foreach ($n in 'NativeRenderPlugin', 'OMMBakerPlugin', 'ShaderCompilerPlugin', 'D3D12HeapHook', 'StreamlinePlugin') {
    Copy-Dll "$OUT\$n.dll" $UNITY_PLUGINS
    Copy-Dll "$OUT\$n.pdb" $UNITY_PLUGINS -Optional
}

# --- Streamline 2.11.1 RELEASE runtime -> package (DLSS-SR/RR/G) ---
foreach ($n in 'sl.interposer', 'sl.common', 'sl.dlss_g', 'sl.reflex', 'sl.pcl', 'nvngx_dlssg',
                'sl.dlss', 'nvngx_dlss', 'sl.dlss_d', 'nvngx_dlssd') {
    Copy-Dll "$SL_BIN\$n.dll" $UNITY_PLUGINS
}

# --- DXC + OMM -> package ---
Copy-Dll "$DXC\dxcompiler.dll" $UNITY_PLUGINS
Copy-Dll "$DXC\dxil.dll"       $UNITY_PLUGINS
Copy-Dll "$OMM\omm-lib.dll"    $UNITY_PLUGINS

# --- NRIPlugin / PrepareLight / heap hook -> Assets ---
foreach ($n in 'NRIPlugin', 'PrepareLight', 'D3D12HeapHook') {
    Copy-Dll "$OUT\$n.dll" $UNITY_ASSETS_PLUGINS
    Copy-Dll "$OUT\$n.pdb" $UNITY_ASSETS_PLUGINS -Optional
}

# --- NRD / NRI -> Assets ---
Copy-Dll "$NRD_DIR\NRD.dll" $UNITY_ASSETS_PLUGINS
Copy-Dll "$NRD_DIR\NRD.pdb" $UNITY_ASSETS_PLUGINS -Optional
Copy-Dll "$NRI_DIR\NRI.dll" $UNITY_ASSETS_PLUGINS
Copy-Dll "$NRI_DIR\NRI.pdb" $UNITY_ASSETS_PLUGINS -Optional

# --- NGX release models -> Assets (win duplicate-name collision in the player build) ---
Copy-Dll "$SL_BIN\nvngx_dlss.dll"  $UNITY_ASSETS_PLUGINS
Copy-Dll "$SL_BIN\nvngx_dlssd.dll" $UNITY_ASSETS_PLUGINS

Write-Host ''
Write-Host '============================================================'
Write-Host ' Build and copy successful!'
Write-Host "  Package Plugins: $UNITY_PLUGINS"
Write-Host "  Assets Plugins:  $UNITY_ASSETS_PLUGINS"
Write-Host '============================================================'
Write-Host ''
