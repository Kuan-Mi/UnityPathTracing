@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Download Bistro Scene Data (via Packman)
echo ============================================================
echo.

set PACKMAN_DIR=%~dp0RenderingPlugin\External\NRD-Sample\External\Packman
set DEPS_XML=%~dp0RenderingPlugin\External\NRD-Sample\Dependencies.xml
set DATA_DIR=%~dp0RenderingPlugin\External\NRD-Sample\_Data
set DEST_DIR=%~dp0UnityProject\Assets\Gltf

echo [1/3] Running Packman to download nri_data package...
call "%PACKMAN_DIR%\packman.cmd" pull "%DEPS_XML%" -p windows-x86_64 -t nri_data_version=2.3
if errorlevel 1 (
    echo [ERROR] Packman download failed.
    exit /b 1
)
echo.

set BISTRO_DIR=%DATA_DIR%\Scenes\Bistro

if not exist "%BISTRO_DIR%" (
    echo [ERROR] Bistro scene directory not found: %BISTRO_DIR%
    exit /b 1
)

echo [2/3] Copying BistroInterior.bin and BistroExterior.bin to %DEST_DIR% ...
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
copy /Y "%BISTRO_DIR%\BistroInterior.bin" "%DEST_DIR%\BistroInterior.bin"
if errorlevel 1 (
    echo [ERROR] Failed to copy BistroInterior.bin.
    exit /b 1
)
copy /Y "%BISTRO_DIR%\BistroExterior.bin" "%DEST_DIR%\BistroExterior.bin"
if errorlevel 1 (
    echo [ERROR] Failed to copy BistroExterior.bin.
    exit /b 1
)
echo.

echo [3/3] Copying Textures folder to %DEST_DIR%\Textures ...
robocopy "%BISTRO_DIR%\Textures" "%DEST_DIR%\Textures" /E /NFL /NDL /NJH /NJS /NC /NS
if errorlevel 1 (
    if errorlevel 8 (
        echo [ERROR] robocopy failed with critical error.
        exit /b 1
    )
)
echo.

echo [4/4] Fixing small BC4/BC5 DDS textures...
powershell -ExecutionPolicy Bypass -File "%~dp0fix_dds_small_bc.ps1"
if errorlevel 1 (
    echo [ERROR] fix_dds_small_bc.ps1 failed.
    exit /b 1
)
echo.

echo Done. Files copied to:
echo   %DEST_DIR%\BistroInterior.bin
echo   %DEST_DIR%\Textures\
echo.
endlocal
