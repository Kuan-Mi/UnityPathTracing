@echo off
setlocal

:: Build configuration: pass as first argument (Debug / Release). Default: Debug.
set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=Debug
set OUT_DIR=RenderingPlugin\_Build\%CONFIG%

:: NGX snippet flavor: 'dev' pairs with Debug, 'rel' with Release.
set NGX_FLAVOR=rel
if /I "%CONFIG%"=="Debug" set NGX_FLAVOR=dev

:: NRD/NRI are built externally; fall back to their Debug output if the
:: requested configuration was never built there.
set NRD_DIR=RenderingPlugin\_ExternalBuild\NRD_build\%CONFIG%
set NRI_DIR=RenderingPlugin\_ExternalBuild\NRI_build\%CONFIG%

echo ============================================================
echo  NativeRenderPlugin Build and Copy (%CONFIG%)
echo ============================================================
echo.

:: Create output directories
set UNITY_PLUGINS=UnityProject\Packages\top.kuanmi.native-rendering\Plugins\x86_64
set UNITY_ASSETS_PLUGINS=UnityProject\Assets\Plugins\x86_64
mkdir "%UNITY_PLUGINS%" 2>nul
mkdir "%UNITY_ASSETS_PLUGINS%" 2>nul

:: Check if the DLL is locked by Unity
set DLL_PATH=%UNITY_PLUGINS%\NativeRenderPlugin.dll
if exist "%DLL_PATH%" (
    (2>nul (>> "%DLL_PATH%" echo off)) || (
        echo [WARN] NativeRenderPlugin.dll is currently locked by Unity.
        echo        Please close Unity Editor before building, then run this script again.
        echo.
        pause
        exit /b 1
    )
)

:: Build
call build.bat %CONFIG%
if errorlevel 1 (
    pause
    exit /b 1
)

:: Copy DLLs to Unity
echo Copying DLLs to Unity...

:: Project build outputs (PDB copies fail harmlessly for configs that emit none)
copy /Y "%OUT_DIR%\NativeRenderPlugin.dll"    "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\NativeRenderPlugin.pdb"    "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\OMMBakerPlugin.dll"        "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\OMMBakerPlugin.pdb"        "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\ShaderCompilerPlugin.dll"  "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\ShaderCompilerPlugin.pdb"  "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\D3D12HeapHook.dll"         "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\D3D12HeapHook.pdb"         "%UNITY_PLUGINS%\" >nul

:: SwapChainHookPlugin — standalone diagnostic DLL (DLSS-FG present-hook probe).
:: Disabled by default in Unity; enable "Load on startup" on this DLL to test.
copy /Y "%OUT_DIR%\SwapChainHookPlugin.dll"   "%UNITY_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\SwapChainHookPlugin.pdb"   "%UNITY_PLUGINS%\" >nul

:: DLSS-G frame-generation model — required by SwapChainHookPlugin's DlssgProbe.
:: Must sit beside SwapChainHookPlugin.dll so NGX can locate it.
copy /Y "RenderingPlugin\_deps\ngx-src\lib\Windows_x86_64\%NGX_FLAVOR%\nvngx_dlssg.dll" "%UNITY_PLUGINS%\" >nul

:: DXC (dxcompiler / dxil)
copy /Y "RenderingPlugin\_deps\dxc-nuget\build\native\bin\x64\dxcompiler.dll" "%UNITY_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_deps\dxc-nuget\build\native\bin\x64\dxil.dll"       "%UNITY_PLUGINS%\" >nul

:: OMM lib
copy /Y "RenderingPlugin\_deps\omm-src\bin\omm-lib.dll" "%UNITY_PLUGINS%\" >nul

:: Denoiser / PrepareLight -> Assets\Plugins\x86_64
copy /Y "%OUT_DIR%\Denoiser.dll"              "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\Denoiser.pdb"              "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\PrepareLight.dll"          "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\PrepareLight.pdb"          "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\D3D12HeapHook.dll"         "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%OUT_DIR%\D3D12HeapHook.pdb"         "%UNITY_ASSETS_PLUGINS%\" >nul

:: NRD / NRI -> Assets\Plugins\x86_64
copy /Y "%NRD_DIR%\NRD.dll" "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%NRD_DIR%\NRD.pdb" "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%NRI_DIR%\NRI.dll" "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%NRI_DIR%\NRI.pdb" "%UNITY_ASSETS_PLUGINS%\" >nul

:: DLSS / DLSS-D -> Assets\Plugins\x86_64
copy /Y "%NRI_DIR%\nvngx_dlss.dll"  "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "%NRI_DIR%\nvngx_dlssd.dll" "%UNITY_ASSETS_PLUGINS%\" >nul

echo.
echo ============================================================
echo  Build and copy successful!
echo  Package Plugins: %UNITY_PLUGINS%\
echo    NativeRenderPlugin.dll
echo    OMMBakerPlugin.dll
echo    omm-lib.dll
echo    ShaderCompilerPlugin.dll
echo    D3D12HeapHook.dll
echo    SwapChainHookPlugin.dll  ^(diagnostic; enable "Load on startup" to test^)
echo    nvngx_dlssg.dll          ^(%NGX_FLAVOR%; DLSS-G model for the probe^)
echo    dxcompiler.dll
echo    dxil.dll
echo  Assets Plugins:  %UNITY_ASSETS_PLUGINS%\
echo    Denoiser.dll
echo    PrepareLight.dll
echo    D3D12HeapHook.dll
echo    NRD.dll
echo    NRI.dll
echo    nvngx_dlss.dll
echo    nvngx_dlssd.dll
echo ============================================================
echo.
pause
