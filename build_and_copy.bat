@echo off
setlocal

echo ============================================================
echo  NativeRenderPlugin Build and Copy
echo ============================================================
echo.

:: Create output directories
set UNITY_PLUGINS=UnityProject\Packages\top.kuanmi.native-ray-tracing\Plugins\x86_64
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
call build.bat
if errorlevel 1 (
    pause
    exit /b 1
)

:: Copy DLLs to Unity
echo Copying DLLs to Unity...

:: Project build outputs
copy /Y "RenderingPlugin\_Build\Debug\NativeRenderPlugin.dll"    "%UNITY_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\NativeRenderPlugin.pdb"    "%UNITY_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\OMMBakerPlugin.dll"        "%UNITY_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\OMMBakerPlugin.pdb"        "%UNITY_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\ShaderCompilerPlugin.dll"  "%UNITY_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\ShaderCompilerPlugin.pdb"  "%UNITY_PLUGINS%\" >nul

:: Denoiser / PrepareLight -> Assets\Plugins\x86_64
copy /Y "RenderingPlugin\_Build\Debug\Denoiser.dll"              "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\Denoiser.pdb"              "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\PrepareLight.dll"          "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_Build\Debug\PrepareLight.pdb"          "%UNITY_ASSETS_PLUGINS%\" >nul

:: NRD / NRI -> Assets\Plugins\x86_64
copy /Y "RenderingPlugin\_ExternalBuild\NRD_build\Debug\NRD.dll" "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_ExternalBuild\NRD_build\Debug\NRD.pdb" "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_ExternalBuild\NRI_build\Debug\NRI.dll" "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_ExternalBuild\NRI_build\Debug\NRI.pdb" "%UNITY_ASSETS_PLUGINS%\" >nul

:: DLSS / DLSS-D -> Assets\Plugins\x86_64
copy /Y "RenderingPlugin\_ExternalBuild\NRI_build\Debug\nvngx_dlss.dll"  "%UNITY_ASSETS_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_ExternalBuild\NRI_build\Debug\nvngx_dlssd.dll" "%UNITY_ASSETS_PLUGINS%\" >nul


:: DXC (dxcompiler / dxil)
copy /Y "RenderingPlugin\_deps\dxc-nuget\build\native\bin\x64\dxcompiler.dll" "%UNITY_PLUGINS%\" >nul
copy /Y "RenderingPlugin\_deps\dxc-nuget\build\native\bin\x64\dxil.dll"       "%UNITY_PLUGINS%\" >nul

echo.
echo ============================================================
echo  Build and copy successful!
echo  Package Plugins: %UNITY_PLUGINS%\
echo    NativeRenderPlugin.dll
echo    OMMBakerPlugin.dll
echo    ShaderCompilerPlugin.dll
echo    dxcompiler.dll
echo    dxil.dll
echo  Assets Plugins:  %UNITY_ASSETS_PLUGINS%\
echo    Denoiser.dll
echo    PrepareLight.dll
echo    NRD.dll
echo    NRI.dll
echo    nvngx_dlss.dll
echo    nvngx_dlssd.dll
echo ============================================================
echo.
pause
