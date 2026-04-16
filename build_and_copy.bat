@echo off
setlocal

echo ============================================================
echo  NativeRenderPlugin Build and Copy
echo ============================================================
echo.

:: Create output directory
set UNITY_PLUGINS=UnityProject\Packages\top.kuanmi.native-ray-tracing\Plugins\x86_64
mkdir "%UNITY_PLUGINS%" 2>nul

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
copy /Y "NativePlugin\build\Debug\NativeRenderPlugin.dll" "%UNITY_PLUGINS%\" >nul
copy /Y "NativePlugin\build\Debug\NativeRenderPlugin.pdb" "%UNITY_PLUGINS%\" >nul
copy /Y "NativePlugin\build\Debug\OMMBakerPlugin.dll"        "%UNITY_PLUGINS%\" >nul
copy /Y "NativePlugin\build\Debug\OMMBakerPlugin.pdb"        "%UNITY_PLUGINS%\" >nul
copy /Y "NativePlugin\build\Debug\ShaderCompilerPlugin.dll"  "%UNITY_PLUGINS%\" >nul
copy /Y "NativePlugin\build\Debug\ShaderCompilerPlugin.pdb"  "%UNITY_PLUGINS%\" >nul

echo.
echo ============================================================
echo  Build and copy successful!
echo  Output: %UNITY_PLUGINS%\NativeRenderPlugin.dll
echo           %UNITY_PLUGINS%\OMMBakerPlugin.dll
echo           %UNITY_PLUGINS%\ShaderCompilerPlugin.dll
echo ============================================================
echo.
pause
