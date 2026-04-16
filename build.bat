@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  NativeRenderPlugin Build
echo ============================================================
echo.

:: Check for Unity Plugin API headers
set UNITY_HEADERS=NativePlugin\shared\include\unity\IUnityInterface.h
if not exist "%UNITY_HEADERS%" (
    echo [ERROR] Unity Plugin API headers not found.
    echo.
    echo Please copy the following files from your Unity installation:
    echo   ^<UnityInstall^>\Editor\Data\PluginAPI\IUnityInterface.h
    echo   ^<UnityInstall^>\Editor\Data\PluginAPI\IUnityGraphics.h
    echo   ^<UnityInstall^>\Editor\Data\PluginAPI\IUnityGraphicsD3D12.h
    echo   ^<UnityInstall^>\Editor\Data\PluginAPI\IUnityRenderingExtensions.h
    echo.
    echo To: NativePlugin\shared\include\unity\
    echo.

    :: Try to auto-locate Unity
    set UNITY_COMMON="C:\Program Files\Unity\Hub\Editor"
    if exist !UNITY_COMMON! (
        echo Attempting to auto-copy from latest Unity installation...
        for /f "delims=" %%v in ('dir /b /ad /o-n "!UNITY_COMMON!"') do (
            set UNITY_PLUGIN_API=!UNITY_COMMON!\%%v\Editor\Data\PluginAPI
            if exist "!UNITY_PLUGIN_API!\IUnityInterface.h" (
                echo Found: !UNITY_PLUGIN_API!
                mkdir NativePlugin\shared\include\unity 2>nul
                copy "!UNITY_PLUGIN_API!\IUnityInterface.h"           NativePlugin\shared\include\unity\ >nul
                copy "!UNITY_PLUGIN_API!\IUnityGraphics.h"            NativePlugin\shared\include\unity\ >nul
                copy "!UNITY_PLUGIN_API!\IUnityGraphicsD3D12.h"       NativePlugin\shared\include\unity\ >nul
                copy "!UNITY_PLUGIN_API!\IUnityRenderingExtensions.h" NativePlugin\shared\include\unity\ >nul
                echo Headers copied successfully.
                goto :headers_ok
            )
        )
    )

    echo Could not auto-locate Unity. Please copy headers manually.
    exit /b 1
)

:headers_ok
echo [OK] Unity Plugin API headers found.
echo.

:: Configure CMake
echo Configuring CMake...
cd NativePlugin
if not exist build mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DNR_SKIP_UNITY_COPY=ON
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    cd ..\..
    exit /b 1
)

echo.
echo Building...
cmake --build . --config Debug
if errorlevel 1 (
    echo [ERROR] Build failed.
    cd ..\..
    exit /b 1
)

cd ..\..

echo.
echo ============================================================
echo  Build successful!
echo  Output: NativePlugin\build\Debug\
echo ============================================================
echo.
exit /b 0
