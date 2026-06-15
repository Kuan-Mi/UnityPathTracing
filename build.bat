@echo off
setlocal enabledelayedexpansion

:: Build configuration: pass as first argument (Debug / Release). Default: Debug.
set CONFIG=%~1
if "%CONFIG%"=="" set CONFIG=Debug

echo ============================================================
echo  NativeRenderPlugin Build (%CONFIG%)
echo ============================================================
echo.

echo Initializing git submodules...
git submodule update --init --recursive RenderingPlugin/External/NRD
git submodule update --init --recursive RenderingPlugin/External/NRI
git submodule update --init RenderingPlugin/External/donut
git submodule update --init RenderingPlugin/External/RTXDI-Library
git submodule update --init RenderingPlugin/External/RTXPT
git submodule update --init RenderingPlugin/External/NRD-Sample
git submodule update --init RenderingPlugin/External/RTXDI
git submodule update --init RenderingPlugin/External/NVAPI
if errorlevel 1 (
    echo [ERROR] git submodule update failed.
    exit /b 1
)
echo [OK] Submodules ready.
echo.

set BUILD_DIR=RenderingPlugin\_Build

:: Detect the installed Visual Studio version via vswhere (ships with VS 2017+).
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist %VSWHERE% (
    echo [ERROR] vswhere.exe not found. Is Visual Studio 2017 or newer installed?
    exit /b 1
)

set VS_MAJOR=
for /f "tokens=1 delims=." %%v in ('%VSWHERE% -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion') do set VS_MAJOR=%%v

if "%VS_MAJOR%"=="17" set CMAKE_GENERATOR=Visual Studio 17 2022
if "%VS_MAJOR%"=="16" set CMAKE_GENERATOR=Visual Studio 16 2019
if "%VS_MAJOR%"=="15" set CMAKE_GENERATOR=Visual Studio 15 2017

if "%CMAKE_GENERATOR%"=="" (
    echo [ERROR] No supported Visual Studio with C++ tools found ^(detected version: "%VS_MAJOR%"^).
    exit /b 1
)
echo [OK] Using generator: %CMAKE_GENERATOR%
echo.

:: Configure CMake (only when cache and project files are both present)
if not exist "%BUILD_DIR%\CMakeCache.txt" goto :configure
if not exist "%BUILD_DIR%\ALL_BUILD.vcxproj" goto :configure
goto :build

:configure
echo Configuring CMake...
cmake -S RenderingPlugin -B "%BUILD_DIR%" -G "%CMAKE_GENERATOR%" -A x64 -T host=x64 -DNR_SKIP_UNITY_COPY=ON
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

:build

echo.
echo Building...
cmake --build "%BUILD_DIR%" --config %CONFIG%
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Build successful!
echo  Output: %BUILD_DIR%\%CONFIG%\
echo ============================================================
echo.
exit /b 0
