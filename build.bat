@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  NativeRenderPlugin Build
echo ============================================================
echo.

:: Initialize / update git submodules (NRD + NRI)
echo Initializing git submodules...
git submodule update --init --recursive
if errorlevel 1 (
    echo [ERROR] git submodule update failed.
    exit /b 1
)
echo [OK] Submodules ready.
echo.

set BUILD_DIR=RenderingPlugin\_Build

:: Configure CMake (only when cache and project files are both present)
if not exist "%BUILD_DIR%\CMakeCache.txt" goto :configure
if not exist "%BUILD_DIR%\ALL_BUILD.vcxproj" goto :configure
goto :build

:configure
echo Configuring CMake...
cmake -S RenderingPlugin -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -T host=x64 -DNR_SKIP_UNITY_COPY=ON
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

:build

echo.
echo Building...
cmake --build "%BUILD_DIR%" --config Debug
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Build successful!
echo  Output: %BUILD_DIR%\Debug\
echo ============================================================
echo.
exit /b 0
