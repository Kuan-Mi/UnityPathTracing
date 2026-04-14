@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  RenderingPlugin Build
echo ============================================================
echo.

:: Initialize / update git submodules (NRD + NRI)
echo Initializing git submodules...
git submodule update --init --recursive
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] git submodule update failed.
    exit /b %ERRORLEVEL%
)
echo [OK] Submodules ready.
echo.

:: Allow overriding config via first argument, e.g.: build.bat Debug
set BUILD_CONFIG=Release
if not "%~1"=="" set BUILD_CONFIG=%~1

:: Configure
if not exist "_Build" mkdir "_Build"
cd "_Build"

echo Configuring CMake ^(config: %BUILD_CONFIG%^)...
cmake .. -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed.
    cd ..
    exit /b %ERRORLEVEL%
)

:: Build
echo.
echo Building ^(%BUILD_CONFIG%^)...
cmake --build . --config %BUILD_CONFIG% -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    cd ..
    exit /b %ERRORLEVEL%
)

cd ..

echo.
echo ============================================================
echo  Build successful!  [%BUILD_CONFIG%]
echo  DLLs copied to: ..\UnityProject\Assets\Plugins\x86_64\
echo ============================================================
echo.
exit /b 0
