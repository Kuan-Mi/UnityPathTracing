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
:: Pass --reconfigure as second argument to force a full CMake reconfigure
set BUILD_CONFIG=Release
if not "%~1"=="" set BUILD_CONFIG=%~1

set BUILD_DIR=RenderingPlugin\_Build
set FORCE_RECONFIGURE=0
if /i "%~2"=="--reconfigure" set FORCE_RECONFIGURE=1

:: Configure only when necessary (first run, or --reconfigure requested)
if not exist "%BUILD_DIR%\CMakeCache.txt" set FORCE_RECONFIGURE=1

if "%FORCE_RECONFIGURE%"=="0" goto :skip_configure

echo Configuring CMake (config: %BUILD_CONFIG%)...
if exist "%BUILD_DIR%\CMakeCache.txt" del /f /q "%BUILD_DIR%\CMakeCache.txt"
cmake -S RenderingPlugin -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed.
    exit /b %ERRORLEVEL%
)
goto :build

:skip_configure
echo [SKIP] CMake already configured. Pass --reconfigure to force.

:build

:: Build
echo.
echo Building (%BUILD_CONFIG%)...
cmake --build "%BUILD_DIR%" --config %BUILD_CONFIG% -j %NUMBER_OF_PROCESSORS%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed.
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo  Build successful!  [%BUILD_CONFIG%]
echo  DLLs copied to: UnityProject\Assets\Plugins\x86_64\
echo ============================================================
echo.
exit /b 0
