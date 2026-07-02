@echo off
setlocal

:: Builds the Unity player in batch mode via Assets\Editor\CommandLineBuild.cs.
:: Usage: build_player.bat [output exe path]
::   Default output: UnityProject\Build\UnityPathTracing.exe
:: Unity Editor must be closed (the project can only be open in one editor).

set PROJECT=%~dp0UnityProject
set BUILD_PATH=%~1
if "%BUILD_PATH%"=="" set BUILD_PATH=Build/UnityPathTracing.exe
set LOG=%TEMP%\unity_player_build.log

:: Read the editor version from ProjectVersion.txt (first m_EditorVersion line)
set UNITY_VERSION=
for /f "tokens=2" %%v in ('findstr /B /C:"m_EditorVersion:" "%PROJECT%\ProjectSettings\ProjectVersion.txt"') do (
    if not defined UNITY_VERSION set UNITY_VERSION=%%v
)
set UNITY_EXE=C:\Program Files\Unity\Hub\Editor\%UNITY_VERSION%\Editor\Unity.exe
if not exist "%UNITY_EXE%" (
    echo [ERROR] Unity %UNITY_VERSION% not found at "%UNITY_EXE%".
    exit /b 1
)

echo ============================================================
echo  Unity Player Build (%UNITY_VERSION%)
echo  Output: %BUILD_PATH%
echo  Log:    %LOG%
echo ============================================================
echo.

"%UNITY_EXE%" -batchmode -quit -projectPath "%PROJECT%" -executeMethod CommandLineBuild.BuildWindows -buildPath "%BUILD_PATH%" -logFile "%LOG%"
if errorlevel 1 (
    echo [ERROR] Build failed. Last log lines:
    powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 30"
    exit /b 1
)

echo [OK] Build succeeded.
findstr /C:"Build result:" "%LOG%"
exit /b 0
