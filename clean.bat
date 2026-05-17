@echo off
setlocal

set BUILD_DIR=RenderingPlugin\_Build

for %%D in ("%BUILD_DIR%" "RenderingPlugin\_deps" "RenderingPlugin\_ExternalBuild") do (
    if exist "%%~D" (
        echo Cleaning %%~D ...
        rmdir /s /q "%%~D"
        echo [OK] Done.
    ) else (
        echo [INFO] %%~D not found, skipping.
    )
)
