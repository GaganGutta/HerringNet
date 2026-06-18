@echo off
REM Drag a folder of images onto this file, or double-click and paste a path.
REM It runs fish detection on every image and saves annotated copies + a
REM counts CSV into the outputs folder, then opens the results.

setlocal
cd /d "%~dp0"

if "%~1"=="" (
    set /p "folder=Drag your image folder here and press Enter: "
) else (
    set "folder=%~1"
)

REM Strip surrounding quotes if present.
set "folder=%folder:"=%"

if "%folder%"=="" (
    echo No folder given.
    pause
    exit /b 1
)

echo.
echo Running HerringNet on: %folder%
echo.

python -m cli.main batch "%folder%" --open

echo.
pause
