@echo off
setlocal
if "%~1"=="" (
    echo Drag a folder of images onto this file to count the fish in it.
    pause
    exit /b 1
)
cd /d "%~dp0"
set "VENV_EXE=%~dp0.venv\Scripts\fishcount.exe"
if exist "%VENV_EXE%" (
    "%VENV_EXE%" detect "%~1" --open
) else (
    where fishcount >nul 2>nul
    if not errorlevel 1 (
        fishcount detect "%~1" --open
    ) else (
        py -m fishcount detect "%~1" --open
    )
)
echo.
pause
