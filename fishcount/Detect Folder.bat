@echo off
rem Relaunch inside "cmd /k" so this window ALWAYS stays open, even on errors.
if not "%FISHCOUNT_KEEPOPEN%"=="1" (
    set "FISHCOUNT_KEEPOPEN=1"
    cmd /k ""%~f0" %*"
    exit /b
)
setlocal
cd /d "%~dp0"
> "%~dp0last-run.log" echo [%date% %time%] launched with: %*
if "%~1"=="" (
    echo Drag a folder of images onto this file to detect fish in it.
) else (
    if exist "%~dp0.venv\Scripts\fishcount.exe" (
        "%~dp0.venv\Scripts\fishcount.exe" detect "%~1" --open
    ) else (
        where fishcount >nul 2>nul
        if not errorlevel 1 (
            fishcount detect "%~1" --open
        ) else (
            py -m fishcount detect "%~1" --open
        )
    )
)
echo.
echo Finished. This window stays open; close it when you are done reading.
