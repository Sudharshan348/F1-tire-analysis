@echo off
rem ==========================================================================
rem  F1 Tire Degradation - run the Differential Evolution pit-lap optimizer
rem  Prints the baseline grid result vs the DE optimum, then waits.
rem  Runs with --no-save so the demo artifacts in artifacts\ are not touched.
rem ==========================================================================
title F1 Pit-Lap Optimizer

set "VENV=C:\Users\Arush\OneDrive\Desktop\venv"
set "PYTHONUTF8=1"

cd /d "%~dp0"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Python environment not found at %VENV%
    pause
    exit /b 1
)
call "%VENV%\Scripts\activate.bat"

echo Running the full-window pit-lap optimizer (about 10 seconds) ...
echo.
python optimizer.py --no-save

echo.
pause
