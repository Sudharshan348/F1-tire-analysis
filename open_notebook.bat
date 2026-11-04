@echo off
rem ==========================================================================
rem  F1 Tire Degradation - open the already-executed notebook in Jupyter
rem  All outputs and plots are saved inside executed_notebook.ipynb, so
rem  nothing needs to be re-run. Do NOT click "Run All" during the demo
rem  (it retrains every model and takes a long time).
rem  Close this window (or press Ctrl+C twice) to stop Jupyter.
rem ==========================================================================
title Jupyter - executed_notebook.ipynb (close this window to stop)

set "VENV=C:\Users\Arush\OneDrive\Desktop\venv"
set "PYTHONUTF8=1"

cd /d "%~dp0"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Python environment not found at %VENV%
    pause
    exit /b 1
)
if not exist "executed_notebook.ipynb" (
    echo [ERROR] executed_notebook.ipynb not found in %CD%
    pause
    exit /b 1
)
call "%VENV%\Scripts\activate.bat"

rem Mark the notebook as trusted so interactive Plotly outputs render.
jupyter trust executed_notebook.ipynb >nul 2>&1

echo Opening executed_notebook.ipynb in Jupyter (browser opens automatically) ...
echo.
jupyter notebook executed_notebook.ipynb

echo.
echo Jupyter has stopped.
pause
