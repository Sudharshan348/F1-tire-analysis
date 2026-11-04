@echo off
rem ==========================================================================
rem  F1 Tire Degradation - launch the Streamlit dashboard
rem  Double-click this file. The dashboard opens in your browser at
rem  http://localhost:8501 . Close this window (or press Ctrl+C) to stop it.
rem ==========================================================================
title F1 Dashboard (close this window to stop)

set "VENV=C:\Users\Arush\OneDrive\Desktop\venv"
set "URL=http://localhost:8501"
set "PYTHONUTF8=1"

cd /d "%~dp0"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Python environment not found at %VENV%
    pause
    exit /b 1
)
call "%VENV%\Scripts\activate.bat"

rem If the dashboard is already running, just open the browser.
powershell -NoProfile -Command "try { Invoke-WebRequest '%URL%/_stcore/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo Dashboard is already running - opening %URL%
    start "" "%URL%"
    ping -n 3 127.0.0.1 >nul
    exit /b 0
)

rem Open the browser as soon as the server answers (runs in the background).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest '%URL%/_stcore/health' -UseBasicParsing -TimeoutSec 1 | Out-Null; Start-Process '%URL%'; break } catch { Start-Sleep -Milliseconds 500 } }"

echo Starting dashboard at %URL% ...
echo (The browser opens automatically. Close this window to stop the dashboard.)
echo.
python -m streamlit run app.py --server.port 8501

echo.
echo The dashboard has stopped.
pause
