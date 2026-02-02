@echo off
echo Starting PoUW CAPTCHA Demo...
echo.

echo [1/2] Starting FastAPI Backend on http://localhost:8000
echo       API Docs: http://localhost:8000/docs
start "PoUW Backend" cmd /k "cd /d F:\Projects\antigravity\Capstone\server && F:\programfiles\python\python.exe -m uvicorn app.main:app --reload --port 8000"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Frontend Demo on http://localhost:3000
start "PoUW Frontend" cmd /k "cd /d F:\Projects\antigravity\Capstone\demo\frontend && python -m http.server 3000"

echo.
echo ============================================
echo   Demo is running!
echo   Backend: http://localhost:8000
echo   API Docs: http://localhost:8000/docs  
echo   Frontend: http://localhost:3000
echo ============================================
echo.
echo Press any key to open the demo in your browser...
pause > nul
start http://localhost:3000
