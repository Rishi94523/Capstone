@echo off
REM Quick start script for PoUW CAPTCHA (Windows)

echo ========================================
echo PoUW CAPTCHA Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "server\venv" (
    echo Creating Python virtual environment...
    cd server
    python -m venv venv
    cd ..
)

REM Activate virtual environment and install dependencies
echo Installing Python dependencies...
call server\venv\Scripts\activate.bat
cd server
pip install fastapi uvicorn sqlalchemy pydantic pydantic-settings python-jose passlib redis httpx python-dotenv 2>nul
cd ..

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file...
    copy .env.example .env
)

REM Update .env to use SQLite for simplicity
echo Updating configuration for local development...
powershell -Command "(gc .env) -replace 'DATABASE_URL=.*', 'DATABASE_URL=sqlite:///./pouw_captcha.db' | Out-File -encoding ASCII .env"
powershell -Command "(gc .env) -replace 'REDIS_URL=.*', 'REDIS_URL=redis://localhost:6379/0' | Out-File -encoding ASCII .env"

echo.
echo ========================================
echo Starting PoUW CAPTCHA Server
echo ========================================
echo.
echo Server will start at: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Demo: Open demo/frontend/index.html in browser
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server
cd server
call venv\Scripts\activate.bat
python -c "from app.models.base import Base; from sqlalchemy import create_engine; engine = create_engine('sqlite:///./pouw_captcha.db'); Base.metadata.create_all(engine); print('Database initialized')"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
