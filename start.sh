#!/bin/bash
# Quick start script for PoUW CAPTCHA (Linux/Mac)

echo "========================================"
echo "PoUW CAPTCHA Quick Start"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "server/venv" ]; then
    echo "Creating Python virtual environment..."
    cd server
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment and install dependencies
echo "Installing Python dependencies..."
source server/venv/bin/activate
cd server
pip install fastapi uvicorn sqlalchemy pydantic pydantic-settings python-jose passlib redis httpx python-dotenv >/dev/null 2>&1
cd ..

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Update .env to use SQLite for simplicity
echo "Updating configuration for local development..."
sed -i '' 's|DATABASE_URL=.*|DATABASE_URL=sqlite:///./pouw_captcha.db|g' .env 2>/dev/null || sed -i 's|DATABASE_URL=.*|DATABASE_URL=sqlite:///./pouw_captcha.db|g' .env
sed -i '' 's|REDIS_URL=.*|REDIS_URL=redis://localhost:6379/0|g' .env 2>/dev/null || sed -i 's|REDIS_URL=.*|REDIS_URL=redis://localhost:6379/0|g' .env

echo ""
echo "========================================"
echo "Starting PoUW CAPTCHA Server"
echo "========================================"
echo ""
echo "Server will start at: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Demo: Open demo/frontend/index.html in browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
cd server
source venv/bin/activate
python -c "from app.models.base import Base; from sqlalchemy import create_engine; engine = create_engine('sqlite:///./pouw_captcha.db'); Base.metadata.create_all(engine); print('Database initialized')"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
