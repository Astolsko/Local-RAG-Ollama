# Start RAG-LLM Application Suite

# 1. Start Redis if not already running
$redisProc = Get-Process "redis-server" -ErrorAction SilentlyContinue
if ($null -eq $redisProc) {
    Write-Host "Redis is not running. Starting Redis server..." -ForegroundColor Yellow
    # Try starting directly or fall back to default path
    try {
        Start-Process "redis-server" -WindowStyle Minimized -ErrorAction Stop
    } catch {
        try {
            Start-Process "C:\Program Files\Redis\redis-server.exe" -WindowStyle Minimized -ErrorAction Stop
        } catch {
            Write-Host "Could not automatically start Redis. Please make sure it is installed and running." -ForegroundColor Red
        }
    }
} else {
    Write-Host "Redis is already running." -ForegroundColor Green
}

# 2. Launch Backend in the background (hidden window)
Write-Host "Launching Backend (FastAPI) on Port 8000 in background..." -ForegroundColor Cyan
Start-Process "conda" -ArgumentList "run", "-n", "RAG", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory "$PSScriptRoot\backend" `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$PSScriptRoot\backend.log" `
    -RedirectStandardError "$PSScriptRoot\backend_err.log"

# 3. Launch Frontend in the background (hidden window)
Write-Host "Launching Frontend (React/Vite) on Port 5173 in background..." -ForegroundColor Cyan
Start-Process "conda" -ArgumentList "run", "-n", "RAG", "npm", "run", "dev" `
    -WorkingDirectory "$PSScriptRoot\frontend" `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$PSScriptRoot\frontend.log" `
    -RedirectStandardError "$PSScriptRoot\frontend_err.log"

# 4. Wait for startup and open browser
Write-Host "Waiting for servers to start..." -ForegroundColor Gray
Start-Sleep -Seconds 4
Write-Host "Opening application in browser..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host "Application is running in the background." -ForegroundColor Green
Write-Host "To stop the application, run stop-app.bat" -ForegroundColor Yellow
