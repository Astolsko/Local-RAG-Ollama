# Stop RAG-LLM Application Suite processes running in the background

Write-Host "Stopping backend and frontend processes..." -ForegroundColor Yellow

# Kill Uvicorn (FastAPI backend) processes
$uvicornProcs = Get-Process "uvicorn" -ErrorAction SilentlyContinue
if ($uvicornProcs) {
    Stop-Process -Name "uvicorn" -Force
}

# Kill Node/NPM (Vite frontend dev server) processes
$nodeProcs = Get-Process "node" -ErrorAction SilentlyContinue
if ($nodeProcs) {
    Stop-Process -Name "node" -Force
}

# Kill python instances running in the RAG environment to ensure uvicorn child tasks are closed
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*\envs\RAG\*" } | Stop-Process -Force

Write-Host "Application processes stopped successfully." -ForegroundColor Green
