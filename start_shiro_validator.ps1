$ErrorActionPreference = "Stop"

$root = "C:\Users\smara\OneDrive\Desktop\Shiro_Validator"

Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "cd '$root'; $env:PYTHONPATH='src'; uvicorn shiro.validator.conjunction_api:app --host 127.0.0.1 --port 8000 --reload"
)

Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "cd '$root\web'; npm run dev -- --host 127.0.0.1 --port 5173"
)

Write-Host "Backend:  http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
