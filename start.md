# Start Guide (Windows)

This file contains the exact steps to run backend and frontend locally.

## Backend

Run in `cmd.exe`:

```cmd
cd /d "D:\Major Project\backend"
.\venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Frontend

Run in `cmd.exe`:

```cmd
cd /d "D:\Major Project\frontend"
npm run dev -- --port 3000
```

## PowerShell Note

If you want to use PowerShell and `npm` is blocked, enable scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Quick Checks

```cmd
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/trading/watchlist
```

## Demo Mode (No Postgres/Redis)

Demo mode is enabled by default in `backend/app/config.py`.

To force it in `.env`:

```env
DEMO_MODE=True
```
