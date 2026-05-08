@echo off
:: start_for_pipeline.bat -- Start DeerFlow gateway on port 8001 for overnight-pipeline.
::
:: Usage:
::   start_for_pipeline.bat          -- start in foreground (blocks until Ctrl+C)
::   start_for_pipeline.bat --daemon -- start in background, return immediately
::
:: The overnight-pipeline calls DeerFlow via http://localhost:8001.
:: Run this script before (or as part of) the pipeline's scheduled task.
::
:: Prerequisites:
::   - uv must be on PATH  (https://docs.astral.sh/uv/getting-started/installation/)
::   - .env must exist in this directory with ANTHROPIC_API_KEY set

setlocal EnableDelayedExpansion

set DEER_FLOW_DIR=%~dp0
set BACKEND_DIR=%DEER_FLOW_DIR%backend
set ENV_FILE=%DEER_FLOW_DIR%.env

:: -- Load .env into environment ---------------------------------------------
if not exist "%ENV_FILE%" (
    echo ERROR: %ENV_FILE% not found. Copy .env.example and fill in your API keys.
    exit /b 1
)

set "_env_tmp=%TEMP%\deer_env_%RANDOM%.cmd"
powershell -NoProfile -Command "$q=[char]34; Get-Content '%ENV_FILE%' | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' } | ForEach-Object { '@set ' + $q + $_ + $q } | Set-Content '%_env_tmp%' -Encoding ASCII"
if exist "%_env_tmp%" (
    call "%_env_tmp%"
    del /q "%_env_tmp%"
)
set "_env_tmp="

:: -- Check ANTHROPIC_API_KEY ------------------------------------------------
if "!ANTHROPIC_API_KEY!"=="" (
    echo ERROR: ANTHROPIC_API_KEY is not set in %ENV_FILE%
    exit /b 1
)

:: -- Check if port 8001 already in use -------------------------------------
netstat -ano | findstr /C:":8001 " >nul 2>&1
if %errorlevel%==0 (
    echo DeerFlow already running on port 8001. Nothing to do.
    exit /b 0
)

:: -- Check uv is available -------------------------------------------------
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: 'uv' not found on PATH. Install from https://docs.astral.sh/uv/
    exit /b 1
)

:: -- Start gateway ---------------------------------------------------------
echo Starting DeerFlow gateway on http://127.0.0.1:8001 ...
cd /d "!BACKEND_DIR!"

if "%1"=="--daemon" (
    echo Running in background (daemon mode).
    start /b "" cmd /c "uv run uvicorn app.gateway.app:app --host 127.0.0.1 --port 8001 >> !DEER_FLOW_DIR!gateway.log 2>&1"
    echo DeerFlow gateway started in background. Logs: !DEER_FLOW_DIR!gateway.log
    exit /b 0
) else (
    uv run uvicorn app.gateway.app:app --host 127.0.0.1 --port 8001
)

endlocal
