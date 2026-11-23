@echo off
rem Create local .venv from the wheels folder - no internet required.
rem Wrapper for setup_offline.ps1 (bypasses ExecutionPolicy restrictions).
rem Options are passed through, e.g.:  setup_offline.cmd -Force

setlocal
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%setup_offline.ps1" %*
exit /b %ERRORLEVEL%
