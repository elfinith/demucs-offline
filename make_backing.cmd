@echo off
rem Backing track (no guitar) from any audio file - fully offline.
rem All arguments are passed through to src\make_backing.py.
rem Examples:
rem   make_backing.cmd "D:\music\song.mp3"
rem   make_backing.cmd "D:\music\*.mp3" -o D:\backing --suppress 0.8
rem   make_backing.cmd --help

setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
if not exist "%PY%" (
    echo Local .venv not found, using system python...
    python -u "%ROOT%src\make_backing.py" %*
    exit /b %ERRORLEVEL%
)
"%PY%" -u "%ROOT%src\make_backing.py" %*
exit /b %ERRORLEVEL%
