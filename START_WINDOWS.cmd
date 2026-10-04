@echo off
cd /d "%~dp0"
py -3.13 app.py
if errorlevel 1 (
  echo.
  echo If Python was not found, open this folder in VS Code and run: python app.py
  pause
)
