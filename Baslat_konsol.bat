@echo off
chcp 65001 >nul
title Anomali ve Fraud Tespit Sistemi
cd /d "%~dp0app"
echo Uygulama baslatiliyor... (sorun olursa hatalar bu pencerede gorunur)
"%~dp0.venv\Scripts\python.exe" "%~dp0app\main.py"
echo.
echo Uygulama kapandi. Cikis kodu: %ERRORLEVEL%
pause
