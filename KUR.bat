@echo off
chcp 65001 >nul
title Anomali ve Fraud Tespit Sistemi - Kurulum
echo.
echo Kurulum baslatiliyor, lutfen bekleyin...
echo (Bu pencere kapanmadan once kurulumu tamamlayacak.)
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0kurulum.ps1"
echo.
echo Devam etmek icin bir tusa basin...
pause >nul
