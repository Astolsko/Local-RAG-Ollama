@echo off
title Stop RAG LLM Application
echo Stopping RAG-LLM background services...
powershell -ExecutionPolicy Bypass -File "%~dp0stop-app.ps1"
echo.
echo Application stopped.
timeout /t 3
