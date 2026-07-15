@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Camera AI Playground - Server
cd /d "%~dp0"

set "PORT=8000"
set "URL=http://localhost:%PORT%/index.html"

echo ============================================================
echo    CAMERA AI PLAYGROUND - Khoi dong server cuc bo
echo ============================================================
echo.
echo [i] Thu muc: %CD%
echo [i] Cong:    %PORT%
echo.

REM ---------- 1) Uu tien Python (http.server co san, khong can cai them) ----------
echo [*] Dang kiem tra Python...
where python >nul 2>nul
if %errorlevel%==0 (
    set "PYCMD=python"
    goto :run_python
)
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYCMD=py"
    goto :run_python
)
echo     - Khong tim thay Python.
echo.

REM ---------- 2) Fallback: Node.js (tu cai http-server qua npx neu can) ----------
echo [*] Dang kiem tra Node.js...
where node >nul 2>nul
if %errorlevel%==0 (
    where npx >nul 2>nul
    if !errorlevel!==0 goto :run_node
)
echo     - Khong tim thay Node.js/npx.
echo.

REM ---------- 3) Khong co moi truong nao ----------
echo ============================================================
echo [X] CHUA CO MOI TRUONG DE CHAY SERVER.
echo.
echo     Hay cai MOT trong hai:
echo       - Python 3:  https://www.python.org/downloads/
echo                    (nho tick "Add python.exe to PATH")
echo       - Node.js :  https://nodejs.org/
echo.
echo     Sau do chay lai file start.bat nay.
echo ============================================================
echo.
pause
exit /b 1

REM ============================================================
:run_python
echo [OK] Dung Python (%PYCMD%). Khong can cai them thu vien.
echo [*] Mo trinh duyet sau 2 giay tai: %URL%
start "" /b cmd /c "ping -n 3 127.0.0.1 >nul & start "" "%URL%""
echo.
echo ------------------------------------------------------------
echo   Server dang chay. DONG cua so nay de tat server.
echo ------------------------------------------------------------
echo.
%PYCMD% -m http.server %PORT%
goto :end

REM ============================================================
:run_node
echo [OK] Dung Node.js + npx http-server.
echo [*] Lan dau se tu tai goi "http-server" (can mang), sau do chay ngay.
echo [*] Mo trinh duyet sau 3 giay tai: %URL%
start "" /b cmd /c "ping -n 4 127.0.0.1 >nul & start "" "%URL%""
echo.
echo ------------------------------------------------------------
echo   Server dang chay. Nhan Ctrl+C hoac dong cua so de tat.
echo ------------------------------------------------------------
echo.
call npx -y http-server -p %PORT% -c-1
goto :end

:end
echo.
echo [i] Server da dung.
pause
endlocal
