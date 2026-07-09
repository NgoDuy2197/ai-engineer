@echo off
REM ==== Bat service Smart Security Cam ====
setlocal
cd /d "%~dp0.."
set "SVC=services\security_cam"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_security_cam.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Smart Security Cam (M:tieng  S:chup  C:reset  Q:thoat)...
python "%SVC%\main.py" %*
pause
