@echo off
REM ==== Bat service Object Detection ====
setlocal
cd /d "%~dp0.."
set "SVC=services\object_detection"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_object_detection.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Object Detection (Q/ESC de thoat)...
python "%SVC%\main.py" %*
pause
