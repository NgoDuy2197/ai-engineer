@echo off
REM ==== Bat service People Counter ====
setlocal
cd /d "%~dp0.."
set "SVC=services\people_counter"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_people_counter.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay People Counter (click 2 diem de ve vach, Q/ESC de thoat)...
python "%SVC%\main.py" %*
pause
