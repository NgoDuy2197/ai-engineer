@echo off
REM ==== Bat service Fitness Coach ====
setlocal
cd /d "%~dp0.."
set "SVC=services\fitness_coach"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_fitness_coach.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Fitness Coach (E:doi bai  C:reset  Q:thoat)...
python "%SVC%\main.py" %*
pause
