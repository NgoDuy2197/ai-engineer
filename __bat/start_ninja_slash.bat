@echo off
REM ==== Bat service Ninja Slash ====
setlocal
cd /d "%~dp0.."
set "SVC=services\ninja_slash"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_ninja_slash.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Ninja Slash (vung ngon tro de chem, C:choi lai, Q:thoat)...
python "%SVC%\main.py" %*
pause
