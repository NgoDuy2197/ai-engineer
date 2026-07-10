@echo off
REM ==== Bat service Space Dodge ====
setlocal
cd /d "%~dp0.."
set "SVC=services\space_dodge"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_space_dodge.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Space Dodge (nghieng dau de ne, C:choi lai, Q:thoat)...
python "%SVC%\main.py" %*
pause
