@echo off
REM ==== Bat game Angel Wings ====
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
set "SVC=services\angel_wings"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_angel_wings.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Angel Wings (F:vay canh  E:hieu ung  +/-:co canh  Q:thoat)...
python "%SVC%\main.py" %*
pause
