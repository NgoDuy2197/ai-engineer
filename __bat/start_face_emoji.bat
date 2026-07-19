@echo off
REM ==== Bat game Face Emoji Party ====
setlocal
cd /d "%~dp0.."
set "SVC=services\face_emoji"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_face_emoji.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Face Emoji Party (R:doi het  M:tieng  +/-:nguong mom  Q:thoat)...
python "%SVC%\main.py" %*
pause
