@echo off
REM ==== Chay camera nhan dien + dan nhan ten ====
setlocal
cd /d "%~dp0.."
set "SVC=services\face_recognition"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_face_recognition.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay camera nhan dien (+/-:nguong  Q:thoat)...
python "%SVC%\main.py" camera %*
pause
