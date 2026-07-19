@echo off
REM ==== Tao embeddings (db.pkl) tu du lieu da gan nhan ====
setlocal
cd /d "%~dp0.."
set "SVC=services\face_recognition"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_face_recognition.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang train...
python "%SVC%\main.py" train %*
pause
