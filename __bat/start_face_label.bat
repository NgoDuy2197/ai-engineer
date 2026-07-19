@echo off
REM ==== Mo man hinh gan nhan anh trong __data\_inbox ====
setlocal
cd /d "%~dp0.."
set "SVC=services\face_recognition"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_face_recognition.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Mo man hinh gan nhan...
python "%SVC%\main.py" label %*
pause
