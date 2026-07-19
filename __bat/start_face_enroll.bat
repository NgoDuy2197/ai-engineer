@echo off
REM ==== Thu mat tu webcam. Vi du: start_face_enroll.bat --name "Nguyen Van A" ====
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
set "SVC=services\face_recognition"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_face_recognition.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"

REM Co tham so (chay tu CLI) thi dung luon; khong co (mo tu Dashboard) thi hoi ten.
if not "%~1"=="" goto run_args

set /p NAME="Nhap ten nguoi can thu mat: "
if "%NAME%"=="" (
  echo [!] Chua nhap ten. Thoat.
  pause & exit /b 1
)
echo [*] Thu mat cho "%NAME%" (SPACE:chup  A:tu dong  Q:thoat)...
python "%SVC%\main.py" enroll --name "%NAME%"
goto end

:run_args
echo [*] Thu mat (SPACE:chup  A:tu dong  Q:thoat)...
python "%SVC%\main.py" enroll %*

:end
pause
