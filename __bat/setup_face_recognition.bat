@echo off
REM ==== Cai dat moi truong ao cho Face Recognition (chay 1 lan) ====
setlocal
cd /d "%~dp0.."
set "SVC=services\face_recognition"

echo [*] Tao moi truong ao...
python -m venv "%SVC%\.venv"
if errorlevel 1 (
  echo [!] Khong tao duoc venv. Kiem tra Python co trong PATH khong.
  pause & exit /b 1
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Cap nhat pip...
python -m pip install --upgrade pip
echo [*] Cai thu vien (opencv-contrib, numpy, Pillow)...
pip install -r "%SVC%\requirements.txt"

echo.
echo [OK] Da cai xong Face Recognition.
echo     1) Tao du lieu : __bat\start_face_enroll.bat --name "Ten Nguoi"
echo        hoac gan nhan: __bat\start_face_label.bat
echo     2) Train        : __bat\start_face_train.bat
echo     3) Chay camera  : __bat\start_face_camera.bat
pause
