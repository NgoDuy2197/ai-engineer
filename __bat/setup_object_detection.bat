@echo off
REM ==== Cai dat moi truong ao cho Object Detection (chay 1 lan) ====
setlocal
cd /d "%~dp0.."
set "SVC=services\object_detection"

echo [*] Tao moi truong ao...
python -m venv "%SVC%\.venv"
if errorlevel 1 (
  echo [!] Khong tao duoc venv. Kiem tra Python co trong PATH khong.
  pause & exit /b 1
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Cap nhat pip...
python -m pip install --upgrade pip
echo [*] Cai thu vien (ultralytics, openvino, opencv)...
pip install -r "%SVC%\requirements.txt"

echo.
echo [OK] Da cai xong Object Detection.
echo     Chay: __bat\start_object_detection.bat
pause
