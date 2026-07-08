@echo off
REM ==== Cai dat moi truong ao cho Hand Gesture Controller (chay 1 lan) ====
setlocal
cd /d "%~dp0.."
set "SVC=services\hand_gesture_controller"

echo [*] Tao moi truong ao...
python -m venv "%SVC%\.venv"
if errorlevel 1 (
  echo [!] Khong tao duoc venv. Kiem tra Python co trong PATH khong.
  pause & exit /b 1
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Cap nhat pip...
python -m pip install --upgrade pip
echo [*] Cai thu vien (mediapipe, opencv)...
pip install -r "%SVC%\requirements.txt"

echo.
echo [OK] Da cai xong Hand Gesture Controller.
echo     Nho copy anh vao: %SVC%\images
echo     Chay: __bat\start_hand_gesture.bat
pause
