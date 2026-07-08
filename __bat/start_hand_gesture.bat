@echo off
REM ==== Bat service Hand Gesture Controller ====
setlocal
cd /d "%~dp0.."
set "SVC=services\hand_gesture_controller"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_hand_gesture.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Hand Gesture Controller (Q/ESC de thoat)...
python "%SVC%\main.py" %*
pause
