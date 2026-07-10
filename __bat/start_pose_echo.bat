@echo off
REM ==== Bat service Pose Echo (Bong Que Tre) ====
setlocal
cd /d "%~dp0.."
set "SVC=services\pose_echo"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_pose_echo.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Pose Echo (+/- chinh do tre, C:ghi lai, Q:thoat)...
python "%SVC%\main.py" %*
pause
