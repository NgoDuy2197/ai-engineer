@echo off
REM ==== Cai dat moi truong ao cho Angel Wings (chay 1 lan) ====
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
set "SVC=services\angel_wings"

echo [*] Tao moi truong ao...
python -m venv "%SVC%\.venv"
if errorlevel 1 (
  echo [!] Khong tao duoc venv. Kiem tra Python co trong PATH khong.
  pause & exit /b 1
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Cap nhat pip...
python -m pip install --upgrade pip
echo [*] Cai thu vien (mediapipe, opencv, numpy)...
pip install -r "%SVC%\requirements.txt"

echo.
echo [OK] Da cai xong Angel Wings.
echo     Bo file PNG canh trai vao: %SVC%\__data\wings\
echo     Roi chay: __bat\start_angel_wings.bat
pause
