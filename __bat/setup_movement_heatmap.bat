@echo off
REM ==== Cai dat moi truong ao cho Movement Heatmap (chay 1 lan) ====
setlocal
cd /d "%~dp0.."
set "SVC=services\movement_heatmap"

echo [*] Tao moi truong ao...
python -m venv "%SVC%\.venv"
if errorlevel 1 (
  echo [!] Khong tao duoc venv. Kiem tra Python co trong PATH khong.
  pause & exit /b 1
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Cap nhat pip...
python -m pip install --upgrade pip
echo [*] Cai thu vien (ultralytics, openvino, opencv, lapx)...
pip install -r "%SVC%\requirements.txt"

echo.
echo [OK] Da cai xong Movement Heatmap.
echo     Chay: __bat\start_movement_heatmap.bat
pause
