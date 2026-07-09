@echo off
REM ==== Bat service Movement Heatmap ====
setlocal
cd /d "%~dp0.."
set "SVC=services\movement_heatmap"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_movement_heatmap.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Movement Heatmap (H:heatmap B:khung S:luu C:reset Q:thoat)...
python "%SVC%\main.py" %*
pause
