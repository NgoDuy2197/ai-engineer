@echo off
REM ==== Bat service Air Painter ====
setlocal
cd /d "%~dp0.."
set "SVC=services\air_painter"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_air_painter.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Air Painter (1 ngon:ve  2 ngon:chon  C:xoa  S:luu  Q:thoat)...
python "%SVC%\main.py" %*
pause
