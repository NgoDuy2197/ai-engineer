@echo off
REM ==== Bat service Fruit Catch ====
setlocal
cd /d "%~dp0.."
set "SVC=services\fruit_catch"

if not exist "%SVC%\.venv\Scripts\python.exe" (
  echo [!] Chua cai dat. Dang chay setup truoc...
  call "%~dp0setup_fruit_catch.bat"
)

call "%SVC%\.venv\Scripts\activate.bat"
echo [*] Dang chay Fruit Catch (di chuyen dau de dua ro, C:choi lai, Q:thoat)...
python "%SVC%\main.py" %*
pause
