@echo off
REM ==== Bat server + mo trang dieu huong vao cac service ====
cd /d "%~dp0"
echo [*] Dang khoi dong Dashboard tai http://localhost:8000 ...
echo [*] Dong cua so nay de tat server.
python "web\server.py"
pause
