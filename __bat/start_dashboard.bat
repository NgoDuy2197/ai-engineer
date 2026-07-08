@echo off
REM ==== Bat Dashboard dieu huong (giong run_dashboard.bat o goc) ====
cd /d "%~dp0.."
echo [*] Dang khoi dong Dashboard tai http://localhost:8000 ...
python "web\server.py"
pause
