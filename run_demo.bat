@echo off
echo ==========================================
echo  🩺 AB-GEN MEDICAL DEMO — Launcher
echo ==========================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado. Instala Python 3.10+
    pause
    exit /b
)

REM Instalar dependencias si es necesario
echo Verificando dependencias...
pip install -r requirements_demo.txt -q

echo.
echo Iniciando servidor...
echo Dashboard: http://localhost:5000
echo.
python server.py

pause
