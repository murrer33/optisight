@echo off
title OptiSight: AI Marketing & Shelf Attention Analyzer
cd /d "%~dp0"

echo ===================================================
echo  OptiSight // AI Marketing & Attention Analyzer
echo ===================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [HATA] Python sisteminizde bulunamadi!
    echo Lutfen https://www.python.org adresinden Python 3.10+ yukleyin.
    pause
    exit /b
)

if not exist "venv" (
    echo [1/3] Sanal ortam olusturuluyor (venv)...
    python -m venv venv
)

echo [2/3] Bagimliliklar kontrol ediliyor...
call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

echo [3/3] OptiSight Baslatiliyor...
echo Kisayollar:
echo   Alt + H : Tekli Dikkat & Bakiş Taramasi
echo   Alt + B : A/B Rekabet Mucadelesi
echo   Alt + Q : Cikis
echo.

python main.py
pause
