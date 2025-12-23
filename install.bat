@echo off
chcp 65001 >nul
echo ============================================================
echo    УСТАНОВКА ЗАВИСИМОСТЕЙ ДЛЯ ПРОЕКТА
echo ============================================================
echo.

echo Проверка Python...
py --version
if errorlevel 1 (
    echo.
    echo ❌ Python не найден!
    echo Установите Python с https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo ✓ Python найден!
echo.
echo Установка зависимостей из requirements.txt...
echo Это может занять несколько минут...
echo.

py -m pip install --upgrade pip
py -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ⚠ Ошибка при установке!
    echo Попробуйте выполнить вручную:
    echo   py -m pip install -r requirements.txt
    pause
    exit /b 1
) else (
    echo.
    echo ============================================================
    echo ✅ ВСЕ ЗАВИСИМОСТИ УСТАНОВЛЕНЫ УСПЕШНО!
    echo ============================================================
    echo.
    echo Следующий шаг: запустите setup.bat
    echo.
)

pause











