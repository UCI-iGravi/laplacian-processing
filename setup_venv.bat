@echo off
REM Creates a Python 3.12 .venv in this repo, installs requirements.txt,
REM and prints activation hint. On any failure, a full log is written to
REM setup_venv.log and the window stays open.

setlocal

set "VENV_DIR=%~dp0.venv"
set "LOG=%~dp0setup_venv.log"
set "PY=py -3.12"

echo === setup_venv.bat run at %DATE% %TIME% === > "%LOG%"

%PY% --version >> "%LOG%" 2>&1
if errorlevel 1 (
    echo Python 3.12 not found via the 'py' launcher.
    echo Install it from https://www.python.org/downloads/release/python-3120/
    echo or run:  winget install Python.Python.3.12
    goto :fail
)

if exist "%VENV_DIR%" (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo Existing venv is not Python 3.12 - removing and recreating.
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Existing Python 3.12 venv at %VENV_DIR% - reusing.
    )
)

if not exist "%VENV_DIR%" (
    echo Creating venv ^(Python 3.12^) at %VENV_DIR% ...
    %PY% -m venv "%VENV_DIR%" >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo Failed to create venv - see setup_venv.log
        goto :fail
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :fail

echo.
echo === Upgrading pip ===
powershell -NoProfile -Command "& { python -m pip install --upgrade pip 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE }"
if errorlevel 1 (
    echo pip upgrade failed - see setup_venv.log
    goto :fail
)

echo.
echo === Installing requirements ^(this can take several minutes^) ===
powershell -NoProfile -Command "& { python -m pip install -r '%~dp0requirements.txt' 2>&1 | Tee-Object -FilePath '%LOG%' -Append; exit $LASTEXITCODE }"
if errorlevel 1 (
    echo.
    echo Dependency install FAILED. See %LOG% for full output.
    goto :fail
)

echo.
echo Done. Activate with:
echo     %VENV_DIR%\Scripts\activate.bat
echo.
echo Run tests with:
echo     python -m pytest tests/
echo.
pause
endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1
