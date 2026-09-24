@echo off
REM Builds CrackSegmentation.exe on Windows, then wraps it into a proper
REM Setup.exe installer via Inno Setup if available. Must be run ON
REM Windows -- PyInstaller cannot cross-compile a Windows .exe from macOS
REM or Linux.
REM
REM Usage (from a Command Prompt or PowerShell):
REM   cd packaging
REM   build_windows.bat
REM
REM Output:
REM   packaging\dist\CrackSegmentation\CrackSegmentation.exe   (raw build)
REM   packaging\dist\CrackSegmentation-Setup.exe               (installer,
REM                                                              only if
REM                                                              Inno Setup
REM                                                              is found)

setlocal
cd /d "%~dp0"

if "%PYTHON_BIN%"=="" set PYTHON_BIN=python

echo == Using %PYTHON_BIN% ==
%PYTHON_BIN% --version
if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install it from https://python.org
    echo ^(the standard installer already includes Tkinter -- no extra step needed^)
    exit /b 1
)

echo == Installing/upgrading build dependencies ^(a venv is recommended^) ==
%PYTHON_BIN% -m pip install --upgrade -r ..\requirements.txt
if errorlevel 1 exit /b 1
REM PyMuPDF (embedded PDF manual panel) is licensed AGPL-3.0/commercial --
REM kept in its own file so it's a visible, separate choice. Installed here
REM by default so the built app has a working manual panel; remove this
REM step to build without it (the panel then just shows a fallback
REM message instead of the rendered PDF -- see requirements-optional.txt).
%PYTHON_BIN% -m pip install --upgrade -r ..\requirements-optional.txt
if errorlevel 1 exit /b 1
REM PyInstaller is a build-time-only tool, not a feature dependency -- kept
REM out of both requirements files above so a plain "pip install -r
REM requirements.txt" (to just run the tool from source) never pulls it in.
%PYTHON_BIN% -m pip install --upgrade "pyinstaller>=6.3"
if errorlevel 1 exit /b 1

echo == Cleaning previous build artifacts ==
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo == Running PyInstaller ==
%PYTHON_BIN% -m PyInstaller crack_segmentation_gui.spec --noconfirm
if errorlevel 1 exit /b 1

echo.
echo == PyInstaller build done ==
echo Folder: %cd%\dist\CrackSegmentation\
echo Run:    %cd%\dist\CrackSegmentation\CrackSegmentation.exe

echo.
echo == Looking for Inno Setup ^(to build a proper Setup.exe installer^) ==
set "ISCC="
where ISCC.exe >nul 2>nul && for /f "delims=" %%I in ('where ISCC.exe') do set "ISCC=%%I"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if defined ISCC (
    echo Found Inno Setup: %ISCC%
    "%ISCC%" crack_segmentation_installer.iss
    if errorlevel 1 (
        echo WARNING: Inno Setup compilation failed -- see the messages above.
        echo The raw build in dist\CrackSegmentation\ is still usable directly.
    ) else (
        echo.
        echo == Installer built ==
        echo %cd%\dist\CrackSegmentation-Setup.exe
    )
) else (
    echo Inno Setup ^(ISCC.exe^) not found -- skipping the Setup.exe step.
    echo Install it once ^(free^): https://jrsoftware.org/isdl.php
    echo Then re-run this script, or compile crack_segmentation_installer.iss
    echo yourself by opening it in the Inno Setup Compiler and clicking Compile.
    echo.
    echo For now you can still share dist\CrackSegmentation\ directly -- zip
    echo that folder and send it; CrackSegmentation.exe inside runs standalone,
    echo it just won't have a Start Menu entry or uninstaller.
)

echo.
echo Some antivirus tools / SmartScreen flag freshly-built, unsigned
echo PyInstaller executables ^(and unsigned installers^) as unrecognized.
echo That's a false positive from the lack of a paid code-signing
echo certificate, not a sign of a problem with this build -- the operator
echo may need to click "More info" -^> "Run anyway" the first time.
endlocal
