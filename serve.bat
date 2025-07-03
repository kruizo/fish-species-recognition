@echo off
setlocal enabledelayedexpansion
echo Current directory: %CD%

echo [INFO] Checking for venv folder...
IF EXIST "venv" (
    echo [INFO] venv folder found.
) ELSE (
    echo [WARN] venv folder NOT found.
)

IF EXIST "venv" (
    echo [INFO] Using existing virtual environment...
    call .\venv\Scripts\activate.bat
    IF !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to activate virtual environment.
        echo [INFO] Removing corrupted venv folder...
        timeout /t 2 /nobreak >nul
        rmdir /s /q venv
        IF EXIST venv (
            echo [WARNING] Could not remove venv folder. Please close any terminals/editors using it and delete manually.
        ) ELSE (
            echo [INFO] venv folder removed successfully.
        )
        echo [INFO] Please run the script again to create a new environment.
        
        pause 
        exit /b 1
    )
    echo [INFO] Virtual environment activated successfully.
    echo [INFO] VIRTUAL_ENV: %VIRTUAL_ENV%
    echo [INFO] Skipping init.py - using existing environment.

    
    echo [INFO] Running App
    python run_browser.py
) ELSE (
    echo [INFO] Creating new virtual environment...
    
    REM Remove existing venv folder if it exists but is invalid
    IF EXIST "venv" (
        echo [INFO] Removing invalid venv folder...
        rmdir /s /q venv
    )
    
    python -m venv venv

    IF !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )

    IF !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause 
        exit /b 1
    )
    
    echo [INFO] Activating new virtual environment...
    call .\venv\Scripts\activate.bat
    IF !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to activate virtual environment.
        echo [INFO] Removing failed venv folder...
        timeout /t 2 /nobreak >nul
        rmdir /s /q venv
        IF EXIST venv (
            echo [WARNING] Could not remove venv folder. Please close any terminals/editors using it and delete manually.
        ) ELSE (
            echo [INFO] venv folder removed successfully.
        )
        echo [INFO] Please run the script again.
        
        pause 
        exit /b 1
    )

    echo [INFO] Running init.py to install dependencies...
    python init.py

    IF !ERRORLEVEL! NEQ 0 (
        echo [ERROR] init.py failed - dependency installation error.
        echo [INFO] Removing venv folder due to installation failure...
        deactivate
        timeout /t 2 /nobreak >nul
        rmdir /s /q venv
        IF EXIST venv (
            echo [WARNING] Could not remove venv folder. Please close any terminals/editors using it and delete manually.
        ) ELSE (
            echo [INFO] venv folder removed successfully.
        )
        echo [INFO] Please fix any issues and run the script again.
        
        pause
        exit /b 1
    )
    
    echo [INFO] Running App
    python run_browser.py
)

endlocal