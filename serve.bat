@echo off

IF NOT EXIST "venv" (
    echo Virtual environment not found. Creating and activating...
    python -m venv venv
    call .\venv\Scripts\activate
    python init.py
    python app.py
) ELSE (
    call .\venv\Scripts\activate
)

python app.py
