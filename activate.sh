#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    startup.sh
fi

source venv/bin/activate
