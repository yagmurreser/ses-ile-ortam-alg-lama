#!/bin/bash
set -e

echo "Python venv oluşturuluyor..."
python3 -m venv .venv
source .venv/bin/activate

echo "Bağımlılıklar yükleniyor..."
pip install -r requirements.txt

echo "Hazır. Backend başlatmak için:"
echo "  source .venv/bin/activate && uvicorn main:app --reload --port 8000"
