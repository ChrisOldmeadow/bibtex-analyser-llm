#!/bin/bash

python3.11 -m venv venv311
source venv311/bin/activate
pip install --upgrade pip
pip install "wordcloud==1.9.2" "Pillow==10.2.0" matplotlib pandas
echo "✅ Environment ready. Run with: source venv311/bin/activate"
