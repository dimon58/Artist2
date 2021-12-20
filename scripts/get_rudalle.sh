#!/bin/sh

. ./venv/bin/activate

cd services/nns
git clone https://github.com/sberbank-ai/ru-dalle.git

cd ru-dalle
pip install -r requirements.txt
python setup.py develop
cd ..
cd ..
cd ..

echo "Downloading pretrained models"
python scripts/download_rudalle.py
