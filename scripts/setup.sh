#!/bin/sh

# install pytorch (~1900 MB)
python3.9 -m venv venv &&
  . venv/bin/activate &&
  pip install -r requirements.txt &&
  pip install torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

#######################################################################################################################

mkdir "services"
mkdir "services/nns"
mkdir "services/nns/pretrained_models"

mkdir "tmp"
mkdir "tmp/inputs"
mkdir "tmp/outputs"

mkdir "logs"

#######################################################################################################################

sh scripts/get_rudalle.sh