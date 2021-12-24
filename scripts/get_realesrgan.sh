#!/bin/sh

apt-get -qq update &&
  apt-get -qq install -y ffmpeg libsm6 libxext6

. ./venv/bin/activate
. ./venv/Scriptsc/activate

cd services/nns
git clone https://github.com/xinntao/Real-ESRGAN.git

cd Real-ESRGAN
pip install torch==1.10.0+cu113 \
  torchvision==0.11.1+cu113 \
  torchaudio==0.10.0+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html \
  pip install -r requirements.txt
python setup.py develop
cd ..
cd ..
cd ..


# download pretrained models

# RealESRGAN
if [ ! -f "services/nns/pretrained_models/RealESRNet_x4plus.pth" ]; then
  wget "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth" -P "services/nns/pretrained_models"
fi

# neural network working with faces (~100+350 MB)
if [ ! -f "venv/lib/python3.9/site-packages/facexlib/weights/detection_Resnet50_Final.pth" ]; then
  wget "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" -P "venv/lib/python3.9/site-packages/facexlib/weights"
fi

if [ ! -f "venv/lib/python3.9/site-packages/gfpgan/weights/GFPGANCleanv1-NoCE-C2.pth" ]; then
  wget "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth" -P "venv/lib/python3.9/site-packages/gfpgan/weights"
fi