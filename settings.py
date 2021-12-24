import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

PRETRAINED_PATH = Path(__file__).parent / 'pretrained_models'
TEMP_FOLDER = Path(__file__).parent / 'temp'
LOGS_FOLDER = 'logs'

ALLOWED_MEMORY = 3.5  # choose your GPU memory in GB, min value 3.5GB
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
HAFT_PRECISION = True
GPU_FP32_PERFORMANCE = 1.911

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default_formatter': {
            'format': '[%(name)s:%(filename)s:%(lineno)d:%(levelname)s:%(asctime)s] %(message)s'
        },
    },

    'handlers': {
        'stream_handler': {
            'class': 'logging.StreamHandler',
            'formatter': 'default_formatter',
        },
        'file_handler': {
            'class': 'logging.FileHandler',
            'formatter': 'default_formatter',
            'filename': f'{LOGS_FOLDER}/log.log',
        },
    },

    'loggers': {
        'root': {
            'handlers': ['stream_handler', 'file_handler'],
            'level': 'INFO',
            'propagate': True
        }
    }
}


class RealESRGanModelsList:
    RealESRGAN_x4plus = 'RealESRGAN_x4plus'
    RealESRNet_x4plus = 'RealESRNet_x4plus'
    RealESRGAN_x4plus_anime_6B = 'RealESRGAN_x4plus_anime_6B'
    RealESRGAN_x2plus = 'RealESRGAN_x2plus'
    RealESRGANv2_anime_xsx2 = 'RealESRGANv2-anime-xsx2'
    RealESRGANv2_animevideo_xsx2_nousm = 'RealESRGANv2-animevideo-xsx2-nousm'
    RealESRGANv2_animevideo_xsx2 = 'RealESRGANv2-animevideo-xsx2'
    RealESRGANv2_anime_xsx4 = 'RealESRGANv2-anime-xsx4'
    RealESRGANv2_animevideo_xsx4_nousm = 'RealESRGANv2-animevideo-xsx4-nousm'
    RealESRGANv2_animevideo_xsx4 = 'RealESRGANv2-animevideo-xsx4'


REALESRGAN_MODEL = RealESRGanModelsList.RealESRGAN_x4plus
REALESRGAN_TILES = 350  # если много видеопамяти можно уменьшить до 0

RUDALLE_SUPERRESOLUTION = True  # включить улучшение качества изображений после ru-dalle
RUDALLE_REALESRGAN_MODEL_UPSCALER = RealESRGanModelsList.RealESRGAN_x4plus

TELEGRAM_MAX_IMAGE_WIDTH = 1280
TELEGRAM_MAX_IMAGE_HEIGHT = 1280

TELEGRAM_MAX_THUMB_WIDTH = 320
TELEGRAM_MAX_THUMB_HEIGHT = 320


