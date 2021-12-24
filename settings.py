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

REALESRGAN_TILES = 350  # если много видеопамяти можно уменьшить до 0
