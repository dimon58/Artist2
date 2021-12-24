import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

LOGS_FOLDER = 'logs'

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
            'filename': 'logs/log.log',
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

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

ALLOWED_MEMORY = 3.5  # choose your GPU memory in GB, min value 3.5GB

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

PRETRAINED_PATH = Path(__file__).parent / 'pretrained_models'
TEMP_FOLDER = Path(__file__).parent / 'temp'

REALESRGAN_TILES = 350  # если много видеопамяти можно уменьшить до 0
