import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from constants import RealESRGanModelsList

load_dotenv()

# -------------------- paths -------------------- #

NNS_PATH = Path(__file__).parent / 'services' / 'nns'  # путь до папки с нейросетями
TEMP_FOLDER = Path(__file__).parent / 'temp'  # путь до папки с временными файлами
LOGS_FOLDER = 'logs'  # папка для логов

# ------------------- Logging ------------------- #

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

# ------------------- Telegram ------------------- #

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')  # токен телеграм бота

TELEGRAM_MAX_IMAGE_WIDTH = 1280  # максимальная ширина отправляемого изображения
TELEGRAM_MAX_IMAGE_HEIGHT = 1280  # максимальная высота отправляемого изображения

TELEGRAM_MAX_THUMB_WIDTH = 320  # максимальная ширина отправляемого изображения для предпросмотра
TELEGRAM_MAX_THUMB_HEIGHT = 320  # максимальная высота отправляемого изображения для предпросмотра

# ------------------- GPU ------------------- #

# cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mlc, xla, lazy, vulkan, meta, hpu
DEVICE = ('cuda:0' if torch.cuda.is_available() else 'cpu')  # устройство для работы нейросетей. cuda

ALLOWED_MEMORY = 'auto'  # choose your GPU memory in GB, min value 3.5GB or 'auto'

# Использование половинной точности. Выбор из: True, False, 'auto'.
# Если auto, то используется True, если возможно
HAFT_PRECISION = 'auto'

GPU_FP32_PERFORMANCE = 1.911  # теоретическая производительность GPU в TFlops

# ------------------- NNS ------------------- #
PRETRAINED_PATH = NNS_PATH / 'pretrained_models'  # путь до хранилища весов моделей

REALESRGAN_MODEL = RealESRGanModelsList.RealESRGAN_x4plus  # модель для апскейла
REALESRGAN_UPSCALE = 4  # результирующее улучшение изображения
# размер тайла для апскейла. если не хватает видеопамяти можно поставить 350. Если установить 0, то не будет разбивки.
REALESRGAN_TILES = 640  # при 640 будет не более 4 тайлов при апскейле изображения из телеграма

RUDALLE_SUPERRESOLUTION = True  # включить улучшение качества изображений после ru-dalle
RUDALLE_REALESRGAN_MODEL_UPSCALER = RealESRGanModelsList.RealESRGAN_x2plus  # модель для апскейла после ru-dalle
RUDALLE_REALESRGAN_UPSCALE = 4  # результирующее улучшение изображения после ru-dalle
RUDALLE_ENABLE_PREVIEW = True  # Включить отправку сжатой версии изображения, если исходное слишком большое
RUDALLE_BS = 'auto'  # максимальное количество картинок, которые генерируются за раз

# выбор предобученой модели для трансформации в аниме
# варианты: celeba_distill.pt, face_paint_512_v1.pt, face_paint_512_v2.pt, paprika.pt
ANIMEGANV2_PATH = NNS_PATH / 'animegan2-pytorch' / 'weights' / 'face_paint_512_v2.pt'
# так как большие разрешения фото для трансформации в аниме не влезают в память,
# то нужно уменьшить разрешение до некоторого значения
ANIMEGANV2_DOWNSCALE_SIZE = 512
ANIMEGANV2_X32 = None  # Resize images to multiple of 32
