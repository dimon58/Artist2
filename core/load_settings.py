import logging.config
import os

import coloredlogs
import torch

import settings

os.makedirs(settings.LOGS_FOLDER, exist_ok=True)
os.makedirs(settings.TEMP_FOLDER, exist_ok=True)

logging.config.dictConfig(settings.LOGGING_CONFIG)
coloredlogs.install(fmt=settings.LOGGING_CONFIG['formatters']['default_formatter']['format'])

logger = logging.getLogger('root.settings')
logger.info('Loading settings')


def _setup_rudalle():
    logger.debug('Loading rudalle settings')

    if settings.RUDALLE_BS == 'auto':

        if settings.ALLOWED_MEMORY < 4.5:
            settings.DEVICE = 'cpu'
            settings.RUDALLE_BS = 1
        elif settings.ALLOWED_MEMORY < 5:
            settings.RUDALLE_BS = 1
        elif settings.ALLOWED_MEMORY <= 9.5:
            settings.RUDALLE_BS = 5
        else:
            settings.RUDALLE_BS = 5


def _set_cpu_as_device():
    settings.DEVICE = 'cpu'
    settings.ALLOWED_MEMORY = 0
    settings.HAFT_PRECISION = False


def _setup_gpu():
    logger.debug('Loading GPU settings')

    if settings.DEVICE.startswith('cpu'):
        _set_cpu_as_device()

    if not torch.cuda.is_available():
        logger.error(f'Cuda is not available. Using cpu.')
        _set_cpu_as_device()

    if not settings.DEVICE.startswith('cuda'):
        logger.error(f'May be errors with device {settings.DEVICE}')
        settings.ALLOWED_MEMORY = 0
        settings.HAFT_PRECISION = False
        return

    try:
        device_properties = torch.cuda.get_device_properties(settings.DEVICE)

    except AssertionError as e:
        # Invalid device id
        logger.error(f'{e}. Using cpu.')
        _set_cpu_as_device()
        return

    except RuntimeError as e:
        # Incorrect device string
        logger.error(f'{e}. Using cpu.')
        _set_cpu_as_device()
        return

    if settings.ALLOWED_MEMORY == 'auto':
        settings.ALLOWED_MEMORY = device_properties.total_memory / 1024 ** 3

    if settings.HAFT_PRECISION == 'auto':
        settings.HAFT_PRECISION = torch.cuda.is_bf16_supported()


def load_settings():
    _setup_gpu()
    _setup_rudalle()


if __name__ == '__main__':
    load_settings()
