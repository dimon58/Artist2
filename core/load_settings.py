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

    if not settings.DEVICE.startswith('cuda') and not settings.DEVICE.startswith('cpu'):
        logger.warning(f'Unexpected device {settings.DEVICE}. May be errors with it.')
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

    total_memory = device_properties.total_memory / 1024 ** 3

    if settings.ALLOWED_MEMORY == 'auto':
        settings.ALLOWED_MEMORY = total_memory

    if total_memory < settings.ALLOWED_MEMORY:
        msg = f'Total memory lower than allowed memory. Max allowed memory size is {round(total_memory, 1)} GB. ' \
              f'Setting allowed memory = total memory. Work may be unstable.'
        logger.error(msg)
        settings.ALLOWED_MEMORY = total_memory

    if settings.HAFT_PRECISION == 'auto':
        settings.HAFT_PRECISION = torch.cuda.is_bf16_supported()

    if torch.__version__ >= '1.8.0':
        k = settings.ALLOWED_MEMORY / total_memory
        torch.cuda.set_per_process_memory_fraction(k, 0)


def load_settings():
    _setup_gpu()
    _setup_rudalle()


if __name__ == '__main__':
    load_settings()
