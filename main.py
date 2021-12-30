import logging.config
import os

import coloredlogs
from aiogram import executor

import settings

os.makedirs(settings.LOGS_FOLDER, exist_ok=True)
os.makedirs(settings.TEMP_FOLDER, exist_ok=True)

logging.config.dictConfig(settings.LOGGING_CONFIG)
root_logger = logging.getLogger('root')

coloredlogs.install(fmt=settings.LOGGING_CONFIG['formatters']['default_formatter']['format'])

if __name__ == '__main__':

    root_logger.info('Program start')
    root_logger.info(f'Available {round(settings.ALLOWED_MEMORY, 1)} GB gpu memory')
    root_logger.debug(f'Loading handlers')

    from core import *
    from handlers import *

    executor.start_polling(dp, skip_updates=True)
