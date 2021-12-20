import logging.config
import os

import coloredlogs
from aiogram import executor

import settings

os.makedirs('logs', exist_ok=True)

logging.config.dictConfig(settings.LOGGING_CONFIG)
root_logger = logging.getLogger('root')

coloredlogs.install(fmt=settings.LOGGING_CONFIG['formatters']['default_formatter']['format'])

if __name__ == '__main__':
    from core import *
    from handlers import *

    root_logger.info('Program start')
    executor.start_polling(dp, skip_updates=True)
