import logging

from aiogram import executor

import core
import settings

root_logger = logging.getLogger('root')

core.load_settings()

if __name__ == '__main__':
    root_logger.info('Program start')
    root_logger.info(f'Available {round(settings.ALLOWED_MEMORY, 1)} GB gpu memory')
    root_logger.debug(f'Loading handlers')

    from core import *
    from handlers import *

    register_handlers()
    executor.start_polling(dp, skip_updates=True)
