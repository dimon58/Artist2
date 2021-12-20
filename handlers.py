import logging
import time

from aiogram import types
from tqdm.contrib.telegram import trange

import settings
from core import dp

logger = logging.getLogger('root.handlers')


@dp.message_handler(commands=['draw'])
async def draw(message: types.Message):
    from services.inference_rudalle import generate
    logger.info(f'Draw "{message.text}" request by {message.from_user.id}')

    if len(message.text) == 5:
        await message.answer("Напишите описание изображения. /draw *описание*")
        return

    await message.answer('Начинаю рисовать...')
    async for i in trange(10, token=settings.TELEGRAM_BOT_TOKEN, chat_id=message.chat.id):
        time.sleep(0.5)

    generate('Абоба')


@dp.message_handler(commands=['help', 'h'])
async def help(message: types.Message):
    logger.info(f'Help request by the user {message.from_user.id}')

    with open('static/help.md', 'r', encoding='utf8') as file:
        await message.answer(file.read(), parse_mode=types.ParseMode.MARKDOWN)
