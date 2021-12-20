import logging

from aiogram import types

from core import dp
from services.inference_rudalle import generate

logger = logging.getLogger('root.handlers')


@dp.message_handler(commands=['draw'])
async def draw(message: types.Message):
    logger.info(f'Draw "{message.text}" request by {message.from_user.id}')

    if len(message.text) == 5:
        await message.answer("Напишите описание изображения. /draw *описание*")
        return

    text = message.text[5:]

    await message.answer(f'Начинаю рисовать "{text}"')

    image = generate(text, chat_id=message.chat.id)[0]
    await message.answer_photo(image.getvalue())


@dp.message_handler(commands=['help', 'h'])
async def help(message: types.Message):
    logger.info(f'Help request by the user {message.from_user.id}')

    with open('static/help.md', 'r', encoding='utf8') as file:
        await message.answer(file.read(), parse_mode=types.ParseMode.MARKDOWN)
