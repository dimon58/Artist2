import logging

from aiogram import types

from core import dp
from services.inference_rudalle import generate
from services.utils import pil2tg

logger = logging.getLogger('root.handlers')


@dp.message_handler(commands=['draw'])
async def draw(message: types.Message):
    logger.info(f'Draw "{message.text}" request by {message.from_user.id}')

    if len(message.text) <= 6:
        return await message.answer("Напишите описание изображения. /draw *описание*")

    text = message.text[6:]

    await message.answer(f'Начинаю рисовать "{text}"')

    image = generate(text, chat_id=message.chat.id)[0]

    await message.answer_photo(pil2tg(image))
