import logging

from aiogram import types

from core import dp

logger = logging.getLogger('root.handlers')


@dp.message_handler(commands=['help', 'h'])
async def help(message: types.Message):
    logger.info(f'Help request by the user {message.from_user.id}')

    with open('static/help.md', 'r', encoding='utf8') as file:
        await message.answer(file.read(), parse_mode=types.ParseMode.MARKDOWN)


@dp.message_handler()
async def other_command(message: types.Message):
    logger.info(f'Unknown command by the user {message.from_user.id}')
    await message.answer("Неизвестная команда. Напишите /help для помощи")
