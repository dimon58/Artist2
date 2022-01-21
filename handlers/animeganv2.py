import logging
import os
import time
from typing import Optional

from PIL import Image
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ContentType

import settings
from core import dp
from services.nns import inference_animeganv2
from services.utils import download_photo
from services.utils import send_photo

logger = logging.getLogger('root.handlers')


class ConvertToAnimeForm(StatesGroup):
    image = State()


@dp.message_handler(commands=['anime', 'an'])
async def upscale_start(message: types.Message):
    logger.info(f'Start converting to anime by user {message.from_user.id}')
    # Set state
    await ConvertToAnimeForm.image.set()

    await message.answer("Теперь отправьте картинку, которую хотите анимефицировать")


# You can use state '*' if you need to handle all states
@dp.message_handler(state='*', commands=['cancel'])
@dp.message_handler(lambda message: message.text.lower() == 'cancel', state='*')
async def cancel_handler(message: types.Message, state: FSMContext, raw_state: Optional[str] = None):
    logger.info(f'Cancel converting to anime by user {message.from_user.id}')

    if raw_state is None:
        return

    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Canceled.', reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(lambda message: message.content_type not in (ContentType.PHOTO, ContentType.DOCUMENT),
                    state=ConvertToAnimeForm.image)
async def failed_process_image(message: types.Message):
    logger.info(f'Wrong image format for converting to anime by user {message.from_user.id}')

    return await message.reply("Нужно отправить изображение")


@dp.message_handler(content_types=(ContentType.PHOTO, ContentType.DOCUMENT), state=ConvertToAnimeForm.image)
async def process_image(message: types.Message, state: FSMContext):
    logger.info(f'Convert image to anime request by user {message.from_user.id}')

    await message.answer('Начинаю трансформацию')

    try:
        image_path = await download_photo(message, 'converting to anime')
    except ValueError as e:
        return await message.reply(f"Нужно отправить изображение, а не {message.document.mime_type}")

    async with state.proxy() as data:
        data.state = None

    original_image = Image.open(image_path)

    start = time.perf_counter()

    try:

        if max(original_image.width, original_image.height) > settings.ANIMEGANV2_DOWNSCALE_SIZE:
            original_image.thumbnail(
                (settings.ANIMEGANV2_DOWNSCALE_SIZE, settings.ANIMEGANV2_DOWNSCALE_SIZE),
                Image.ANTIALIAS
            )

        converted_image = inference_animeganv2.inference(
            [original_image],
            device=settings.DEVICE,
            x32=settings.ANIMEGANV2_X32
        )[0]

    except RuntimeError as e:
        logger.exception(e)
        return await message.answer(str(e))

    end = time.perf_counter()
    logger.debug(f'{converted_image.width}x{converted_image.height} converted in {end - start:.4f} sec')

    ###################################################################################################################
    os.remove(image_path)

    await send_photo(message, converted_image, converted_image, 'upscaled.png')
