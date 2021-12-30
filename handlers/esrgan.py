import logging
import os
import time
import uuid
from typing import Optional

import cv2
from PIL import Image
from aiogram import types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ContentType

import settings
from core import dp
from services.nns.inference_realesrgan import upscale
from services.utils import send_photo, calc_approx_upscale_time

logger = logging.getLogger('root.handlers')


class UpscaleForm(StatesGroup):
    image = State()


@dp.message_handler(commands=['upscale', 'us'])
async def upscale_start(message: types.Message):
    logger.info(f'Start upscale by user {message.from_user.id}')
    # Set state
    await UpscaleForm.image.set()

    await message.answer("Теперь отправьте картинку, которую хотите заапскейлить")


# You can use state '*' if you need to handle all states
@dp.message_handler(state='*', commands=['cancel'])
@dp.message_handler(lambda message: message.text.lower() == 'cancel', state='*')
async def cancel_handler(message: types.Message, state: FSMContext, raw_state: Optional[str] = None):
    logger.info(f'Cancel upscale by user {message.from_user.id}')

    if raw_state is None:
        return

    # Cancel state and inform user about it
    await state.finish()
    # And remove keyboard (just in case)
    await message.reply('Canceled.', reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(lambda message: message.content_type not in (ContentType.PHOTO, ContentType.DOCUMENT),
                    state=UpscaleForm.image)
async def failed_process_image(message: types.Message):
    logger.info(f'Wrong image format for upscale by user {message.from_user.id}')

    return await message.reply("Нужно отправить изображение")


@dp.message_handler(content_types=(ContentType.PHOTO, ContentType.DOCUMENT))
async def process_image(message: types.Message, state: FSMContext):
    logger.info(f'Upscale image by user {message.from_user.id}')

    if message.content_type == ContentType.PHOTO:
        image = message.photo[-1]
        image_name = f'image_{uuid.uuid4().hex}.jpg'

    else:
        if not message.document.mime_type.startswith('image/'):
            logger.info(f'Wrong image format for upscale by user {message.from_user.id}')
            return await message.reply(f"Нужно отправить изображение, а не {message.document.mime_type}")

        image = message.document
        image_name = f'image_{uuid.uuid4().hex}.{message.document.mime_type.split("/")[-1]}'

    image_path = settings.TEMP_FOLDER / image_name
    logger.debug(f'Downloading image from user {message.from_user.id}')
    await image.download(destination_file=image_path)

    async with state.proxy() as data:
        data.state = None

    original_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    os.remove(image_path)

    w, h, _ = original_image.shape
    await message.answer(f"Ожидайте примерно {calc_approx_upscale_time(w, h)}")

    start = time.perf_counter()

    upscaled_image = upscale(
        original_image,
        model_name=settings.REALESRGAN_MODEL,
        tile=settings.REALESRGAN_TILES,
        face_enhance=True,
        half=settings.HAFT_PRECISION
    )

    upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))

    end = time.perf_counter()
    print(f'{upscaled_image.width // 4}x{upscaled_image.height // 4} upscaled in {end - start:.4f} sec')

    ###################################################################################################################

    await send_photo(message, upscaled_image, upscaled_image, 'upscaled.png')
