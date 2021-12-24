import logging
import time

from aiogram import types

import settings
from core import dp
from services.inference_realesrgan import upscale_pil
from services.inference_rudalle import generate
from services.utils import pil2tg

logger = logging.getLogger('root.handlers')


@dp.message_handler(commands=['draw'])
async def draw(message: types.Message):
    logger.info(f'Draw "{message.text}" request by {message.from_user.id}')

    if len(message.text) <= 6:
        return await message.answer("Напишите описание изображения. /draw *описание*")
    text = message.text[6:]

    ###################################################################################################################

    await message.answer(f'Начинаю рисовать "{text}"')

    image = generate(text, chat_id=message.chat.id)[0]

    if not settings.RUDALLE_SUPERRESOLUTION:
        return await message.answer_photo(pil2tg(image))

    await message.answer(f'Улучшение качества')

    ###################################################################################################################

    start = time.perf_counter()
    upscaled_image = upscale_pil(
        image,
        model_name=settings.RUDALLE_REALESRGAN_MODEL_UPSCALER,
        tile=settings.REALESRGAN_TILES,
        face_enhance=True,
        half=settings.HAFT_PRECISION
    )
    end = time.perf_counter()
    print(f'{upscaled_image.width // 4}x{upscaled_image.height // 4} upscaled in {end - start:.4f} sec')

    ###################################################################################################################

    # Отправляем сообщение в телеграм с учетом максимальных размеров.
    # Если размер pil_image не превышает 1280x1280, то отправляем как обычную картинку, иначе как документ

    if upscaled_image.width <= settings.TELEGRAM_MAX_IMAGE_WIDTH and \
            upscaled_image.height <= settings.TELEGRAM_MAX_IMAGE_HEIGHT:
        return await message.answer_photo(pil2tg(upscaled_image))

    prepared_image = pil2tg(upscaled_image)
    prepared_image.name = 'upscaled.png'

    image.thumbnail((settings.TELEGRAM_MAX_THUMB_WIDTH, settings.TELEGRAM_MAX_THUMB_HEIGHT))
    thumb = pil2tg(image)

    return await message.answer_document(prepared_image, thumb=thumb)
