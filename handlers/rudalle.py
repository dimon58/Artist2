import logging
import time

import torch
from aiogram import types

import settings
from core import dp
from services.inference_realesrgan import upscale_pil
from services.inference_rudalle import generate
from services.utils import pil2tg, send_photo, get_progress_bar

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

    await send_photo(message, upscaled_image, image, 'result.png', preview=settings.RUDALLE_ENABLE_PREVIEW)


@dp.message_handler(commands=['drawm', 'dm'])
async def drawm(message: types.Message):
    logger.info(f'Draw many "{message.text}" request by {message.from_user.id}')

    if len(message.text) <= 6:
        return await message.answer("Напишите описание изображения. /draw *описание*")

    _, num, *text = message.text.split()
    text = ' '.join(text)
    num = int(num)

    ###################################################################################################################

    await message.answer(f'Начинаю рисовать "{text}"')

    images = generate(text, images_num=num, chat_id=message.chat.id)

    if not settings.RUDALLE_SUPERRESOLUTION:
        for image in images:
            await message.answer_photo(pil2tg(image))
        return
    torch.cuda.empty_cache()

    ###################################################################################################################

    progress_bar = get_progress_bar(images, 'Улучшение качества', message.chat.id)
    for image in progress_bar:
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

        ###############################################################################################################

        await send_photo(message, upscaled_image, image, 'result.png', settings.RUDALLE_ENABLE_PREVIEW)
