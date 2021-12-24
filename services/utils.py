import io

from aiogram import types

import settings


def pil2tg(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')

    return buffer.getvalue()


async def send_photo(message, pil_image, thumb_source, filename):
    """
    Отправляем сообщение в телеграм с учетом максимальных размеров.
    Если размер pil_image не превышает 1280x1280, то отправляем как обычную картинку, иначе как документ
    """

    if pil_image.width <= settings.TELEGRAM_MAX_IMAGE_WIDTH and \
            pil_image.height <= settings.TELEGRAM_MAX_IMAGE_HEIGHT:
        return await message.answer_photo(pil2tg(pil_image))

    prepared_image = types.InputFile(pil2tg(pil_image), filename=filename)

    thumb_source.thumbnail((settings.TELEGRAM_MAX_THUMB_WIDTH, settings.TELEGRAM_MAX_THUMB_HEIGHT))
    thumb = pil2tg(thumb_source)

    return await message.answer_document(prepared_image, thumb=thumb)
