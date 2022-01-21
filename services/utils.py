import io
import logging
import uuid

import tqdm
import tqdm.contrib.telegram
from aiogram.types import ContentType

import settings

logger = logging.getLogger('root.utils')


class BytesWrapper(bytes):
    name: str


def pil2tg(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')

    return buffer.getvalue()


async def send_photo(message, pil_image, thumb_source, filename, preview=False):
    """
    Отправляем сообщение в телеграм с учетом максимальных размеров.
    Если размер pil_image не превышает 1280x1280, то отправляем как обычную картинку, иначе как документ
    """

    if pil_image.width <= settings.TELEGRAM_MAX_IMAGE_WIDTH and \
            pil_image.height <= settings.TELEGRAM_MAX_IMAGE_HEIGHT:
        return await message.answer_photo(pil2tg(pil_image))

    prepared_image = BytesWrapper(pil2tg(pil_image))
    prepared_image.name = filename

    thumb_source.thumbnail((settings.TELEGRAM_MAX_THUMB_WIDTH, settings.TELEGRAM_MAX_THUMB_HEIGHT))
    thumb = pil2tg(thumb_source)

    if preview:
        await message.answer_photo(prepared_image)

    return await message.answer_document(prepared_image, thumb=thumb)


async def download_photo(message, handler_action):
    """
    Скачивает фото из сообщения, записывая в логи во время какого действия это происходит.

    Возвращает путь до временного файла
    """
    if message.content_type == ContentType.PHOTO:
        image = message.photo[-1]
        image_name = f'image_{uuid.uuid4().hex}.jpg'

    else:
        if not message.document.mime_type.startswith('image/'):
            msg = f'Wrong image format for {handler_action} by user {message.from_user.id}'

            logger.info(msg)
            await message.reply(f"Нужно отправить изображение, а не {message.document.mime_type}")

            raise ValueError(msg)

        image = message.document
        image_name = f'image_{uuid.uuid4().hex}.{message.document.mime_type.split("/")[-1]}'

    image_path = settings.TEMP_FOLDER / image_name
    logger.debug(f'Downloading image from user {message.from_user.id}')
    await image.download(destination_file=image_path)

    return image_path


def calc_approx_upscale_time(w, h):
    a = 0.0529117 * settings.GPU_FP32_PERFORMANCE / 1.911
    b = 9.79244

    time = int(w * h * a / 1000 + b)

    seconds = time % 60
    minutes = time // 60

    approx_time = "Ожидаемое время"

    if minutes > 0:
        approx_time += f" {minutes} мин"

    if seconds > 0:
        approx_time += f" {seconds} сек"

    return approx_time


def get_progress_bar(iterable, desc, chat_id=None):
    if chat_id is None:
        return tqdm.tqdm(iterable, desc=desc)

    return tqdm.contrib.telegram.tqdm(
        iterable, desc=desc,
        token=settings.TELEGRAM_BOT_TOKEN, chat_id=chat_id
    )
