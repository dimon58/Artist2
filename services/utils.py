import io

import tqdm
import tqdm.contrib.telegram

import settings


class BytesWrapper(bytes):
    name: str


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

    prepared_image = BytesWrapper(pil2tg(pil_image))
    prepared_image.name = filename

    thumb_source.thumbnail((settings.TELEGRAM_MAX_THUMB_WIDTH, settings.TELEGRAM_MAX_THUMB_HEIGHT))
    thumb = pil2tg(thumb_source)

    return await message.answer_document(prepared_image, thumb=thumb)


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
