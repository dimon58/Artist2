import io


def pil2tg(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')

    return buffer.getvalue()
