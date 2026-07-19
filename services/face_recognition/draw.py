"""
Ve chu Unicode (co dau tieng Viet) len frame OpenCV bang Pillow.
cv2.putText khong ve duoc dau tieng Viet, nen dung PIL de ve nhan ten.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:  # noqa
                pass
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def put_text_unicode(frame_bgr, text, org, size=22, color=(255, 255, 255),
                     bg=(0, 150, 0)):
    """Ve text (co dau) tai org=(x,y) goc tren-trai, kem nen mau bg."""
    x, y = org
    font = _get_font(size)
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    if bg is not None:
        draw.rectangle([l - 4, t - 2, r + 4, b + 2], fill=bg[::-1])  # bg la BGR -> RGB
    draw.text((x, y), text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
