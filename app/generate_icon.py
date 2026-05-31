# -*- coding: utf-8 -*-
"""Uygulama ikonu (app_icon.ico) uretir - radar / anomali temasi."""

import os
import math
from PIL import Image, ImageDraw


def lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def draw_icon(size):
    ss = size * 4  # supersample
    img = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))

    # Arka plan dikey gradyan (indigo -> mor)
    top = (67, 56, 202)
    bot = (124, 58, 237)
    bg = Image.new("RGB", (ss, ss))
    bd = ImageDraw.Draw(bg)
    for y in range(ss):
        bd.line([(0, y), (ss, y)], fill=lerp(top, bot, y / ss))
    mask = rounded_mask(ss, int(ss * 0.22))
    img.paste(bg, (0, 0), mask)

    d = ImageDraw.Draw(img)
    cx, cy = ss * 0.5, ss * 0.52
    R = ss * 0.30

    # Radar halkalari
    for r in (R, R * 0.66, R * 0.33):
        bbox = [cx - r, cy - r, cx + r, cy + r]
        d.ellipse(bbox, outline=(255, 255, 255, 90), width=max(2, int(ss * 0.006)))

    # Capraz eksenler
    d.line([(cx - R, cy), (cx + R, cy)], fill=(255, 255, 255, 70), width=max(2, int(ss * 0.005)))
    d.line([(cx, cy - R), (cx, cy + R)], fill=(255, 255, 255, 70), width=max(2, int(ss * 0.005)))

    # Normal noktalar
    pts = [(-0.12, 0.10), (0.16, -0.06), (-0.05, -0.16), (0.05, 0.18), (-0.18, -0.04)]
    for px, py in pts:
        x, y = cx + px * ss, cy + py * ss
        rr = ss * 0.018
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(255, 255, 255, 210))

    # Anomali noktasi (parlak amber, halkali)
    ax, ay = cx + 0.225 * ss, cy - 0.205 * ss
    glow = ss * 0.060
    d.ellipse([ax - glow, ay - glow, ax + glow, ay + glow], fill=(251, 191, 36, 80))
    rr = ss * 0.032
    d.ellipse([ax - rr, ay - rr, ax + rr, ay + rr], fill=(245, 158, 11, 255),
              outline=(255, 255, 255, 255), width=max(2, int(ss * 0.006)))

    img = img.resize((size, size), Image.LANCZOS)
    return img


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.ico")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    imgs = [draw_icon(s) for s in sizes]
    imgs[-1].save(out, format="ICO", sizes=[(s, s) for s in sizes])
    print("Ikon olusturuldu:", out)


if __name__ == "__main__":
    main()
