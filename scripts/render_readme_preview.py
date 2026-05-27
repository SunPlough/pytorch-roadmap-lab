from __future__ import annotations

import math
import random
import struct
import zlib
from pathlib import Path


WIDTH = 960
HEIGHT = 560
PADDING = 56


def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def mix(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def world_to_pixel(x: float, y: float) -> tuple[int, int]:
    px = PADDING + (x + 2.0) / 4.0 * (WIDTH - 2 * PADDING)
    py = HEIGHT - PADDING - (y + 2.0) / 4.0 * (HEIGHT - 2 * PADDING)
    return round(px), round(py)


def pixel_to_world(px: int, py: int) -> tuple[float, float]:
    x = (px - PADDING) / (WIDTH - 2 * PADDING) * 4.0 - 2.0
    y = (HEIGHT - PADDING - py) / (HEIGHT - 2 * PADDING) * 4.0 - 2.0
    return x, y


def set_pixel(buffer: bytearray, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        offset = (y * WIDTH + x) * 3
        buffer[offset : offset + 3] = bytes(color)


def draw_circle(buffer: bytearray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    r2 = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                set_pixel(buffer, x, y, color)


def write_png(path: Path, pixels: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    rows = []
    stride = WIDTH * 3
    for y in range(HEIGHT):
        rows.append(b"\x00" + pixels[y * stride : (y + 1) * stride])

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk("IHDR".encode(), struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
    png += chunk("IDAT".encode(), zlib.compress(b"".join(rows), level=9))
    png += chunk("IEND".encode(), b"")
    path.write_bytes(png)


def main() -> None:
    random.seed(42)
    bg_inner = (248, 250, 252)
    bg_outer = (235, 241, 246)
    blue = (44, 123, 229)
    coral = (229, 79, 65)
    ink = (27, 38, 59)
    grid = (213, 222, 232)
    boundary = (35, 45, 68)

    pixels = bytearray(WIDTH * HEIGHT * 3)

    for py in range(HEIGHT):
        for px in range(WIDTH):
            x, y = pixel_to_world(px, py)
            r = math.sqrt(x * x + y * y)
            score = 1 / (1 + math.exp(-10 * (r - 1.08)))
            color = mix(blue, coral, score)
            fade = 0.22 + 0.24 * min(1.0, abs(r - 1.08))
            color = mix(color, bg_inner, fade)
            vignette = min(1.0, math.sqrt(((px - WIDTH / 2) / WIDTH) ** 2 + ((py - HEIGHT / 2) / HEIGHT) ** 2) * 1.8)
            color = mix(color, bg_outer, vignette * 0.18)
            set_pixel(pixels, px, py, color)

    for value in [i / 2 for i in range(-4, 5)]:
        x1, y1 = world_to_pixel(value, -2)
        x2, y2 = world_to_pixel(value, 2)
        for py in range(min(y1, y2), max(y1, y2) + 1):
            set_pixel(pixels, x1, py, grid)
        x1, y1 = world_to_pixel(-2, value)
        x2, y2 = world_to_pixel(2, value)
        for px in range(min(x1, x2), max(x1, x2) + 1):
            set_pixel(pixels, px, y1, grid)

    for angle_step in range(720):
        angle = angle_step / 720 * 2 * math.pi
        for radius in [1.08, 1.09, 1.10]:
            px, py = world_to_pixel(radius * math.cos(angle), radius * math.sin(angle))
            set_pixel(pixels, px, py, boundary)

    for label, base_radius, color in [(0, 0.72, blue), (1, 1.45, coral)]:
        for _ in range(260):
            angle = random.random() * 2 * math.pi
            radius = random.gauss(base_radius, 0.07)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            px, py = world_to_pixel(x, y)
            draw_circle(pixels, px, py, 4, (255, 255, 255))
            draw_circle(pixels, px, py, 3, color if label else mix(color, ink, 0.12))

    output = Path("assets") / "decision-boundary-preview.png"
    output.parent.mkdir(exist_ok=True)
    write_png(output, pixels)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

