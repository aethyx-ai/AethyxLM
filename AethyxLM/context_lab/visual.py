"""Dense visual-page planning; rendering remains an explicit experiment."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from context_lab.schema import ContextItem, VisualPagePlan


def plan_visual_pages(
    items: list[ContextItem],
    chars_per_page: int = 24000,
    width: int = 1536,
    height: int = 2048,
    columns: int = 3,
) -> tuple[VisualPagePlan, ...]:
    if chars_per_page <= 0 or width <= 0 or height <= 0 or columns <= 0:
        raise ValueError("visual page dimensions and capacity must be positive")
    chunks = []
    current = []
    current_ids = []
    length = 0
    for item in items:
        block = f"[{item.source_id} | {item.kind}]\n{item.text.strip()}\n"
        if current and length + len(block) > chars_per_page:
            chunks.append((tuple(current_ids), "\n".join(current)))
            current, current_ids, length = [], [], 0
        current.append(block)
        current_ids.append(item.source_id)
        length += len(block)
    if current:
        chunks.append((tuple(current_ids), "\n".join(current)))
    return tuple(
        VisualPagePlan(
            page_id=f"page-{index:04d}",
            source_ids=source_ids,
            text=text,
            width=width,
            height=height,
            columns=columns,
            estimated_image_units=math.ceil(width / 512) * math.ceil(height / 512),
        )
        for index, (source_ids, text) in enumerate(chunks, start=1)
    )


def render_visual_pages(pages: tuple[VisualPagePlan, ...], output_dir: Path):
    """Render planned pages with Pillow when explicitly requested."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Install Pillow to render visual context pages") from error
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    font = ImageFont.load_default()
    for page in pages:
        image = Image.new("RGB", (page.width, page.height), "white")
        draw = ImageDraw.Draw(image)
        margin = 28
        gutter = 20
        column_width = (page.width - 2 * margin - (page.columns - 1) * gutter) // page.columns
        chars_per_line = max(20, column_width // 7)
        lines = []
        for raw_line in page.text.splitlines():
            lines.extend(textwrap.wrap(raw_line, chars_per_line) or [""])
        line_height = 13
        lines_per_column = max(1, (page.height - 2 * margin) // line_height)
        dropped = max(0, len(lines) - lines_per_column * page.columns)
        for index, line in enumerate(lines[: lines_per_column * page.columns]):
            column = index // lines_per_column
            row = index % lines_per_column
            x = margin + column * (column_width + gutter)
            y = margin + row * line_height
            draw.text((x, y), line, fill="black", font=font)
        if dropped:
            raise RuntimeError(
                f"Visual page {page.page_id} overflowed by {dropped} lines; "
                "reduce chars_per_page rather than silently dropping context"
            )
        path = output_dir / f"{page.page_id}.png"
        image.save(path, format="PNG", optimize=True)
        outputs.append(path)
    return outputs

