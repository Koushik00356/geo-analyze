"""
core/ocr.py - Text extraction and script detection.

Responsibilities (deliberately narrow):
    1. Pull text out of an image with EasyOCR.
    2. Identify which writing systems (scripts) appear in that text.

It does NOT decide what a script means geographically. Telugu text is a
*signal*; turning "Telugu" into "likely Telangana/Andhra Pradesh" is the
reasoning engine's job (core/reasoning.py). Keeping that boundary clean is
what lets the pipeline stay modular.

Quick test:
    python core/ocr.py images/test.jpg
    python core/ocr.py images/test.jpg --langs en,hi
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property

# --- script detection -------------------------------------------------------
# Unicode block ranges, ordered roughly by how often we expect to hit them.
# A "script" here is a writing system, not a language: Hindi and Marathi both
# use Devanagari, so OCR can name the script with confidence but not the
# language. We surface the script and let reasoning weigh it.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Latin":      [(0x0041, 0x024F)],
    "Greek":      [(0x0370, 0x03FF)],
    "Cyrillic":   [(0x0400, 0x04FF)],
    "Hebrew":     [(0x0590, 0x05FF)],
    "Arabic":     [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "Devanagari": [(0x0900, 0x097F)],
    "Bengali":    [(0x0980, 0x09FF)],
    "Gurmukhi":   [(0x0A00, 0x0A7F)],
    "Gujarati":   [(0x0A80, 0x0AFF)],
    "Oriya":      [(0x0B00, 0x0B7F)],
    "Tamil":      [(0x0B80, 0x0BFF)],
    "Telugu":     [(0x0C00, 0x0C7F)],
    "Kannada":    [(0x0C80, 0x0CFF)],
    "Malayalam":  [(0x0D00, 0x0D7F)],
    "Sinhala":    [(0x0D80, 0x0DFF)],
    "Thai":       [(0x0E00, 0x0E7F)],
    "Hiragana":   [(0x3040, 0x309F)],
    "Katakana":   [(0x30A0, 0x30FF)],
    "Han":        [(0x3400, 0x4DBF), (0x4E00, 0x9FFF)],
    "Hangul":     [(0x1100, 0x11FF), (0xAC00, 0xD7AF)],
}


def detect_script(ch: str) -> str | None:
    """Return the script name for a single character, or None for
    digits, punctuation, symbols and whitespace."""
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


# --- result containers ------------------------------------------------------
@dataclass
class TextBlock:
    """One detected piece of text with where it is and how sure OCR is."""
    text: str
    confidence: float
    bbox: list[list[int]]          # four [x, y] corner points
    scripts: list[str] = field(default_factory=list)


@dataclass
class OCRResult:
    """Everything the OCR stage hands downstream."""
    blocks: list[TextBlock]
    full_text: str
    script_counts: dict[str, int]  # script name -> character count
    dominant_script: str | None
    languages_loaded: list[str]

    @property
    def has_text(self) -> bool:
        return bool(self.blocks)

    @property
    def scripts(self) -> list[str]:
        """All scripts seen, most frequent first - a tidy signal list
        for the reasoning engine."""
        return sorted(self.script_counts, key=self.script_counts.get, reverse=True)


def _aggregate_scripts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        s = detect_script(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    return counts


# --- engine -----------------------------------------------------------------
class OCREngine:
    """Wraps EasyOCR. The Reader is expensive to build (it loads neural
    models), so we create it once, lazily, and reuse it across images.

    Note on languages: EasyOCR only *reads* scripts whose models you load.
    Default is English. To read Indic scripts pass e.g. langs=["en", "hi"]
    or ["en", "te"]. EasyOCR can't combine arbitrary script families in one
    Reader - English pairs with most, but two unrelated Indic scripts may
    be rejected. If you need many, run multiple engines and merge results.
    """

    def __init__(self, languages: list[str] | tuple[str, ...] = ("en",),
                 gpu: bool | None = None):
        self.languages = list(languages)
        self._gpu = gpu          # None -> auto-detect on first use
        self._reader = None

    @cached_property
    def gpu(self) -> bool:
        if self._gpu is not None:
            return self._gpu
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @property
    def reader(self):
        if self._reader is None:
            import easyocr  # imported here so the module loads fast
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def analyze(self, image_path: str, min_confidence: float = 0.30) -> OCRResult:
        """Run OCR on one image and return structured results."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # EasyOCR returns [(bbox, text, confidence), ...]
        raw = self.reader.readtext(image_path)

        blocks: list[TextBlock] = []
        for bbox, text, conf in raw:
            text = text.strip()
            if not text or conf < min_confidence:
                continue
            blocks.append(TextBlock(
                text=text,
                confidence=round(float(conf), 3),
                bbox=[[int(x), int(y)] for x, y in bbox],
                scripts=sorted(_aggregate_scripts(text)),
            ))

        full_text = "\n".join(b.text for b in blocks)
        script_counts = _aggregate_scripts(full_text)
        dominant = (max(script_counts, key=script_counts.get)
                    if script_counts else None)

        return OCRResult(
            blocks=blocks,
            full_text=full_text,
            script_counts=script_counts,
            dominant_script=dominant,
            languages_loaded=self.languages,
        )


# --- standalone test harness ------------------------------------------------
def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OCR + script detection on one image.")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--langs", default="en",
                        help="comma-separated EasyOCR language codes (default: en)")
    parser.add_argument("--min-conf", type=float, default=0.30,
                        help="drop detections below this confidence (default: 0.30)")
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
    except Exception:
        console = None

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    engine = OCREngine(languages=langs)

    def say(msg):
        console.print(msg) if console else print(msg)

    say(f"Loading EasyOCR ({', '.join(langs)}, gpu={engine.gpu})...")
    result = engine.analyze(args.image, min_confidence=args.min_conf)

    if not result.has_text:
        say("No text detected.")
        return

    if console:
        table = Table(box=box.ROUNDED, title="Detected text")
        table.add_column("Conf", justify="right")
        table.add_column("Scripts")
        table.add_column("Text")
        for b in result.blocks:
            table.add_row(f"{b.confidence:.2f}", ", ".join(b.scripts) or "-", b.text)
        console.print(table)
    else:
        for b in result.blocks:
            print(f"  [{b.confidence:.2f}] ({', '.join(b.scripts)}) {b.text}")

    say(f"\nScripts present: {result.script_counts}")
    say(f"Dominant script: {result.dominant_script}")


if __name__ == "__main__":
    _main()