"""
main.py - the `geo analyze` terminal tool.

Stitches the working modules into one command:

    image  ->  OCR (text + scripts)  ->  reasoning  ->  location hypotheses

Usage:
    python main.py analyze images/test.jpg
    python main.py analyze images/test.jpg --langs en,te
    python main.py analyze images/test.jpg --langs en,te --min-conf 0.4

(Once installed as a console script named `geo`, this becomes
 `geo analyze images/test.jpg` - see the note at the bottom.)
"""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.ocr import OCREngine, OCRResult
from core.reasoning import ReasoningEngine, LocationResult

app = typer.Typer(add_completion=False, help="Visual geolocation from a single image.")
console = Console()

KNOWLEDGE_PATH = os.path.join("data", "knowledge.json")


# --- rendering (kept separate so it's testable without running OCR) ---------
def render_report(ocr: OCRResult, location: LocationResult) -> None:
    # 1. Text the image revealed
    if ocr.has_text:
        t = Table(box=box.ROUNDED, title="Text detected")
        t.add_column("Conf", justify="right")
        t.add_column("Scripts")
        t.add_column("Text")
        for b in ocr.blocks:
            t.add_row(f"{b.confidence:.2f}", ", ".join(b.scripts) or "-", b.text)
        console.print(t)
        console.print(f"Scripts present: [bold]{ocr.script_counts}[/bold]   "
                      f"(dominant: {ocr.dominant_script})\n")
    else:
        console.print("[yellow]No text detected in image.[/yellow]\n")

    # 2. Where the evidence points
    if not location.country:
        console.print(Panel("No location-bearing evidence found.",
                            title="Location", border_style="red"))
        return

    loc = Table(box=box.SIMPLE_HEAVY, show_header=True)
    loc.add_column("Level")
    loc.add_column("Hypothesis", style="bold")
    loc.add_column("Confidence", justify="right")
    loc.add_column("Supported by")

    top = location.country[0]
    loc.add_row("Country", top.label, f"{top.confidence:.0%}", ", ".join(top.support))
    for h in location.country[1:3]:
        loc.add_row("", h.label, f"{h.confidence:.0%}", ", ".join(h.support))

    if location.region:
        for i, h in enumerate(location.region[:4]):
            level = "Region" if i == 0 else ""
            loc.add_row(level, h.label, f"{h.confidence:.0%}", ", ".join(h.support))
    else:
        loc.add_row("Region", "(no sub-region signal)", "-", "")

    loc.add_row("City", "insufficient evidence", "-", "")
    console.print(Panel(loc, title="Location hypotheses", border_style="cyan"))

    # 3. Honest footer
    console.print("[dim]Confidence reflects strength of evidence, not a calibrated "
                  "probability. Alternatives are kept deliberately.[/dim]")


# --- the command ------------------------------------------------------------
@app.command()
def analyze(
    image: str = typer.Argument(..., help="path to the image to analyze"),
    langs: str = typer.Option("en", help="comma-separated EasyOCR language codes, e.g. en,te"),
    min_conf: float = typer.Option(0.30, help="drop OCR detections below this confidence"),
    knowledge: str = typer.Option(KNOWLEDGE_PATH, help="path to knowledge.json"),
) -> None:
    """Analyze a single image and print location hypotheses with evidence."""
    if not os.path.exists(image):
        console.print(f"[red]Image not found:[/red] {image}")
        raise typer.Exit(code=1)

    lang_list = [s.strip() for s in langs.split(",") if s.strip()]
    ocr_engine = OCREngine(languages=lang_list)
    brain = ReasoningEngine(knowledge)

    console.print(f"Analyzing [bold]{image}[/bold]  "
                  f"(langs={','.join(lang_list)}, gpu={ocr_engine.gpu})\n")

    ocr_result = ocr_engine.analyze(image, min_confidence=min_conf)
    location = brain.reason(brain.from_ocr(ocr_result))

    render_report(ocr_result, location)


if __name__ == "__main__":
    app()


# To get the literal `geo analyze ...` command from your handoff, add an entry
# point in pyproject.toml:
#
#     [project.scripts]
#     geo = "main:app"
#
# then `pip install -e .` inside geo_env. Until then, use `python main.py analyze`.