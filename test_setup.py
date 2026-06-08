#!/usr/bin/env python3
"""
test_setup.py - Environment verification for the geo-ai project.

Confirms every core dependency imports inside geo_env and reports
versions + hardware acceleration. Run this before building any modules.

Usage:
    python test_setup.py          # quick import + version checks
    python test_setup.py --deep   # also loads YOLO + EasyOCR models
                                   # (downloads model weights on first run)
"""

import importlib
import sys
from importlib import metadata

# --- optional pretty output -------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    HAVE_RICH = True
except Exception:
    console = None
    HAVE_RICH = False


def out(msg=""):
    if HAVE_RICH:
        console.print(msg)
    else:
        print(msg)


def version_of(mod, dist=None):
    v = getattr(mod, "__version__", None)
    if v:
        return str(v)
    if dist:
        try:
            return metadata.version(dist)
        except Exception:
            pass
    return "unknown"


def check(module_name, dist=None, probe=None):
    """Import a module and optionally run probe(mod) -> str for extra info."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return False, f"import failed: {e}"
    detail = f"v{version_of(mod, dist)}"
    if probe is not None:
        try:
            extra = probe(mod)
            if extra:
                detail += f"  -  {extra}"
        except Exception as e:
            detail += f"  -  probe failed: {e}"
    return True, detail


# --- probes -----------------------------------------------------------------
def torch_probe(torch):
    if torch.cuda.is_available():
        return f"CUDA ready ({torch.cuda.get_device_name(0)})"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "MPS (Apple GPU) ready"
    return "CPU only"


def yolo_probe(ultralytics):
    from ultralytics import YOLO  # noqa: F401
    return "YOLO entry point OK"


# --- checks -----------------------------------------------------------------
CHECKS = [
    ("torch",         lambda: check("torch", probe=torch_probe)),
    ("torchvision",   lambda: check("torchvision")),
    ("opencv-python", lambda: check("cv2", dist="opencv-python")),
    ("pillow",        lambda: check("PIL", dist="pillow")),
    ("numpy",         lambda: check("numpy")),
    ("transformers",  lambda: check("transformers")),
    ("easyocr",       lambda: check("easyocr", dist="easyocr")),
    ("ultralytics",   lambda: check("ultralytics", probe=yolo_probe)),
    ("rich",          lambda: check("rich")),
    ("typer",         lambda: check("typer")),
]


def deep_checks():
    out()
    out("Running deep checks (may download model weights on first run)...")
    results = []
    try:
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        results.append(("YOLOv8n model load", True, "weights loaded"))
    except Exception as e:
        results.append(("YOLOv8n model load", False, str(e)))
    try:
        import easyocr
        easyocr.Reader(["en"], gpu=False)
        results.append(("EasyOCR reader (en)", True, "reader initialised"))
    except Exception as e:
        results.append(("EasyOCR reader (en)", False, str(e)))
    return results


def render(results):
    if HAVE_RICH:
        table = Table(box=box.ROUNDED)
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")
        for name, ok, detail in results:
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            table.add_row(name, status, detail)
        console.print(table)
    else:
        for name, ok, detail in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:16} {detail}")


def main():
    deep = "--deep" in sys.argv

    out(f"Python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        out("  note: project targets Python 3.12.x")
    out()

    results = [(name, *fn()) for name, fn in CHECKS]
    if deep:
        results += deep_checks()

    render(results)

    failed = [name for name, ok, _ in results if not ok]
    out()
    if failed:
        out(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        sys.exit(1)
    out("All checks passed. Environment is ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()