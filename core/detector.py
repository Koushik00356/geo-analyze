"""
core/detector.py - Object detection (YOLOv8 / COCO).

Honest scope, v1:
  - PRIMARY value: an inventory of what's in the frame (cars, people, signs,
    plants, vehicles...). Useful context, and the foundation for more.
  - SECONDARY value: weak location signal. Stock YOLOv8 only knows COCO's 80
    generic classes, and "car" or "person" says almost nothing about *where*.
    So most detections contribute NOTHING to location - by design. The
    reasoning engine ignores any class not listed in knowledge.json's
    "objects" section, which currently holds only forward-looking,
    region-specific classes (e.g. auto_rickshaw) that a CUSTOM-trained model
    would emit. Until you train one, objects enrich the inventory, not the
    location guess. That honesty is the point.

Same plug interface as OCR: it produces `Evidence`, so it drops straight into
the reasoning engine with zero changes there.

Quick test:
    python core/detector.py images/test.jpg
    python core/detector.py images/test.jpg --model yolov8s.pt --min-conf 0.4


from __future__ import annotations
"""
from __future__ import annotations
import os
from collections import Counter
from dataclasses import dataclass
from functools import cached_property

# Evidence is defined in the reasoning module. Dual-import so this file runs
# both as part of the package (from main) and standalone for testing.
try:
    from core.reasoning import Evidence
except ImportError:
    from reasoning import Evidence


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: list[int]          # [x1, y1, x2, y2]


@dataclass
class DetectionResult:
    detections: list[Detection]

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(d.class_name for d in self.detections))

    @property
    def present(self) -> set[str]:
        return {d.class_name for d in self.detections}

    @property
    def has_objects(self) -> bool:
        return bool(self.detections)


class ObjectDetector:
    """Wraps a YOLOv8 model. Like OCREngine, the model loads lazily and is
    reused, because construction is the expensive part."""

    def __init__(self, model_path: str = "yolov8n.pt", gpu: bool | None = None):
        self.model_path = model_path
        self._gpu = gpu
        self._model = None

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
    def model(self):
        if self._model is None:
            from ultralytics import YOLO          # first run downloads weights
            self._model = YOLO(self.model_path)
        return self._model

    def analyze(self, image_path: str, min_confidence: float = 0.30) -> DetectionResult:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        device = 0 if self.gpu else "cpu"
        res = self.model(image_path, conf=min_confidence, device=device, verbose=False)[0]
        names = res.names
        detections = []
        for box in res.boxes:
            cid = int(box.cls[0])
            detections.append(Detection(
                class_name=names[cid],
                confidence=round(float(box.conf[0]), 3),
                bbox=[int(v) for v in box.xyxy[0].tolist()],
            ))
        return DetectionResult(detections)


def to_evidence(result: DetectionResult) -> list[Evidence]:
    """One Evidence item per distinct class. Strength rises with both how many
    instances were seen and how confident the detections were. Unmapped
    classes are emitted too, but the reasoning engine drops them - so generic
    COCO objects cost nothing and region-specific ones (once a custom model
    exists) contribute automatically."""
    out = []
    for name in result.present:
        confs = [d.confidence for d in result.detections if d.class_name == name]
        avg = sum(confs) / len(confs)
        strength = round(min(1.0, len(confs) / 3.0) * avg, 3)
        out.append(Evidence("object", name, strength, "detector"))
    return out


# --- standalone test --------------------------------------------------------
def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Object detection inventory for one image.")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--min-conf", type=float, default=0.30)
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
    except Exception:
        console = None

    def say(m):
        console.print(m) if console else print(m)

    det = ObjectDetector(model_path=args.model)
    say(f"Loading {args.model} (gpu={det.gpu})...")
    result = det.analyze(args.image, min_confidence=args.min_conf)

    if not result.has_objects:
        say("No objects detected.")
        return

    if console:
        t = Table(box=box.ROUNDED, title="Object inventory")
        t.add_column("Class", style="bold")
        t.add_column("Count", justify="right")
        for name, n in sorted(result.counts.items(), key=lambda kv: kv[1], reverse=True):
            t.add_row(name, str(n))
        console.print(t)
    else:
        for name, n in sorted(result.counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:18} x{n}")

    ev = to_evidence(result)
    located = [e for e in ev if e.key in ("auto_rickshaw", "indian_number_plate", "yellow_black_taxi")]
    say(f"\nDetected {len(result.detections)} objects across {len(result.counts)} classes.")
    say(f"Location-bearing objects: {[e.key for e in located] or 'none (generic COCO classes only)'}")


if __name__ == "__main__":
    _main()