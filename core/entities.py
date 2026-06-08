"""
core/entities.py - Mine the OCR text for location-bearing entities.

Three kinds of signal, all from text you already have (no new model):
  1. Place names  - fuzzy-matched against a gazetteer in knowledge.json. The
     fuzziness matters: OCR is noisy ("Hyderabad" might come back slightly
     mangled), so we accept close matches and let match quality set strength.
  2. Plate codes  - Indian registration codes whose first two letters encode
     the state (TS=Telangana, AP=Andhra Pradesh, ...). Near-definitive when
     readable.
  3. Format patterns - currency (Rs/INR), +91 phone numbers, 6-digit PINs.

Each becomes an `Evidence` and flows into the reasoning engine unchanged. A
recognised city is the first signal that can break the Telangana/AP tie and
yield an actual city prediction.

Quick test:
    python core/entities.py "Venkataramana Colour Lab, Hyderabad"
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import cached_property

try:
    from core.reasoning import Evidence
except ImportError:
    from reasoning import Evidence


class EntityExtractor:
    def __init__(self, knowledge_path: str = "data/knowledge.json",
                 fuzzy_threshold: float = 0.84):
        with open(knowledge_path, encoding="utf-8") as f:
            self.kb = json.load(f)
        self.threshold = fuzzy_threshold

    @cached_property
    def place_keys(self) -> list[str]:
        return list(self.kb.get("places", {}).keys())

    @cached_property
    def plate_codes(self) -> set[str]:
        return set(self.kb.get("plate_codes", {}).keys())

    def extract(self, text: str) -> list[Evidence]:
        if not text:
            return []
        return self._places(text) + self._plate_codes(text) + self._patterns(text)

    # -- place names (fuzzy) --
    def _places(self, text: str) -> list[Evidence]:
        words = re.findall(r"[A-Za-z]+", text)
        ngrams = set()
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                ngrams.add(" ".join(words[i:i + n]))
        if not ngrams:
            return []
        found: dict[str, float] = {}
        for key in self.place_keys:
            kl = key.lower()
            best = max((SequenceMatcher(None, kl, g.lower()).ratio() for g in ngrams), default=0.0)
            if best >= self.threshold:
                found[key] = best
        return [Evidence("place", k, round(v, 3), "entities") for k, v in found.items()]

    # -- vehicle plate state codes --
    def _plate_codes(self, text: str) -> list[Evidence]:
        up = text.upper()
        hits: dict[str, Evidence] = {}
        for m in re.finditer(r"\b([A-Z]{2})[\s-]?\d{1,2}[\s-]?[A-Z]{0,3}[\s-]?\d{1,4}\b", up):
            code = m.group(1)
            if code in self.plate_codes:
                hits[code] = Evidence("plate_code", code, 0.9, "entities")
        return list(hits.values())

    # -- format patterns --
    def _patterns(self, text: str) -> list[Evidence]:
        out = []
        compact = re.sub(r"[\s\-()]", "", text)
        if "\u20b9" in text or re.search(r"\bRs\.?\b", text, re.I) or re.search(r"\bINR\b", text, re.I):
            out.append(Evidence("pattern", "inr_currency", 0.7, "entities"))
        if re.search(r"(\+?91)?\b[6-9]\d{9}\b", compact):
            out.append(Evidence("pattern", "indian_phone", 0.85, "entities"))
        if re.search(r"\b\d{6}\b", text):
            out.append(Evidence("pattern", "indian_pincode", 0.4, "entities"))
        return out


# --- standalone test --------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    kb = os.path.join(here, "..", "data", "knowledge.json")
    if not os.path.exists(kb):
        kb = "data/knowledge.json"

    text = sys.argv[1] if len(sys.argv) > 1 else "Venkataramana Colour Lab, Hyderabad. Rs 200. +91 9876543210"
    ex = EntityExtractor(kb)
    print(f"Text: {text!r}\n")
    found = ex.extract(text)
    if not found:
        print("No location-bearing entities found.")
    for e in found:
        print(f"  {e.category:11} {e.key:16} strength={e.strength}")