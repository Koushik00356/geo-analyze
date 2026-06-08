"""
core/reasoning.py - Turn evidence into ranked, explained location guesses.

The heart of the project: it does NOT classify, it *reasons*. It takes a bag
of weak signals (scripts, objects, place names, plate codes, format patterns)
and asks "what location best explains all of this together?" - always
returning evidence and confidence, never a bare answer, always keeping
alternatives.

Scoring (log-likelihood):
  Each hypothesis h scores  sum over evidence of  strength * LLR(evidence, h),
  LLR = log(1 + d * w / (1 - d)), d = distinctiveness, w = how strongly the
  signal points at h. Scores -> softmax (with an always-present "Elsewhere"
  at score 0) -> confidence. A signal that doesn't point at h adds exactly 0,
  so generic signals can never move the answer.

Evidence categories map to knowledge.json sections by pluralising the
category name (script -> scripts, place -> places, plate_code -> plate_codes,
...). A new evidence type therefore needs only a new KB section - no engine
changes.

Quick test:
    python core/reasoning.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

OTHER_COUNTRY = "Elsewhere"
OTHER_REGION = "Elsewhere (unspecified)"
OTHER_CITY = "Elsewhere (unspecified)"
_MAX_DISTINCT = 0.98


def _quality(strength: float) -> str:
    return "strong" if strength >= 0.8 else "moderate" if strength >= 0.5 else "weak"


def _llr(distinctiveness: float, weight: float) -> float:
    if weight <= 0:
        return 0.0
    d = min(max(distinctiveness, 0.0), _MAX_DISTINCT)
    return math.log(1.0 + d * weight / (1.0 - d))


@dataclass
class Evidence:
    category: str
    key: str
    strength: float
    source: str = ""

    def describe(self) -> str:
        return f"{self.key} {self.category} ({_quality(self.strength)})"


@dataclass
class Hypothesis:
    label: str
    confidence: float
    support: list[str] = field(default_factory=list)


@dataclass
class LocationResult:
    country: list[Hypothesis]
    region: list[Hypothesis]
    city: list[Hypothesis]
    evidence: list[Evidence]

    @property
    def top_country(self) -> Hypothesis | None:
        return self.country[0] if self.country else None

    def _block(self, title, hyps, n, width):
        if not hyps:
            return [f"{title}: (no signal)"]
        out = [f"{title}:"]
        for h in hyps[:n]:
            sup = ", ".join(h.support) or "(residual uncertainty)"
            out.append(f"  {h.label:<{width}} {h.confidence:5.0%}   <- {sup}")
        return out

    def summary(self) -> str:
        lines = ["LOCATION HYPOTHESES", "=" * 40]
        if not self.country:
            lines.append("No location-bearing evidence found.")
            return "\n".join(lines)
        lines += self._block("Country", self.country, 3, 18)
        lines += self._block("Region / state", self.region, 4, 26)
        if self.city:
            lines += self._block("City", self.city, 3, 26)
        else:
            lines.append("City: insufficient evidence from current signals.")
        return "\n".join(lines)


class ReasoningEngine:
    def __init__(self, knowledge_path: str = "data/knowledge.json"):
        with open(knowledge_path, encoding="utf-8") as f:
            self.kb = json.load(f)

    def from_script_counts(self, counts: dict[str, int], source: str = "ocr") -> list[Evidence]:
        return [Evidence("script", s, min(1.0, n / 10.0), source) for s, n in counts.items()]

    def from_ocr(self, ocr_result) -> list[Evidence]:
        return self.from_script_counts(ocr_result.script_counts)

    def reason(self, evidence: list[Evidence]) -> LocationResult:
        scores = {"country": defaultdict(float), "region": defaultdict(float), "city": defaultdict(float)}
        support = {"country": defaultdict(list), "region": defaultdict(list), "city": defaultdict(list)}
        seen = {"country": False, "region": False, "city": False}

        for ev in evidence:
            section = self.kb.get(ev.category + "s", {})   # pluralise -> KB section
            entry = section.get(ev.key)
            if not entry:
                continue
            d = entry.get("distinctiveness", 0.0)
            best = {"country": {}, "region": {}, "city": {}}
            for cand in entry.get("candidates", []):
                c = cand["country"]
                w = cand.get("weight", 1.0)
                best["country"][c] = max(best["country"].get(c, 0.0), w)
                if cand.get("region"):
                    rk = f"{cand['region']}, {c}"
                    best["region"][rk] = max(best["region"].get(rk, 0.0), w)
                if cand.get("city"):
                    ck = f"{cand['city']}, {c}"
                    best["city"][ck] = max(best["city"].get(ck, 0.0), w)
            for level in ("country", "region", "city"):
                for key, w in best[level].items():
                    contrib = ev.strength * _llr(d, w)
                    if contrib > 0:
                        seen[level] = True
                        scores[level][key] += contrib
                        support[level][key].append((ev, contrib))

        return LocationResult(
            country=self._rank(scores["country"], support["country"], OTHER_COUNTRY) if seen["country"] else [],
            region=self._rank(scores["region"], support["region"], OTHER_REGION) if seen["region"] else [],
            city=self._rank(scores["city"], support["city"], OTHER_CITY) if seen["city"] else [],
            evidence=evidence,
        )

    @staticmethod
    def _rank(scores, supports, other_label) -> list[Hypothesis]:
        scores = dict(scores)
        scores.setdefault(other_label, 0.0)
        mx = max(scores.values())
        exps = {k: math.exp(v - mx) for k, v in scores.items()}
        z = sum(exps.values())
        hyps = []
        for k, s in scores.items():
            contribs = sorted(supports.get(k, []), key=lambda t: t[1], reverse=True)
            seen_d, sup = set(), []
            for ev, _ in contribs:
                desc = ev.describe()
                if desc not in seen_d:
                    seen_d.add(desc)
                    sup.append(desc)
            hyps.append(Hypothesis(k, exps[k] / z, sup))
        return sorted(hyps, key=lambda h: h.confidence, reverse=True)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(here, "..", "data", "knowledge.json")
    if not os.path.exists(kb_path):
        kb_path = "data/knowledge.json"
    engine = ReasoningEngine(kb_path)
    counts = {"Latin": 81, "Telugu": 35}
    print(f"Input scripts: {counts}\n")
    print(engine.reason(engine.from_script_counts(counts)).summary())