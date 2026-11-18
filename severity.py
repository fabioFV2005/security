from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

DEFAULT_WEIGHTS = {
    "person": 1,
    "face_mask": 1,
    "face_covered": 3,
    "night": 1,
    "zone": 1,
    "dwell": 1,
    "group_covered": 1,
}


@dataclass
class Person:
    track_id: int
    box: Tuple[int, int, int, int]
    conf: float
    face: str = "unknown"
    face_conf: float = 0.0
    in_zone: bool = False
    dwell: float = 0.0


@dataclass
class Assessment:
    level: int
    score: int
    event: str
    reasons: List[str] = field(default_factory=list)
    worst: Optional[Person] = None

    @property
    def name(self):
        return LEVELS[self.level]


def level_index(name):
    return LEVELS.index(name.upper())


def score_to_level(score):
    if score <= 0:
        return 0
    if score == 1:
        return 1
    if score == 2:
        return 2
    if score <= 4:
        return 3
    return 4


def is_night(hour, start, end):
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


class SeverityEngine:
    def __init__(self, cfg, zone_enabled):
        self.night_hours = cfg.get("night_hours", [22, 6])
        self.dwell_seconds = cfg.get("dwell_seconds", 20)
        self.weights = {**DEFAULT_WEIGHTS, **(cfg.get("weights") or {})}
        self.zone_enabled = zone_enabled

    def assess(self, people, now=None):
        if not people:
            return Assessment(0, 0, "none")
        now = now or datetime.now()
        w = self.weights
        night = is_night(now.hour, *self.night_hours)
        covered_count = sum(1 for p in people if p.face == "face_covered")

        best = None
        for p in people:
            score = w["person"]
            reasons = ["person"]
            if p.face == "face_covered":
                score += w["face_covered"]
                reasons.append("face covered")
            elif p.face == "face_mask":
                score += w["face_mask"]
                reasons.append("face mask")
            if night:
                score += w["night"]
                reasons.append("night time")
            if self.zone_enabled and p.in_zone:
                score += w["zone"]
                reasons.append("restricted zone")
            if p.dwell >= self.dwell_seconds:
                score += w["dwell"]
                reasons.append(f"loitering {int(p.dwell)}s")
            if covered_count >= 2 and p.face == "face_covered":
                score += w["group_covered"]
                reasons.append(f"{covered_count} people with covered faces")
            if best is None or score > best[0]:
                best = (score, reasons, p)

        score, reasons, worst = best
        event = "person_face_covered" if covered_count else "person"
        return Assessment(score_to_level(score), score, event, reasons, worst)
