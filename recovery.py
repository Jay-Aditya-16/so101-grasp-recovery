"""Inspectable grasp-region ranking; no robot motion or vision inference."""
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    region: str
    width_mm: float
    stability: float  # estimated from visible geometry, 0..1
    reachable: bool


@dataclass(frozen=True)
class Experience:
    object_class: str
    region: str
    success: bool


class Memory:
    def __init__(self, experiences=()):
        self.experiences = list(experiences)

    def add(self, experience):
        self.experiences.append(experience)

    def save(self, path):
        Path(path).write_text(json.dumps([vars(e) for e in self.experiences], indent=2))

    @classmethod
    def load(cls, path):
        return cls(Experience(**item) for item in json.loads(Path(path).read_text()))

    def rank(self, object_class, candidates, max_width_mm, failed_region=None):
        """Return feasible candidates and transparent scores. Similarity is exact class only."""
        if max_width_mm <= 0:
            raise ValueError('max_width_mm must be positive')
        counts = defaultdict(lambda: [0, 0])
        for e in self.experiences:
            if e.object_class == object_class:
                counts[e.region][0] += int(e.success)
                counts[e.region][1] += 1
        ranked = []
        for c in candidates:
            if not 0 <= c.stability <= 1:
                raise ValueError('stability must be between 0 and 1')
            if not c.reachable or not 0 < c.width_mm <= max_width_mm:
                continue
            successes, total = counts[c.region]
            # Beta(1,1) prior; failures lower the estimate immediately.
            history = (successes + 1) / (total + 2)
            score = c.stability + history - (0.5 if c.region == failed_region else 0)
            ranked.append({'region': c.region, 'score': score, 'history_successes': successes,
                           'history_trials': total, 'current_failure': c.region == failed_region})
        return sorted(ranked, key=lambda item: (-item['score'], item['region']))
