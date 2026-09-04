"""Regenerates discourse_lab/data/scenarios/default.json.

Produces exactly the JSON schema the stance editor emits: density normalised
to bin width so it integrates to 1 over [-1, 1]. Run once and commit.
"""

import json
import sys
from pathlib import Path

BINS = 128


def bin_centre(i):
    return -1 + (2 * (i + 0.5)) / BINS


def gauss_mix(components):
    d = []
    for i in range(BINS):
        x = bin_centre(i)
        v = sum((w * pow(2.718281828459045, -((x - mu) ** 2) / (2 * sd * sd))) / sd
                for mu, sd, w in components)
        d.append(v)
    m = max(d)
    return [v / m for v in d]


def normalise(density, floor):
    p = [max(v, floor) for v in density]
    s = sum(p)
    return [v / s * BINS for v in p]  # integrates to 1 over [-1, 1]


PRESETS = {
    "majority_left": gauss_mix([[-0.42, 0.32, 1.0], [0.55, 0.22, 0.28]]),
    "polarized": gauss_mix([[-0.6, 0.2, 1.0], [0.6, 0.2, 1.0]]),
    "symmetric": gauss_mix([[0.0, 0.34, 1.0]]),
}


def axis(name, pole_neg, pole_pos, density, floor, cost_neg, cost_pos):
    p = normalise(density, floor)
    mean = sum(p[i] / BINS * bin_centre(i) for i in range(BINS))
    var = sum(p[i] / BINS * (bin_centre(i) - mean) ** 2 for i in range(BINS))
    sd = var ** 0.5
    below = sum(p[i] / BINS for i in range(BINS) if bin_centre(i) < 0)
    return {
        "name": name,
        "pole_neg": pole_neg,
        "pole_pos": pole_pos,
        "marginal": {"kind": "empirical", "bins": BINS, "support": [-1, 1],
                     "density": [round(v, 5) for v in p]},
        "expression_cost": {"neg": cost_neg, "pos": cost_pos},
        "derived": {"mean": round(mean, 4), "sd": round(sd, 4), "skew": 0.0,
                    "share_neg": round(below, 4)},
    }


def main():
    axes = [
        axis("provision", "market", "state", PRESETS["majority_left"], 0.004, 0.0, 0.35),
        axis("openness", "closed", "open", PRESETS["polarized"], 0.004, 0.3, 0.1),
        axis("institutional trust", "distrust", "trust", PRESETS["symmetric"], 0.004, 0.0, 0.0),
    ]
    data = {"scenario": {"stance_axes": axes}}
    out = Path(__file__).resolve().parents[1] / "discourse_lab" / "data" / "scenarios" / "default.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
