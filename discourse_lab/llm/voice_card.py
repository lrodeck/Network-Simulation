"""Voice cards (spec §2.10): one call per user, cached forever, keyed by
`hash(archetype, quantized_traits)` so similar users share cards and the
cache hit rate stays high.

    input:  trait vector rendered as a labeled feature list
    output: 3-sentence persona + 3 concrete writing tics + register notes
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from discourse_lab.config import structural_hash
from discourse_lab.io.workspace import voice_cards_dir
from discourse_lab.llm.client import LLMClient
from discourse_lab.llm.quantize import Bands
from discourse_lab.population import Population

# The traits a voice card is conditioned on: personality + meta (identity and
# temperament, not moment-to-moment content) plus the two behavior traits
# that most shape how someone writes.
VOICE_CARD_TRAITS = (
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism",
    "plasticity", "conviction",
    "contrarianism", "credulity",
)

SYSTEM_PROMPT = (
    "You write short persona notes for a social-media simulation. Given a "
    "labeled trait profile, invent one consistent, specific voice for a "
    "person with those traits. Never mention the trait labels themselves."
)


@dataclass
class VoiceCard:
    persona: str
    tics: tuple[str, ...]
    register: str
    raw_text: str

    def to_dict(self) -> dict:
        return {"persona": self.persona, "tics": list(self.tics), "register": self.register, "raw_text": self.raw_text}

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceCard":
        return cls(persona=d["persona"], tics=tuple(d["tics"]), register=d["register"], raw_text=d["raw_text"])


def voice_card_key(archetype: str, bands: dict[str, str]) -> str:
    return structural_hash({"archetype": archetype, "bands": bands})


def fit_bands(pop: Population, n_bands: int = 5) -> Bands:
    cols = {name: pop.X_used[:, pop.trait_names.index(name)] for name in VOICE_CARD_TRAITS}
    return Bands.fit(cols, n_bands=n_bands)


def user_bands(pop: Population, bands: Bands, user_idx: int) -> dict[str, str]:
    values = {name: float(pop.X_used[user_idx, pop.trait_names.index(name)]) for name in VOICE_CARD_TRAITS}
    return bands.label_row(values)


def build_voice_card_messages(archetype: str, bands: dict[str, str]) -> list[dict]:
    feature_list = "\n".join(f"- {name}: {label}" for name, label in bands.items())
    user_prompt = (
        f"Archetype: {archetype}\n\nTrait profile:\n{feature_list}\n\n"
        "Return exactly this format, nothing else:\n"
        "PERSONA: <three sentences describing who this person is and how they post>\n"
        "TICS: <tic one>; <tic two>; <tic three>\n"
        "REGISTER: <one line on formality/vocabulary/punctuation habits>"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def parse_voice_card(text: str) -> VoiceCard:
    persona_m = re.search(r"PERSONA:\s*(.+?)(?:\n[A-Z]+:|\Z)", text, re.DOTALL)
    tics_m = re.search(r"TICS:\s*(.+?)(?:\n[A-Z]+:|\Z)", text, re.DOTALL)
    register_m = re.search(r"REGISTER:\s*(.+?)(?:\n[A-Z]+:|\Z)", text, re.DOTALL)

    persona = persona_m.group(1).strip() if persona_m else text.strip()
    tics = tuple(t.strip() for t in tics_m.group(1).split(";") if t.strip()) if tics_m else ()
    register = register_m.group(1).strip() if register_m else ""
    return VoiceCard(persona=persona, tics=tics, register=register, raw_text=text)


def get_voice_card(
    client: LLMClient,
    archetype: str,
    bands: dict[str, str],
    cache_dir: Path | None = None,
    temperature: float = 0.8,
    max_tokens: int = 220,
) -> VoiceCard:
    """Cached forever on disk, keyed by `hash(archetype, quantized_traits)`."""
    cache_dir = cache_dir or voice_cards_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{voice_card_key(archetype, bands)}.json"

    if path.exists():
        return VoiceCard.from_dict(json.loads(path.read_text(encoding="utf-8")))

    messages = build_voice_card_messages(archetype, bands)
    text = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    card = parse_voice_card(text)
    path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    return card
