"""LLM realization (spec §2.10, dev §6 step 12) — offline only, never inside
the tick. Voice cards + batched rendering are the text-generation pass;
channel-3 adjudication (spec §2.9) is the only place an LLM touches
dynamics, gated and applied explicitly, never automatically.
"""

from discourse_lab.llm.adjudication import (
    AdjudicationResult,
    SalientEvent,
    apply_adjudication,
    detect_salient_events,
    request_adjudication,
)
from discourse_lab.llm.client import LLMClient, OllamaCloudClient
from discourse_lab.llm.realize import realize
from discourse_lab.llm.render import render_posts
from discourse_lab.llm.voice_card import VoiceCard, get_voice_card

__all__ = [
    "OllamaCloudClient",
    "LLMClient",
    "realize",
    "render_posts",
    "VoiceCard",
    "get_voice_card",
    "SalientEvent",
    "AdjudicationResult",
    "detect_salient_events",
    "request_adjudication",
    "apply_adjudication",
]
