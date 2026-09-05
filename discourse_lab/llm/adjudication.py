"""Channel 3 — LLM adjudication (spec §2.9, §2.10): rare, event-triggered,
the only place the LLM touches dynamics. Runs offline over a completed run
(never inside the tick, same as realization) and returns bounded deltas a
caller may fold into a later run's drift via `clip(Delta_llm, -eps, eps)`;
it does not mutate anything by itself except through `apply_adjudication`,
called explicitly.

Salient events (queued, not auto-applied): a post in the top 1% of
engagement, or a pile-on (more than `pile_on_threshold` hostile replies).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.llm.client import LLMClient
from discourse_lab.llm.voice_card import VoiceCard
from discourse_lab.population import Population

SYSTEM_PROMPT = (
    "You adjudicate a rare, high-salience event in a social-media simulation. "
    "Given a person's voice and a digest of what just happened to them, decide "
    "whether and how their underlying traits should shift as a result. Be "
    "conservative: most events justify a small nudge, not a large one."
)


@dataclass(frozen=True)
class SalientEvent:
    post_id: int
    author: int
    kind: str  # "top_engagement" | "pile_on"
    detail: dict


@dataclass
class AdjudicationResult:
    deltas: dict[str, float]
    justification: str


def detect_salient_events(
    posts: PostBatch,
    reply_counts: np.ndarray,
    top_percentile: float = 0.99,
    pile_on_threshold: int = 20,
) -> list[SalientEvent]:
    """`reply_counts` is reply-only engagement per post (aligned with
    `posts`), tracked separately from `posts.engagement_count` (which sums
    every action type) since a pile-on is specifically about replies.
    """
    events: list[SalientEvent] = []
    if len(posts) == 0:
        return events

    threshold = np.quantile(posts.engagement_count, top_percentile) if len(posts) > 1 else np.inf
    for i in range(len(posts)):
        if posts.engagement_count[i] >= threshold and posts.engagement_count[i] > 0:
            events.append(
                SalientEvent(
                    post_id=int(posts.id[i]), author=int(posts.author[i]), kind="top_engagement",
                    detail={"engagement_count": int(posts.engagement_count[i])},
                )
            )
        if reply_counts[i] > pile_on_threshold:
            events.append(
                SalientEvent(
                    post_id=int(posts.id[i]), author=int(posts.author[i]), kind="pile_on",
                    detail={"reply_count": int(reply_counts[i])},
                )
            )
    return events


def build_adjudication_messages(voice_card: VoiceCard, event: SalientEvent, trait_names: list[str]) -> list[dict]:
    if event.kind == "top_engagement":
        digest = f"One of their posts drew unusually high engagement ({event.detail['engagement_count']} actions)."
    elif event.kind == "pile_on":
        digest = f"One of their posts is being piled on with {event.detail['reply_count']} hostile replies."
    else:
        raise ValueError(f"unknown salient event kind: {event.kind!r}")

    prompt = (
        f"Voice: {voice_card.persona}\n\nEvent: {digest}\n\n"
        f"Traits available to adjust: {', '.join(trait_names)}.\n"
        "Return exactly one JSON object, nothing else, of the form:\n"
        '{"trait_deltas": {"<trait>": <small float, positive or negative>, ...}, "justification": "<one sentence>"}\n'
        "Include only traits that should actually move; omit the rest."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def parse_adjudication(text: str, trait_names: list[str], max_delta: float) -> AdjudicationResult:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return AdjudicationResult(deltas={}, justification="")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return AdjudicationResult(deltas={}, justification="")

    raw_deltas = data.get("trait_deltas", {})
    deltas = {
        name: float(np.clip(value, -max_delta, max_delta))
        for name, value in raw_deltas.items()
        if name in trait_names
    }
    return AdjudicationResult(deltas=deltas, justification=str(data.get("justification", "")))


def request_adjudication(
    client: LLMClient,
    voice_card: VoiceCard,
    event: SalientEvent,
    trait_names: list[str],
    max_delta: float = 0.1,
    temperature: float = 0.4,
) -> AdjudicationResult:
    messages = build_adjudication_messages(voice_card, event, trait_names)
    text = client.chat(messages, temperature=temperature, max_tokens=200)
    return parse_adjudication(text, trait_names, max_delta)


def apply_adjudication(pop: Population, user_idx: int, result: AdjudicationResult) -> None:
    """Applied to *stored* traits, same discipline as every other drift
    channel (spec §1.1) — folds in as the `clip(Delta_llm, ...)` term the
    next time `dynamics.drift.apply_drift` runs, if a caller chooses to wire
    it in; not called automatically anywhere.
    """
    from discourse_lab.population.links import to_used

    for trait, delta in result.deltas.items():
        col = pop.trait_names.index(trait)
        pop.X_stored[user_idx, col] += delta
        pop.X_used[user_idx, col] = to_used(pop.X_stored[user_idx : user_idx + 1, col], pop.links[col])[0]
