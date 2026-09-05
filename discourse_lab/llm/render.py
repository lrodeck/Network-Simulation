"""Rendering (spec §2.10): batched, 20-50 posts per call.

    input:  voice_card, post dims (as labeled bands: "arousal: high"),
            thread context (parent chain, truncated), topic label
    output: post text only

Lazy realization: render only the subgraph a human will read (a run of 10^6
interactions can be fully analyzed numerically; text is generated only for
the few thousand posts actually inspected).
"""

from __future__ import annotations

import re

from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.llm.client import LLMClient
from discourse_lab.llm.quantize import Bands
from discourse_lab.llm.voice_card import VoiceCard

POST_DIMS_FOR_RENDER = ("arousal", "valence", "provocativeness", "novelty", "specificity", "quality")

SYSTEM_PROMPT = (
    "You write short social-media posts for a research simulation. Each item "
    "gives a voice (persona, tics, register), labeled content dials, and "
    "thread context. Write only the post text in that voice — never mention "
    "the labels, never add commentary."
)


def fit_post_bands(posts: PostBatch, n_bands: int = 5) -> Bands:
    cols = {dim: getattr(posts, dim) for dim in POST_DIMS_FOR_RENDER}
    return Bands.fit(cols, n_bands=n_bands)


def _thread_context(posts: PostBatch, i: int, topic_names: list[str] | None) -> str:
    topic = topic_names[posts.topic[i]] if topic_names else f"topic {posts.topic[i]}"
    if posts.parent[i] == -1:
        return f"A new root post about {topic}."
    return f"A {posts.kind[i]} at depth {posts.depth[i]} in a thread about {topic}."


def build_render_messages(
    posts: PostBatch,
    indices: list[int],
    voice_cards: dict[int, VoiceCard],
    bands: Bands,
    topic_names: list[str] | None = None,
) -> list[dict]:
    items = []
    for item_no, i in enumerate(indices, start=1):
        card = voice_cards[int(posts.author[i])]
        dims = "; ".join(f"{dim}: {bands.label(dim, getattr(posts, dim)[i])}" for dim in POST_DIMS_FOR_RENDER)
        items.append(
            f"### Item {item_no}\n"
            f"Voice: {card.persona} Tics: {', '.join(card.tics)}. Register: {card.register}\n"
            f"Content dials: {dims}\n"
            f"Context: {_thread_context(posts, i, topic_names)}"
        )
    user_prompt = (
        "\n\n".join(items)
        + "\n\nFor each item, output one line formatted exactly as `<item number>: <post text>` "
        "and nothing else."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def parse_render_batch(text: str, n: int) -> dict[int, str]:
    """Maps 1-based item number (as sent to the model) -> rendered text."""
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.]\s*(.+)$", line)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n:
                out[idx] = m.group(2).strip()
    return out


def render_batch(
    client: LLMClient,
    posts: PostBatch,
    indices: list[int],
    voice_cards: dict[int, VoiceCard],
    bands: Bands,
    topic_names: list[str] | None = None,
    temperature: float = 0.8,
    max_tokens: int = 120,
) -> dict[int, str]:
    """Renders one call's worth of posts (spec: 20-50 per call). Returns
    `{post_id: text}` for whichever items the model actually answered.
    """
    if not indices:
        return {}
    messages = build_render_messages(posts, indices, voice_cards, bands, topic_names)
    text = client.chat(messages, temperature=temperature, max_tokens=max_tokens * len(indices))
    by_item = parse_render_batch(text, len(indices))
    return {int(posts.id[indices[item_no - 1]]): rendered for item_no, rendered in by_item.items()}


def render_posts(
    client: LLMClient,
    posts: PostBatch,
    voice_cards: dict[int, VoiceCard],
    post_ids: list[int] | None = None,
    topic_names: list[str] | None = None,
    batch_size: int = 30,
    temperature: float = 0.8,
    max_tokens: int = 120,
) -> dict[int, str]:
    """Lazy realization entry point: render only `post_ids` (default: every
    post in `posts`), batched.
    """
    if post_ids is None:
        indices = list(range(len(posts)))
    else:
        id_to_index = {int(pid): i for i, pid in enumerate(posts.id)}
        indices = [id_to_index[pid] for pid in post_ids if pid in id_to_index]

    bands = fit_post_bands(posts)
    results: dict[int, str] = {}
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        results.update(render_batch(client, posts, batch, voice_cards, bands, topic_names, temperature, max_tokens))
    return results
