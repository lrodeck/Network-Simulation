"""Realization entry point (spec §2.10): a view on a completed run, not part
of it. Builds (or reuses cached) voice cards for every author touched, then
renders the requested posts in batches. Wiring this to a persisted `Run`'s
posts (once posts.parquet exists, dev §5) is a follow-up — today it takes a
`PostBatch` and `Population` directly, which is exactly what a live
`run_iter` session or an in-memory analysis already has on hand.
"""

from __future__ import annotations

from pathlib import Path

from discourse_lab.config import Config
from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.llm.client import LLMClient
from discourse_lab.llm.render import render_posts
from discourse_lab.llm.voice_card import fit_bands, get_voice_card, user_bands
from discourse_lab.population import Population


def realize(
    client: LLMClient,
    cfg: Config,
    posts: PostBatch,
    pop: Population,
    post_ids: list[int] | None = None,
    topic_names: list[str] | None = None,
    cache_dir: Path | None = None,
) -> dict[int, str]:
    """Render `post_ids` (default: every post in `posts`) into text, lazily
    fetching/caching one voice card per author actually touched.
    """
    world = cfg.world
    if post_ids is None:
        authors = set(int(a) for a in posts.author)
    else:
        wanted = set(post_ids)
        authors = {int(a) for a, pid in zip(posts.author, posts.id) if int(pid) in wanted}

    bands_fitted = fit_bands(pop, n_bands=world.n_bands)
    archetype_of = dict(zip(range(len(pop.archetype_labels)), pop.archetype_labels))

    voice_cards = {}
    for author in authors:
        archetype = pop.archetype_names[archetype_of[author]]
        bands = user_bands(pop, bands_fitted, author)
        voice_cards[author] = get_voice_card(
            client, archetype, bands, cache_dir=cache_dir, temperature=world.temperature,
            max_tokens=world.voice_card_max_tokens,
        )

    return render_posts(
        client, posts, voice_cards, post_ids=post_ids, topic_names=topic_names,
        batch_size=world.render_batch_size, temperature=world.temperature, max_tokens=world.render_max_tokens,
    )
