"""Step 12 verification (dev §6): world config + band quantization + voice
cards + batched, lazy rendering + gated channel-3 adjudication. No real
network calls — a FakeLLMClient stands in for Ollama Cloud so these tests
are deterministic and offline; OllamaCloudClient itself is only checked for
its fast-fail-without-a-key behavior (this sandbox's egress policy denies
ollama.com anyway, so a live-network test wouldn't run here regardless).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from discourse_lab.config import Config
from discourse_lab.dynamics import ExpressionMap, generate_posts
from discourse_lab.llm.adjudication import (
    apply_adjudication,
    build_adjudication_messages,
    detect_salient_events,
    parse_adjudication,
    request_adjudication,
)
from discourse_lab.llm.client import LLMError, OllamaCloudClient
from discourse_lab.llm.quantize import Bands, band_labels
from discourse_lab.llm.realize import realize
from discourse_lab.llm.render import build_render_messages, fit_post_bands, parse_render_batch, render_posts
from discourse_lab.llm.voice_card import (
    build_voice_card_messages,
    fit_bands,
    get_voice_card,
    parse_voice_card,
    user_bands,
    voice_card_key,
)
from discourse_lab.population import sample_population


class FakeLLMClient:
    """Deterministic stand-in for Ollama Cloud: `respond` maps a call index
    to canned text, so tests can assert exactly how many calls were made.
    """

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, temperature=0.7, max_tokens=None):
        self.calls.append(messages)
        if len(self.calls) > len(self.responses):
            raise AssertionError("FakeLLMClient called more times than it has canned responses")
        return self.responses[len(self.calls) - 1]


def _setup(n_users=60, n_posts=12, seed=0):
    cfg = dataclasses.replace(Config(), population=dataclasses.replace(Config().population, n_users=n_users))
    rng = np.random.default_rng(seed)
    pop = sample_population(cfg, rng)
    K, D = cfg.population.n_topics, cfg.stance_dims()
    expr = ExpressionMap.build(pop.trait_names, K)
    authors = rng.choice(n_users, size=n_posts, replace=True)
    posts = generate_posts(authors, pop, expr, np.zeros(K), np.zeros((K, D)), eta=0.3, rng=rng)
    return cfg, rng, pop, posts


# --------------------------------------------------------------------------
# band quantization
# --------------------------------------------------------------------------


def test_bands_label_monotonically():
    values = {"x": np.linspace(0, 1, 1000)}
    bands = Bands.fit(values, n_bands=5)
    assert bands.label("x", 0.0) == "very low"
    assert bands.label("x", 0.5) in ("medium", "low", "high")
    assert bands.label("x", 1.0) == "very high"
    assert band_labels(5) == ("very low", "low", "medium", "high", "very high")


def test_bands_are_relative_to_the_population():
    """The whole point of quantile bands: "high" means high for this
    population, not an arbitrary fixed cutoff.
    """
    low_pop = {"x": np.linspace(0, 1, 1000)}
    shifted_pop = {"x": np.linspace(100, 101, 1000)}
    bands_low = Bands.fit(low_pop)
    bands_shifted = Bands.fit(shifted_pop)
    assert bands_low.label("x", 0.9) == "very high"
    assert bands_shifted.label("x", 0.9) == "very low"  # same raw value, opposite band


# --------------------------------------------------------------------------
# voice cards
# --------------------------------------------------------------------------


def test_voice_card_key_is_content_addressed():
    bands_a = {"openness": "high", "neuroticism": "low"}
    bands_b = {"openness": "low", "neuroticism": "low"}
    assert voice_card_key("poster", bands_a) == voice_card_key("poster", bands_a)
    assert voice_card_key("poster", bands_a) != voice_card_key("poster", bands_b)
    assert voice_card_key("poster", bands_a) != voice_card_key("lurker", bands_a)


def test_parse_voice_card_extracts_all_three_fields():
    text = "PERSONA: Blunt and online too much.\nTICS: em dashes; one-word replies; sics\nREGISTER: lowercase, no punctuation"
    card = parse_voice_card(text)
    assert "Blunt" in card.persona
    assert card.tics == ("em dashes", "one-word replies", "sics")
    assert "lowercase" in card.register


def test_get_voice_card_caches_and_does_not_recall_the_client(tmp_path):
    client = FakeLLMClient(["PERSONA: A person.\nTICS: a; b; c\nREGISTER: plain"])
    bands = {"openness": "medium"}

    card1 = get_voice_card(client, "poster", bands, cache_dir=tmp_path)
    card2 = get_voice_card(client, "poster", bands, cache_dir=tmp_path)  # should hit the cache

    assert len(client.calls) == 1
    assert card1.persona == card2.persona


def test_voice_card_prompt_never_leaks_raw_floats():
    _, _, pop, _ = _setup()
    bands_fitted = fit_bands(pop)
    bands = user_bands(pop, bands_fitted, 0)
    messages = build_voice_card_messages("poster", bands)
    prompt = messages[1]["content"]
    # every value in the prompt is a verbal label, never something that
    # parses as a bare float
    for line in prompt.splitlines():
        if ":" in line and line.strip().startswith("-"):
            value = line.split(":", 1)[1].strip()
            with pytest.raises(ValueError):
                float(value)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_render_batch_is_lazy_only_requested_posts_get_rendered(tmp_path):
    cfg, rng, pop, posts = _setup(n_posts=20)
    card_response = "PERSONA: A poster.\nTICS: a; b; c\nREGISTER: plain"
    render_response = "1: hello world\n2: totally agree"

    unique_authors = sorted(set(int(a) for a in posts.author[:2]))
    client = FakeLLMClient([card_response] * len(unique_authors) + [render_response])

    wanted_ids = [int(posts.id[0]), int(posts.id[1])]
    results = realize(client, cfg, posts, pop, post_ids=wanted_ids, cache_dir=tmp_path)

    assert set(results.keys()) == set(wanted_ids)
    assert len(results) == 2  # not all 20 posts were rendered


def test_render_batch_respects_batch_size_and_parses_multiple_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("DLAB_HOME", str(tmp_path))
    cfg, rng, pop, posts = _setup(n_posts=10)
    unique_authors = sorted(set(int(a) for a in posts.author))
    card_responses = ["PERSONA: A poster.\nTICS: a; b; c\nREGISTER: plain"] * len(unique_authors)

    # batch size 4 over 10 posts -> 3 calls
    render_responses = [
        "\n".join(f"{i + 1}: text-{i}" for i in range(4)),
        "\n".join(f"{i + 1}: text-{i + 4}" for i in range(4)),
        "\n".join(f"{i + 1}: text-{i + 8}" for i in range(2)),
    ]
    client = FakeLLMClient(card_responses + render_responses)
    world = dataclasses.replace(cfg.world, render_batch_size=4)
    cfg2 = dataclasses.replace(cfg, world=world)

    results = realize(client, cfg2, posts, pop)
    assert len(results) == 10
    assert len(client.calls) == len(unique_authors) + 3


def test_parse_render_batch_ignores_malformed_lines():
    text = "1: a real post\nsome noise the model added\n2: another real post"
    parsed = parse_render_batch(text, 2)
    assert parsed == {1: "a real post", 2: "another real post"}


# --------------------------------------------------------------------------
# channel 3: adjudication (offline, gated, never auto-applied)
# --------------------------------------------------------------------------


def test_detect_salient_events_flags_top_engagement_and_pile_on():
    cfg, rng, pop, posts = _setup(n_posts=100)
    posts.engagement_count[:] = rng.poisson(2, size=100)
    posts.engagement_count[0] = 1000  # unambiguously top 1%
    reply_counts = np.zeros(100, dtype=int)
    reply_counts[1] = 50  # pile-on

    events = detect_salient_events(posts, reply_counts, top_percentile=0.99, pile_on_threshold=20)
    kinds_by_post = {e.post_id: e.kind for e in events}
    assert kinds_by_post[int(posts.id[0])] == "top_engagement"
    assert kinds_by_post[int(posts.id[1])] == "pile_on"


def test_adjudication_deltas_are_clipped_to_max_delta():
    text = '{"trait_deltas": {"plasticity": 5.0, "conviction": -5.0}, "justification": "big shift"}'
    result = parse_adjudication(text, ["plasticity", "conviction"], max_delta=0.1)
    assert result.deltas["plasticity"] == 0.1
    assert result.deltas["conviction"] == -0.1


def test_adjudication_ignores_unknown_traits():
    text = '{"trait_deltas": {"not_a_real_trait": 0.05}, "justification": "x"}'
    result = parse_adjudication(text, ["plasticity"], max_delta=0.1)
    assert result.deltas == {}


def test_apply_adjudication_only_moves_the_named_user():
    from discourse_lab.llm.adjudication import AdjudicationResult

    _, _, pop, _ = _setup(n_users=10)
    x_before = pop.X_stored.copy()
    result = AdjudicationResult(deltas={"plasticity": 0.05}, justification="test")

    apply_adjudication(pop, user_idx=3, result=result)

    col = pop.trait_names.index("plasticity")
    assert pop.X_stored[3, col] != x_before[3, col]
    other_rows = [i for i in range(10) if i != 3]
    np.testing.assert_array_equal(pop.X_stored[other_rows], x_before[other_rows])


def test_request_adjudication_end_to_end_with_fake_client():
    from discourse_lab.llm.voice_card import VoiceCard
    from discourse_lab.llm.adjudication import SalientEvent

    client = FakeLLMClient(['{"trait_deltas": {"plasticity": 0.03}, "justification": "mild update"}'])
    card = VoiceCard(persona="A calm poster.", tics=(), register="plain", raw_text="")
    event = SalientEvent(post_id=1, author=0, kind="top_engagement", detail={"engagement_count": 500})

    result = request_adjudication(client, card, event, trait_names=["plasticity"], max_delta=0.1)
    assert result.deltas["plasticity"] == pytest.approx(0.03)
    assert len(client.calls) == 1


# --------------------------------------------------------------------------
# Ollama Cloud client: fails fast without a key, never touches the network
# --------------------------------------------------------------------------


def test_ollama_cloud_client_fails_fast_without_api_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    client = OllamaCloudClient(api_key=None)
    with pytest.raises(LLMError, match="OLLAMA_API_KEY"):
        client.chat([{"role": "user", "content": "hi"}])
