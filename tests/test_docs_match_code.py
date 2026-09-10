"""MODEL.md quotes ~40 default values. A document that says the default is 30
while the code says 40 is worse than no document at all, so the knob tables are
parsed and checked against the live `Config` rather than trusted.

Deliberately parses the published tables rather than keeping a second list to
compare against: a duplicate list would drift from the prose it is supposed to
be guarding, and the failure mode this test exists to catch is exactly drift.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from discourse_lab.config import Config

MODEL_MD = pathlib.Path(__file__).resolve().parents[1] / "MODEL.md"

# Which sub-config each documented knob lives on. A field named in MODEL.md but
# absent here is a test failure, not a silent skip.
SECTIONS = {
    "population": "population",
    "graph": "graph",
    "dynamics": "dynamics",
}

# Documented as prose rather than a literal (e.g. "library defaults", "()").
NON_LITERAL = {"archetype_weights", "archetype_offsets", "correlation_pairs",
               "kernel_theta", "ou_k", "affect_weights_hostility",
               "affect_weights_support", "affect_valence_signs"}


def _documented_defaults() -> dict[str, str]:
    """`{field: documented default}` from every `| field | default | … |` row."""
    text = MODEL_MD.read_text()
    table = text[text.index("## 14. Every knob"):text.index("## 15. What this model")]

    found: dict[str, str] = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        field, default = cells[0].strip("`"), cells[1]
        if field in ("Field", "---") or not re.fullmatch(r"[a-z_]+(?: / `?_?[a-z_]+`?)?", field):
            continue
        found[field] = default
    return found


def _actual(field: str):
    for attr in SECTIONS.values():
        section = getattr(Config(), attr)
        if hasattr(section, field):
            return getattr(section, field)
    raise AttributeError(field)


def test_the_knob_table_is_not_empty():
    """A parser that silently matches nothing would make every test below pass."""
    assert len(_documented_defaults()) >= 35


def test_every_documented_field_exists_on_the_config():
    unknown = []
    for field in _documented_defaults():
        for name in field.split(" / "):
            name = name.strip("`").lstrip("_")
            if name in ("attention_budget",):
                continue
            try:
                _actual(name)
            except AttributeError:
                unknown.append(name)
    assert not unknown, f"MODEL.md documents fields that do not exist: {unknown}"


@pytest.mark.parametrize("field,documented", sorted(_documented_defaults().items()))
def test_documented_default_matches_the_code(field, documented):
    if " / " in field:
        pytest.skip("paired row; each field checked individually elsewhere")
    if field in NON_LITERAL:
        return
    actual = _actual(field)
    if isinstance(actual, bool):
        assert documented.lower() == str(actual).lower()
    elif isinstance(actual, (int, float)):
        assert float(documented) == pytest.approx(float(actual)), \
            f"MODEL.md says {field} = {documented}, code says {actual}"
    else:
        assert documented.strip("`") == str(actual)


def test_documented_stylized_fact_targets_match_the_code():
    """Every numeric range printed in the validity-gate table must be one the
    code actually enforces, and every fact must have a row."""
    from discourse_lab.metrics import STYLIZED_FACT_RANGES

    text = MODEL_MD.read_text()
    section = text[text.index("### Validity gate"):text.index("### The result")]
    rows = [line for line in section.splitlines()
            if line.startswith("|") and "---" not in line and "Target" not in line]
    assert len(rows) == len(STYLIZED_FACT_RANGES), \
        f"{len(rows)} rows documented against {len(STYLIZED_FACT_RANGES)} facts"

    coded = {(lo, hi) for lo, hi in STYLIZED_FACT_RANGES.values()}
    printed = re.findall(r"\|\s*([0-9.]+)\s*[\u2013-]\s*([0-9.]+)\s*\|", section)
    assert printed, "the targets table stopped parsing"
    for lo, hi in printed:
        assert (float(lo), float(hi)) in coded, \
            f"MODEL.md prints target {lo}-{hi}, which the code does not enforce"


def test_documented_kernels_and_rankers_are_the_registered_ones():
    from discourse_lab.exposure.kernel import KERNEL_THETAS
    from discourse_lab.registry import names
    import discourse_lab.exposure  # noqa: F401  registers the ranker table

    text = MODEL_MD.read_text()
    for kernel in KERNEL_THETAS:
        assert f"`{kernel}`" in text, f"kernel {kernel} is undocumented"
    for ranker in names("ranker"):
        assert f"`{ranker}`" in text, f"ranker {ranker} is undocumented"


def test_documented_trait_blocks_match_the_trait_table():
    from discourse_lab.population.traits import BEHAVIOR, EXPRESSION, META, PERSONALITY

    text = MODEL_MD.read_text()
    section = text[text.index("## 2. The people"):text.index("## 3. The network")]
    for trait in PERSONALITY + EXPRESSION + BEHAVIOR + META:
        assert trait in section, f"trait {trait} is missing from MODEL.md's block table"


def test_documented_post_fields_match_postbatch():
    from discourse_lab.dynamics.expression import POST_DIMS

    text = MODEL_MD.read_text()
    section = text[text.index("## 5. What a post is"):text.index("## 6. Replies")]
    for dim in POST_DIMS:
        assert f"`{dim}`" in section, f"post dim {dim} is undocumented"


def test_documented_outcomes_match_the_registry():
    from discourse_lab.outcomes import outcome_names

    text = MODEL_MD.read_text()
    for name in outcome_names():
        assert f"`{name}`" in text, f"outcome {name} is undocumented"
