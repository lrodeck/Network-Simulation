"""Change spec V8.3: the linked-metric registry. Two instances of a
reported metric being a reparameterisation of another metric it was then
compared against as independent evidence were each found and written up as
findings before being caught (`recovery_fraction`/`split_ratio`,
`experiments/experiment03_bubble_intervention.py`). This registers the
known links so a comparison helper can raise instead of repeating the
mistake a third time.
"""

from __future__ import annotations

import pytest

import discourse_lab.experiments.experiment03_bubble_intervention  # noqa: F401  (registers split_ratio/recovery_fraction)
import discourse_lab.outcomes  # noqa: F401  (registers the outcome-registry entries)
from discourse_lab.registry import (
    ALGEBRAIC_LINKS,
    LinkedMetricError,
    check_comparison,
    derives_from,
    get,
    names,
    register,
)


def test_algebraic_links_is_seeded_not_empty():
    """A registry with an empty table is the same failure as a docstring
    promising a de-escalation path that never fires (V8.3's own framing) --
    it must actually carry the links this change spec found."""
    assert len(ALGEBRAIC_LINKS) >= 3
    assert frozenset({"ingroup", "outgroup"}) in ALGEBRAIC_LINKS
    assert frozenset({"toward_mean", "toward_own_pole"}) in ALGEBRAIC_LINKS
    assert frozenset({"algorithmic_share", "rank_penalty"}) in ALGEBRAIC_LINKS


def test_check_comparison_raises_on_a_known_algebraic_link_either_order():
    with pytest.raises(LinkedMetricError):
        check_comparison("ingroup", "outgroup")
    with pytest.raises(LinkedMetricError):
        check_comparison("outgroup", "ingroup")  # order must not matter


def test_check_comparison_is_silent_on_unrelated_metrics():
    check_comparison("camp_share", "posting_gini")  # must not raise
    check_comparison("bubble", "topic_narrowing")


def test_check_comparison_allows_comparing_a_metric_to_itself():
    check_comparison("recovery_fraction", "recovery_fraction")


def test_recovery_fraction_and_split_ratio_are_registered_as_a_derives_from_pair():
    """This is the case study that motivated V8.3: `recovery_fraction` is a
    clean parent/child of `split_ratio` (`recovery_fraction == b *
    (1 - split_ratio)`), registered via `derives_from` rather than
    `ALGEBRAIC_LINKS` (which is for the messier, non-parent/child pairs)."""
    assert "split_ratio" in names("cross_run_metric")
    assert "recovery_fraction" in names("cross_run_metric")
    assert derives_from("cross_run_metric", "recovery_fraction") == ("split_ratio",)
    assert derives_from("cross_run_metric", "split_ratio") == ()

    with pytest.raises(LinkedMetricError):
        check_comparison("recovery_fraction", "split_ratio")
    with pytest.raises(LinkedMetricError):
        check_comparison("split_ratio", "recovery_fraction")


def test_register_rejects_duplicate_names_within_a_kind():
    # A throwaway KIND, not "outcome": the registry is process-global and
    # has no unregister, so registering under a real kind here would leak
    # into every other test that assumes "outcome" is exactly its own six
    # (tests/test_outcomes.py, tests/test_docs_match_code.py).
    register("__test_only_kind__", "dupe")(lambda: None)
    with pytest.raises(ValueError, match="duplicate registration"):
        register("__test_only_kind__", "dupe")(lambda: None)


def test_get_reports_available_names_on_a_miss():
    with pytest.raises(KeyError, match="voice_inequality"):
        get("outcome", "not_a_real_outcome")
