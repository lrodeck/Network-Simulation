"""Reading the simulation in the scenario's own words.

The scenario format has always carried the vocabulary — each stance axis has a
`name` and two pole labels, and topics can be named — and until now the only
consumer anywhere was the LLM renderer. Every measure, table, figure and log
line said `stance_0 = -1.2` where it could say "provision: leans market".

For normative work that is not cosmetic. "Cross-cutting exposure fell 30%" is
uninterpretable without knowing which cleavage stopped being crossed.

This package imports numpy and scipy only — never matplotlib — so it stays
usable from the core and from a headless sweep.
"""

from discourse_lab.semantics.lexicon import Lexicon, lexicon_for
from discourse_lab.semantics.narrate import (
    DiscourseSummary,
    Narrator,
    describe_run,
    describe_state,
)

__all__ = [
    "DiscourseSummary",
    "Lexicon",
    "Narrator",
    "describe_run",
    "describe_state",
    "lexicon_for",
]
