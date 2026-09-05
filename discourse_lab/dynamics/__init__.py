"""Timing, generation, and (later steps) exposure/reaction/drift (spec §2.3-2.8)."""

from discourse_lab.dynamics.cascade import CascadeState, check_r_eff, derive_posts, follower_counts, r_eff
from discourse_lab.dynamics.expression import ExpressionMap
from discourse_lab.dynamics.hawkes import HawkesThreads
from discourse_lab.dynamics.perception import PerceivedState, compute_perception
from discourse_lab.dynamics.posts import PostBatch, generate_posts
from discourse_lab.dynamics.render import stub_render
from discourse_lab.dynamics.timing import FatigueState, circadian_factor, circadian_shape, sample_post_counts

__all__ = [
    "ExpressionMap",
    "HawkesThreads",
    "PostBatch",
    "generate_posts",
    "stub_render",
    "FatigueState",
    "circadian_factor",
    "circadian_shape",
    "sample_post_counts",
    "PerceivedState",
    "compute_perception",
    "CascadeState",
    "derive_posts",
    "follower_counts",
    "r_eff",
    "check_r_eff",
]
