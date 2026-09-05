"""Step 7 verification (dev §6): F_local/F_global blending, measure registry
including the salience/stance agreement pair and bubble index.
"""

from __future__ import annotations

import numpy as np

from discourse_lab.dynamics.perception import compute_perception
from discourse_lab.dynamics.posts import PostBatch
from discourse_lab.exposure.attention import Exposures
from discourse_lab.measures import attention_gini, bubble_index, compute_measure, measure_names, salience_stance_agreement


def _posts(topics: np.ndarray, stances: np.ndarray) -> PostBatch:
    m = len(topics)
    return PostBatch(
        author=np.zeros(m, dtype=int), topic=topics, stance=stances,
        arousal=np.zeros(m), valence=np.zeros(m), provocativeness=np.zeros(m),
        novelty=np.zeros(m), specificity=np.zeros(m), quality=np.zeros(m), length=np.zeros(m),
        id=np.arange(m), t=np.zeros(m, dtype=int), parent=np.full(m, -1), root=np.arange(m),
        depth=np.zeros(m, dtype=int), kind=np.full(m, "post"), engagement_count=np.zeros(m, dtype=int),
    )


def test_zero_exposure_users_collapse_to_global_state():
    n_users, K, D = 5, 3, 2
    s_global = np.array([0.2, 0.3, 0.5])
    sigma_global = np.zeros((K, D))
    sigma_global[1] = [1.0, -1.0]

    posts = _posts(np.array([0]), np.array([[0.0, 0.0]]))
    exposures = Exposures(post_idx=np.empty(0, dtype=int), user_id=np.empty(0, dtype=int), rank=np.empty(0, dtype=int))
    is_follower = np.empty(0, dtype=bool)

    perceived = compute_perception(n_users, exposures, is_follower, posts, s_global, sigma_global)
    np.testing.assert_allclose(perceived.w, 0.0)
    np.testing.assert_allclose(perceived.s_perceived, np.tile(s_global, (n_users, 1)))
    np.testing.assert_allclose(perceived.sigma_perceived, np.tile(sigma_global, (n_users, 1, 1)))


def test_pure_follower_exposures_give_w_one_and_pure_local_perception():
    n_users, K, D = 2, 2, 1
    s_global = np.array([0.5, 0.5])
    sigma_global = np.zeros((K, D))

    # user 0 sees 3 posts on topic 0 with stance 2.0; user 1 sees none
    posts = _posts(np.array([0, 0, 0]), np.array([[2.0], [2.0], [2.0]]))
    exposures = Exposures(post_idx=np.array([0, 1, 2]), user_id=np.array([0, 0, 0]), rank=np.zeros(3, dtype=int))
    is_follower = np.array([True, True, True])

    perceived = compute_perception(n_users, exposures, is_follower, posts, s_global, sigma_global)
    assert perceived.w[0] == 1.0
    assert perceived.w[1] == 0.0
    np.testing.assert_allclose(perceived.s_local[0], [1.0, 0.0])
    np.testing.assert_allclose(perceived.sigma_local[0, 0], [2.0])
    # topic 1 unseen by user 0 -> falls back to global (zero)
    np.testing.assert_allclose(perceived.sigma_perceived[0, 1], sigma_global[1])
    # user 1 had no exposures at all -> pure global
    np.testing.assert_allclose(perceived.s_perceived[1], s_global)


def test_injection_only_exposures_give_w_zero():
    n_users, K, D = 1, 2, 1
    s_global = np.array([0.5, 0.5])
    sigma_global = np.zeros((K, D))
    posts = _posts(np.array([0]), np.array([[5.0]]))
    exposures = Exposures(post_idx=np.array([0]), user_id=np.array([0]), rank=np.array([0]))
    is_follower = np.array([False])  # injected, not a followee

    perceived = compute_perception(n_users, exposures, is_follower, posts, s_global, sigma_global)
    assert perceived.w[0] == 0.0
    np.testing.assert_allclose(perceived.s_perceived[0], s_global)


def test_salience_stance_agreement_pair():
    from discourse_lab.dynamics.perception import PerceivedState

    n, K, D = 3, 2, 1
    s_perceived = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    sigma_perceived = np.zeros((n, K, D))
    sigma_perceived[0, 0] = [1.0]  # user 0's dominant topic is 0, perceived stance 1.0
    sigma_perceived[1, 1] = [1.0]

    perceived = PerceivedState(
        s_local=s_perceived, sigma_local=sigma_perceived, sigma_var_local=np.zeros((n, K)), w=np.ones(n),
        s_perceived=s_perceived, sigma_perceived=sigma_perceived,
    )
    stance_u = np.array([[1.0], [1.0], [0.0]])  # users 0 agrees exactly, user 1 disagrees

    salience, agreement = salience_stance_agreement(perceived, stance_u)
    assert salience > 0.5  # mean of [0.9, 0.8, 0.5]
    # user 0's agreement (0 distance) should pull the mean above a fully-disagreeing case
    stance_all_far = np.array([[5.0], [5.0], [5.0]])
    _, agreement_far = salience_stance_agreement(perceived, stance_all_far)
    assert agreement > agreement_far


def test_bubble_index_high_when_local_variance_far_below_global():
    from discourse_lab.dynamics.perception import PerceivedState

    n, K, D = 4, 1, 1
    s_local = np.ones((n, K))
    sigma_local = np.full((n, K, D), 3.0)  # all posts a user saw carried the identical stance
    sigma_var_local = np.zeros((n, K))     # so within-user variance is exactly 0
    perceived = PerceivedState(
        s_local=s_local, sigma_local=sigma_local, sigma_var_local=sigma_var_local, w=np.ones(n),
        s_perceived=s_local, sigma_perceived=sigma_local,
    )

    high_bubble = bubble_index(perceived, global_stance_var=2.0)
    assert high_bubble > 0.99


def test_bubble_index_near_zero_when_local_matches_global_variance():
    from discourse_lab.dynamics.perception import PerceivedState

    n, K, D = 500, 1, 1
    global_var = 1.5
    s_local = np.ones((n, K))
    sigma_var_local = np.full((n, K), global_var)  # each user's feed is exactly as diverse as the population
    sigma_local = np.zeros((n, K, D))
    perceived = PerceivedState(
        s_local=s_local, sigma_local=sigma_local, sigma_var_local=sigma_var_local, w=np.ones(n),
        s_perceived=s_local, sigma_perceived=sigma_local,
    )

    bubble = bubble_index(perceived, global_stance_var=global_var)
    assert bubble < 0.01


def test_attention_gini_zero_for_uniform_and_high_for_concentrated():
    uniform = np.full(20, 5.0)
    concentrated = np.zeros(20)
    concentrated[0] = 100.0

    assert attention_gini(uniform) < 1e-9
    assert attention_gini(concentrated) > 0.8


def test_compute_perception_tracks_within_user_variance():
    n_users, K, D = 1, 1, 1
    s_global = np.array([1.0])
    sigma_global = np.zeros((K, D))

    stances = np.array([[0.0], [2.0], [4.0]])  # mean 2, variance ((0-2)^2+(0)+ (2)^2)/3 = 8/3
    posts = _posts(np.array([0, 0, 0]), stances)
    exposures = Exposures(post_idx=np.array([0, 1, 2]), user_id=np.array([0, 0, 0]), rank=np.zeros(3, dtype=int))
    is_follower = np.array([True, True, True])

    perceived = compute_perception(n_users, exposures, is_follower, posts, s_global, sigma_global)
    np.testing.assert_allclose(perceived.sigma_local[0, 0], [2.0])
    np.testing.assert_allclose(perceived.sigma_var_local[0, 0], 8.0 / 3.0)


def test_measure_registry_lists_and_dispatches():
    names = measure_names()
    assert {"salience_stance_agreement", "bubble_index", "attention_gini"} <= set(names)

    result = compute_measure("attention_gini", np.array([1.0, 1.0, 1.0]))
    assert result == 0.0
