import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.environment import ACTIONS, SOCEnv
from graders import grade_all
from tasks import TASKS


def assert_open_interval(value):
    numeric = float(value)
    assert 0.0 < numeric < 1.0


def test_grade_all_scores_always_in_open_interval_for_edge_cases():
    edge_cases = [
        None,
        [],
        [1, "x", None],
        [{"event_id": "4625", "detected": "true"}],
        [{"event_id": 4672, "detected": 0}],
        [{"event_id": 5156}],
        [
            {"event_id": 4625, "detected": True},
            {"event_id": 4625, "detected": False},
            {"event_id": 4672, "detected": True},
            {"event_id": 5156, "detected": True},
        ],
    ]

    for events in edge_cases:
        scores = grade_all(events)
        for value in scores.values():
            assert_open_interval(value)



def test_each_task_grader_scores_in_open_interval():
    events = [
        {"event_id": 4625, "detected": True},
        {"event_id": 4625, "detected": False},
        {"event_id": 4672, "detected": True},
        {"event_id": 5156, "detected": False},
        {"event_id": 4624, "detected": False},
    ]

    for task_cfg in TASKS.values():
        score = task_cfg["grader"](events)
        assert_open_interval(score)



def test_env_reset_reward_is_strictly_between_zero_and_one():
    env = SOCEnv(stochastic=False)
    state = env.reset()
    assert_open_interval(state["reward"])



def test_env_step_reward_edge_paths_are_strictly_between_zero_and_one():
    env = SOCEnv(stochastic=False, profile="aggressive")
    env.reset()

    # Deterministic chain plus this sequence exercises prior edge values:
    # 0.0 (miss), 1.0 (detect), and >1.0 (contain bonuses).
    for action in ["ignore", "contain", "contain", "investigate", "contain"]:
        step_state = env.step(action)
        assert_open_interval(step_state["reward"])



def test_env_step_reward_randomized_actions_stay_in_open_interval():
    rng = random.Random(123)

    for profile in ["balanced", "aggressive", "stealthy"]:
        env = SOCEnv(stochastic=True, seed=11, profile=profile, episode_length=12)
        state = env.reset()
        assert_open_interval(state["reward"])

        done = False
        while not done:
            action = rng.choice(ACTIONS)
            state = env.step(action)
            assert_open_interval(state["reward"])
            done = state["done"]


def _build_random_events(case_id):
    rng = random.Random(2026 + case_id)
    events = []
    for _ in range(rng.randint(0, 40)):
        events.append(
            {
                "event_id": rng.choice([4624, 4625, 4672, 4688, 5156, "4625", "x", None]),
                "detected": rng.choice([True, False, 1, 0, "true", "false", None]),
            }
        )
    return events


@pytest.mark.parametrize("case_id", range(100))
def test_open_interval_grader_random_case(case_id):
    events = _build_random_events(case_id)
    for value in grade_all(events).values():
        assert_open_interval(value)


@pytest.mark.parametrize("case_id", range(100))
def test_open_interval_environment_random_episode(case_id):
    rng = random.Random(8000 + case_id)
    profile = ["balanced", "aggressive", "stealthy"][case_id % 3]
    env = SOCEnv(stochastic=True, seed=500 + case_id, profile=profile, episode_length=12)
    state = env.reset()
    assert_open_interval(state["reward"])

    done = False
    while not done:
        state = env.step(rng.choice(ACTIONS))
        assert_open_interval(state["reward"])
        done = state["done"]


def test_open_interval_stress_10000_cases():
    # 10,000 randomized grader cases.
    for case_id in range(10000):
        events = _build_random_events(100000 + case_id)
        for value in grade_all(events).values():
            assert_open_interval(value)

    # 10,000 randomized environment step cases.
    # Use short episodes to keep runtime practical while still exercising transitions.
    rng = random.Random(91000)
    profiles = ["balanced", "aggressive", "stealthy"]
    steps_checked = 0
    while steps_checked < 10000:
        profile = profiles[steps_checked % 3]
        env = SOCEnv(stochastic=True, seed=120000 + steps_checked, profile=profile, episode_length=3)
        state = env.reset()
        assert_open_interval(state["reward"])

        done = False
        while not done and steps_checked < 10000:
            state = env.step(rng.choice(ACTIONS))
            assert_open_interval(state["reward"])
            done = state["done"]
            steps_checked += 1
