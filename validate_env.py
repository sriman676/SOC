
import yaml

from env.environment import SOCEnv
from tasks import TASKS
from graders import grade_all


def validate_openenv_metadata(path="openenv.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    tasks = cfg.get("tasks") or []
    if not isinstance(tasks, list):
        raise AssertionError("openenv.yaml tasks must be a list")

    graded = 0
    for item in tasks:
        if not isinstance(item, dict):
            continue

        grader = item.get("grader")
        score = item.get("score")
        if isinstance(grader, str) and grader.strip():
            graded += 1
        if score is not None:
            assert 0.0 < float(score) < 1.0

    assert graded >= 4


def validate_graders():
    assert len(TASKS) >= 4
    assert sum(1 for cfg in TASKS.values() if callable(cfg.get("grader"))) >= 4

    edge_cases = [
        None,
        [],
        [1, "x", None],
        [{"event_id": "4625", "detected": "true"}],
        [{"event_id": 4672, "detected": 0}],
        [{"event_id": 4688, "detected": True}],
        [{"event_id": 5156}],
        [
            {"event_id": 4625, "detected": True},
            {"event_id": 4625, "detected": False},
            {"event_id": 4672, "detected": True},
            {"event_id": 4688, "detected": True},
            {"event_id": 5156, "detected": True},
        ],
    ]

    for events in edge_cases:
        scores = grade_all(events)
        for value in scores.values():
            assert 0.0 < float(value) < 1.0

        for task_name, task_cfg in TASKS.items():
            task_score = task_cfg["grader"](events)
            assert 0.0 < float(task_score) < 1.0, task_name


def validate_reward_bounds():
    env = SOCEnv(stochastic=False, profile="aggressive")

    reset_state = env.reset()
    assert 0.0 < float(reset_state["reward"]) < 1.0

    # Force edge-case actions that previously produced 0.0, 1.0, and >1.0 rewards.
    for action in ["ignore", "contain", "contain", "investigate", "contain"]:
        step_state = env.step(action)
        assert 0.0 < float(step_state["reward"]) < 1.0

def validate():
    validate_openenv_metadata()
    validate_graders()
    validate_reward_bounds()

    env = SOCEnv()
    s = env.reset()
    done = False
    steps = 0

    while not done and steps < 20:
        s = env.step("investigate")
        assert "reward" in s
        assert "done" in s
        done = s["done"]
        steps += 1

    print("Validation passed")

if __name__ == "__main__":
    validate()
