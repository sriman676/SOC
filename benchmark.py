
from env.environment import SOCEnv
from agents.proxy_agent import choose_action_with_proxy
import numpy as np


def run_episode(seed, profile):
    env = SOCEnv(stochastic=True, seed=seed, profile=profile, episode_length=7)
    s = env.reset()
    rewards = []
    done = False

    while not done:
        action, _ = choose_action_with_proxy(s["observation"])
        s = env.step(action)
        rewards.append(s["reward"])
        done = s["done"]

    metrics = env.state()["metrics"]
    malicious_seen = metrics.get("malicious_seen", 0)
    detection_rate = 0.0
    if malicious_seen:
        detection_rate = metrics.get("detections", 0) / malicious_seen

    return {
        "mean_reward": float(np.mean(rewards)),
        "detection_rate": detection_rate,
        "false_positives": metrics.get("false_positives", 0),
        "profile": profile,
    }


profiles = ["balanced", "aggressive", "stealthy"]
episodes = [run_episode(seed=i + 101, profile=profiles[i % len(profiles)]) for i in range(9)]
reward_scores = [e["mean_reward"] for e in episodes]
detection_scores = [e["detection_rate"] for e in episodes]
false_positive_scores = [e["false_positives"] for e in episodes]

print("Benchmark Score:", float(np.mean(reward_scores)))
print("Reward StdDev:", float(np.std(reward_scores)))
print("Detection Rate:", float(np.mean(detection_scores)))
print("Avg False Positives:", float(np.mean(false_positive_scores)))

for profile in profiles:
    subset = [e for e in episodes if e["profile"] == profile]
    profile_reward = [e["mean_reward"] for e in subset]
    profile_detect = [e["detection_rate"] for e in subset]
    print(
        f"Profile={profile} AvgReward={float(np.mean(profile_reward)):.3f} "
        f"AvgDetect={float(np.mean(profile_detect)):.3f}"
    )
