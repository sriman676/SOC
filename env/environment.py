
import random


ACTIONS = ["investigate", "ignore", "contain"]
EPSILON = 1e-6
PROFILE_SETTINGS = {
    "balanced": {"malicious_bias": 0.00, "high_severity_bias": 0.00, "contain_bonus": 0.00},
    "aggressive": {"malicious_bias": 0.20, "high_severity_bias": 0.15, "contain_bonus": 0.10},
    "stealthy": {"malicious_bias": 0.05, "high_severity_bias": -0.20, "contain_bonus": 0.00},
}


class SOCEnv:

    def __init__(self, stochastic=False, seed=7, profile="balanced", episode_length=5):
        if profile not in PROFILE_SETTINGS:
            raise ValueError(f"Unknown profile: {profile}")

        self.stochastic = stochastic
        self.seed = seed
        self.rng = random.Random(seed)
        self.profile = profile
        self.episode_length = max(3, episode_length)
        self.episode_id = 0
        self.chain = self._build_attack_chain()
        self.reset()

    def _build_attack_chain(self):
        if self.stochastic:
            return self._build_stochastic_chain()
        return self._build_deterministic_chain()

    def _build_deterministic_chain(self):
        contain_bonus = PROFILE_SETTINGS[self.profile]["contain_bonus"]
        return [
            {
                "event_id": 4625,
                "severity": "high",
                "stage": "initial_access",
                "failed_logins": 18,
                "src_geo": "rare",
                "malicious": True,
                "adversary_profile": self.profile,
                "profile_contain_bonus": contain_bonus,
            },
            {
                "event_id": 4624,
                "severity": "low",
                "stage": "normal_activity",
                "failed_logins": 0,
                "src_geo": "common",
                "malicious": False,
                "adversary_profile": self.profile,
                "profile_contain_bonus": contain_bonus,
            },
            {
                "event_id": 4672,
                "severity": "high",
                "stage": "privilege_escalation",
                "account_role": "service",
                "new_admin_session": True,
                "malicious": True,
                "adversary_profile": self.profile,
                "profile_contain_bonus": contain_bonus,
            },
            {
                "event_id": 4688,
                "severity": "medium",
                "stage": "credential_access",
                "process_name": "procdump.exe",
                "parent_process": "cmd.exe",
                "malicious": True,
                "adversary_profile": self.profile,
                "profile_contain_bonus": contain_bonus,
            },
            {
                "event_id": 5156,
                "severity": "high",
                "stage": "exfiltration",
                "bytes_out": 42000000,
                "dest_reputation": "unknown",
                "malicious": True,
                "adversary_profile": self.profile,
                "profile_contain_bonus": contain_bonus,
            },
        ]

    def _build_stochastic_chain(self):
        stage_event_id = {
            "normal_activity": 4624,
            "initial_access": 4625,
            "privilege_escalation": 4672,
            "credential_access": 4688,
            "exfiltration": 5156,
        }
        stage_weights = {
            "balanced": {
                "normal_activity": 0.30,
                "initial_access": 0.20,
                "privilege_escalation": 0.20,
                "credential_access": 0.15,
                "exfiltration": 0.15,
            },
            "aggressive": {
                "normal_activity": 0.12,
                "initial_access": 0.22,
                "privilege_escalation": 0.24,
                "credential_access": 0.20,
                "exfiltration": 0.22,
            },
            "stealthy": {
                "normal_activity": 0.35,
                "initial_access": 0.25,
                "privilege_escalation": 0.18,
                "credential_access": 0.14,
                "exfiltration": 0.08,
            },
        }
        stage_malicious_prob = {
            "normal_activity": 0.05,
            "initial_access": 0.70,
            "privilege_escalation": 0.75,
            "credential_access": 0.80,
            "exfiltration": 0.90,
        }

        profile_cfg = PROFILE_SETTINGS[self.profile]
        stages = list(stage_weights[self.profile].keys())
        weights = list(stage_weights[self.profile].values())
        chain = []

        for _ in range(self.episode_length):
            stage = self.rng.choices(stages, weights=weights, k=1)[0]
            malicious_prob = stage_malicious_prob[stage] + profile_cfg["malicious_bias"]
            malicious_prob = max(0.0, min(1.0, malicious_prob))
            malicious = self.rng.random() < malicious_prob

            if malicious:
                high_cutoff = 0.5 + profile_cfg["high_severity_bias"]
                high_cutoff = max(0.1, min(0.9, high_cutoff))
                severity = "high" if self.rng.random() < high_cutoff else "medium"
            else:
                severity = "low"

            event = {
                "event_id": stage_event_id[stage],
                "severity": severity,
                "stage": stage,
                "failed_logins": self.rng.randint(0, 25) if stage == "initial_access" else 0,
                "src_geo": "rare" if self.rng.random() < 0.25 else "common",
                "account_role": "service" if stage == "privilege_escalation" else "user",
                "new_admin_session": stage == "privilege_escalation" and malicious,
                "process_name": "procdump.exe" if stage == "credential_access" and malicious else "normal.exe",
                "parent_process": "cmd.exe" if stage == "credential_access" else "explorer.exe",
                "bytes_out": self.rng.randint(1_000_000, 60_000_000) if stage == "exfiltration" else 0,
                "dest_reputation": "unknown" if stage == "exfiltration" and self.rng.random() < 0.7 else "trusted",
                "malicious": malicious,
                "adversary_profile": self.profile,
                "profile_contain_bonus": profile_cfg["contain_bonus"],
            }
            chain.append(event)

        if not any(e["malicious"] for e in chain):
            chain[-1]["malicious"] = True
            chain[-1]["severity"] = "high"
            chain[-1]["stage"] = "exfiltration"
            chain[-1]["event_id"] = 5156
            chain[-1]["bytes_out"] = 25_000_000
            chain[-1]["dest_reputation"] = "unknown"

        return chain

    def _risk_score(self, event):
        score = 0
        severity_weights = {"low": 10, "medium": 25, "high": 45}
        stage_weights = {
            "normal_activity": 0,
            "initial_access": 10,
            "privilege_escalation": 20,
            "credential_access": 20,
            "exfiltration": 30,
        }

        score += severity_weights.get(event.get("severity"), 0)
        score += stage_weights.get(event.get("stage"), 0)

        if event.get("failed_logins", 0) >= 10:
            score += 20
        if event.get("new_admin_session"):
            score += 15
        if event.get("process_name") == "procdump.exe":
            score += 20
        if event.get("bytes_out", 0) > 10000000:
            score += 25
        if event.get("dest_reputation") == "unknown":
            score += 10
        if event.get("src_geo") == "rare":
            score += 8

        return min(score, 100)

    def _strict_reward(self, reward):
        """Clamp rewards to the strict open interval (0, 1)."""
        return min(1.0 - EPSILON, max(EPSILON, float(reward)))

    def reset(self):
        if self.stochastic:
            self.episode_id += 1
            self.chain = self._build_attack_chain()

        self.index = 0
        self.total_reward = 0.0
        self.previous_action = None
        self.metrics = {
            "alerts_seen": 0,
            "malicious_seen": 0,
            "detections": 0,
            "false_positives": 0,
            "missed_malicious": 0,
        }
        return {
            "observation": self.chain[self.index],
            "reward": self._strict_reward(0.5),
            "done": False,
        }

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")

        cur = self.chain[self.index]
        malicious = cur["malicious"]
        risk = self._risk_score(cur)

        if malicious:
            self.metrics["malicious_seen"] += 1

        if malicious and action == "investigate":
            reward = 1.0
            cur["detected"] = True
            self.metrics["detections"] += 1
        elif malicious and action == "contain":
            reward = 1.1
            cur["detected"] = True
            self.metrics["detections"] += 1
        elif malicious and action == "ignore":
            reward = 0.0
            cur["detected"] = False
            self.metrics["missed_malicious"] += 1
        elif not malicious and action == "investigate":
            reward = 0.35
            self.metrics["false_positives"] += 1
        elif not malicious and action == "contain":
            reward = 0.1
            self.metrics["false_positives"] += 1
        else:
            reward = 0.75

        if action == "ignore" and malicious and risk >= 70:
            reward -= 0.25

        if self.previous_action == "investigate" and action == "investigate" and not malicious:
            reward -= 0.1

        if action == "contain" and cur.get("stage") == "exfiltration" and malicious:
            reward += 0.1 + cur.get("profile_contain_bonus", 0.0)

        reward = self._strict_reward(reward)

        self.metrics["alerts_seen"] += 1
        self.total_reward += reward
        self.previous_action = action

        self.index += 1
        done = self.index >= len(self.chain)
        obs = None if done else self.chain[self.index]

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": {
                "risk_score": risk,
                "stage": cur.get("stage"),
                "adversary_profile": cur.get("adversary_profile"),
                "metrics": dict(self.metrics),
            },
        }

    def state(self):
        return {
            "index": self.index,
            "stochastic": self.stochastic,
            "profile": self.profile,
            "episode_id": self.episode_id,
            "total_reward": round(self.total_reward, 3),
            "metrics": dict(self.metrics),
        }
