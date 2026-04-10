from collections.abc import Iterable


EPSILON = 1e-6

TASK_EVENT_IDS = {
    "login_attack_detection": 4625,
    "normal_activity_handling": 4624,
    "privilege_escalation_detection": 4672,
    "credential_access_detection": 4688,
    "data_exfiltration_detection": 5156,
}

TASK_WEIGHTS = {
    "login_attack_detection": 0.18,
    "normal_activity_handling": 0.12,
    "privilege_escalation_detection": 0.25,
    "credential_access_detection": 0.20,
    "data_exfiltration_detection": 0.25,
}


def _strict_open_interval(score):
    """Clamp scores to the open interval (0, 1)."""
    return min(1.0 - EPSILON, max(EPSILON, float(score)))


def _iter_events(events):
    """Return a safe iterable of event dictionaries."""
    if events is None or isinstance(events, (str, bytes)):
        return []
    if isinstance(events, dict):
        return [events]
    if not isinstance(events, Iterable):
        return []
    return events


def _normalized_event_id(event):
    raw = event.get("event_id")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.isdigit():
            return int(text)
    return None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def grade(task_name, events=None):
    if task_name not in TASK_EVENT_IDS:
        raise ValueError(f"Unknown task: {task_name}")

    target = TASK_EVENT_IDS[task_name]
    detected = 0
    total = 0

    for e in _iter_events(events):
        if not isinstance(e, dict):
            continue

        if _normalized_event_id(e) == target:
            total += 1
            if _to_bool(e.get("detected", False)):
                detected += 1

    if total == 0:
        return _strict_open_interval(0.5)

    return _strict_open_interval(detected / total)


def grade_all(events=None):
    results = {}
    weighted_total = 0.0

    for name, weight in TASK_WEIGHTS.items():
        task_score = grade(name, events)
        results[name] = task_score
        weighted_total += task_score * weight

    results["weighted_score"] = _strict_open_interval(weighted_total)
    return results


def grade_login_attack_detection(events=None):
    """Easy: Detect brute-force login attacks (Windows Event ID 4625)."""
    return grade("login_attack_detection", events)


def grade_privilege_escalation_detection(events=None):
    """Medium: Detect privilege escalation events (Windows Event ID 4672)."""
    return grade("privilege_escalation_detection", events)


def grade_normal_activity_handling(events=None):
    """Easy: Correctly handle benign normal-activity events (Windows Event ID 4624)."""
    return grade("normal_activity_handling", events)


def grade_credential_access_detection(events=None):
    """Medium: Detect credential access via process injection (Windows Event ID 4688)."""
    return grade("credential_access_detection", events)


def grade_data_exfiltration_detection(events=None):
    """Hard: Detect data exfiltration over network (Windows Event ID 5156)."""
    return grade("data_exfiltration_detection", events)


TASK_GRADERS = {
    "login_attack_detection": grade_login_attack_detection,
    "normal_activity_handling": grade_normal_activity_handling,
    "privilege_escalation_detection": grade_privilege_escalation_detection,
    "credential_access_detection": grade_credential_access_detection,
    "data_exfiltration_detection": grade_data_exfiltration_detection,
}


def confusion_counts(events=None):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for event in _iter_events(events):
        if not isinstance(event, dict):
            continue

        malicious = _to_bool(event.get("malicious", False))
        detected = _to_bool(event.get("detected", False))

        if malicious and detected:
            true_positive += 1
        elif not malicious and detected:
            false_positive += 1
        elif malicious and not detected:
            false_negative += 1
        else:
            true_negative += 1

    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }
