
def estimate_risk(alert):
    score = 0
    severity_weights = {"low": 10, "medium": 30, "high": 55}
    stage_weights = {
        "normal_activity": 0,
        "initial_access": 10,
        "privilege_escalation": 20,
        "credential_access": 25,
        "exfiltration": 30,
    }

    score += severity_weights.get(alert.get("severity"), 0)
    score += stage_weights.get(alert.get("stage"), 0)

    if alert.get("failed_logins", 0) >= 8:
        score += 12
    if alert.get("new_admin_session"):
        score += 15
    if alert.get("process_name") == "procdump.exe":
        score += 20
    if alert.get("bytes_out", 0) > 10000000:
        score += 20
    if alert.get("dest_reputation") == "unknown":
        score += 10
    if alert.get("src_geo") == "rare":
        score += 8

    return min(score, 100)


def explain_action(alert, action, risk):
    return {
        "event_id": alert.get("event_id"),
        "stage": alert.get("stage"),
        "severity": alert.get("severity"),
        "risk": risk,
        "action": action,
    }


def choose_action(alert, return_explanation=False):
    risk = estimate_risk(alert)
    stage = alert.get("stage")

    if risk >= 85 or stage == "exfiltration":
        action = "contain"
    elif risk >= 45:
        action = "investigate"
    else:
        action = "ignore"

    if return_explanation:
        return action, explain_action(alert, action, risk)
    return action
