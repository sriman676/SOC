import json
import os
import re

from agents.rule_agent import choose_action as fallback_choose_action


def _extract_action(text):
    normalized = (text or "").strip().lower()
    matches = re.findall(r"\b(contain|investigate|ignore)\b", normalized)
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0], "ok"
    if len(unique) == 0:
        return None, "no_action_found"
    return None, "ambiguous_action"


def choose_action_with_proxy(alert):
    model_name = os.environ.get("MODEL_NAME", "soc-agentlab-rule")
    api_base_url = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    local_action = fallback_choose_action(alert)

    if not api_base_url or not api_key:
        return local_action, {
            "source": "fallback",
            "reason": "missing_api_env",
            "local_action": local_action,
        }

    try:
        # Import lazily so local/fallback mode still works when openai isn't installed.
        from openai import OpenAI
    except Exception:
        return local_action, {
            "source": "fallback",
            "reason": "openai_import_error",
            "local_action": local_action,
        }

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    prompt_payload = {
        "event_id": alert.get("event_id"),
        "severity": alert.get("severity"),
        "stage": alert.get("stage"),
        "failed_logins": alert.get("failed_logins", 0),
        "src_geo": alert.get("src_geo"),
        "process_name": alert.get("process_name"),
        "bytes_out": alert.get("bytes_out", 0),
        "dest_reputation": alert.get("dest_reputation"),
        "account_role": alert.get("account_role"),
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a SOC triage assistant. Return exactly one action from "
                "investigate, ignore, or contain."
            ),
        },
        {
            "role": "user",
            "content": f"Choose the best action for this alert: {json.dumps(prompt_payload, sort_keys=True)}",
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content if response.choices else ""
        api_action, parse_reason = _extract_action(content)

        if api_action is None:
            return local_action, {
                "source": "fallback",
                "reason": f"api_parse_{parse_reason}",
                "model": model_name,
                "response": content,
                "local_action": local_action,
            }

        if api_action != local_action:
            return local_action, {
                "source": "fallback",
                "reason": "api_local_mismatch",
                "model": model_name,
                "response": content,
                "api_action": api_action,
                "local_action": local_action,
            }

        return local_action, {
            "source": "local_policy",
            "reason": "api_call_observed",
            "model": model_name,
            "response": content,
            "api_action": api_action,
            "local_action": local_action,
        }
    except Exception as exc:
        return local_action, {
            "source": "fallback",
            "reason": f"api_error_{type(exc).__name__}",
            "local_action": local_action,
        }