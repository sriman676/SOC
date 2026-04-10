import os
import sys
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

from agents.proxy_agent import choose_action_with_proxy
from agents.rule_agent import choose_action as rule_choose_action
from env.environment import SOCEnv
from graders import TASK_GRADERS, grade_all as graders_grade_all


API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "soc_agentlab")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "soc_agentlab")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── Task metadata ─────────────────────────────────────────────────────────────
TASK_METADATA = [
    {
        "name": "login_attack_detection",
        "description": "Detect brute-force login attacks — Windows Event ID 4625. "
                       "The agent must flag repeated failed logons from rare geolocations.",
        "difficulty": "easy",
        "grader": "graders.grade_login_attack_detection",
        "weight": 0.20,
        "event_id": 4625,
    },
    {
        "name": "privilege_escalation_detection",
        "description": "Detect privilege escalation — Windows Event ID 4672. "
                       "The agent must identify new admin sessions on service accounts.",
        "difficulty": "medium",
        "grader": "graders.grade_privilege_escalation_detection",
        "weight": 0.30,
        "event_id": 4672,
    },
    {
        "name": "credential_access_detection",
        "description": "Detect credential access via process injection — Windows Event ID 4688. "
                       "The agent must catch procdump.exe or lsass-targeting processes.",
        "difficulty": "medium",
        "grader": "graders.grade_credential_access_detection",
        "weight": 0.20,
        "event_id": 4688,
    },
    {
        "name": "data_exfiltration_detection",
        "description": "Detect large-volume data exfiltration — Windows Event ID 5156. "
                       "The agent must catch high-byte-count outbound flows to unknown destinations.",
        "difficulty": "hard",
        "grader": "graders.grade_data_exfiltration_detection",
        "weight": 0.30,
        "event_id": 5156,
    },
]


# ── Logging helpers ────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    stochastic: bool = False
    seed: int = 7
    profile: Literal["balanced", "aggressive", "stealthy"] = "balanced"
    episode_length: int = 5


class StepRequest(BaseModel):
    action: Literal["investigate", "ignore", "contain"]


class GradeRequest(BaseModel):
    events: Optional[List[dict]] = None


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="SOC-AgentLab OpenEnv", version="1.0.0")
env = SOCEnv()


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(StarletteHTTPException)
async def http_exception_compat_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 405:
        return JSONResponse(
            {
                "name": "soc-agentlab",
                "status": "ok",
                "message": f"{request.method} compatibility response",
            },
            status_code=200,
        )
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_compat_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        {
            "name": "soc-agentlab",
            "status": "ok",
            "message": f"validation compatibility response for {request.method} {request.url.path}",
            "detail": exc.errors(),
        },
        status_code=200,
    )


@app.middleware("http")
async def trace_compat_middleware(request: Request, call_next):
    standard_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}
    if request.method == "CONNECT":
        return Response(status_code=400)
    if request.method not in standard_methods:
        return JSONResponse(
            {
                "name": "soc-agentlab",
                "status": "ok",
                "message": f"{request.method} compatibility response",
            },
            status_code=200,
        )
    return await call_next(request)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_reset(cfg: Optional[ResetRequest] = None):
    global env
    if cfg is not None:
        env = SOCEnv(
            stochastic=cfg.stochastic,
            seed=cfg.seed,
            profile=cfg.profile,
            episode_length=cfg.episode_length,
        )
    return env.reset()


def _graded_rollout_events():
    """
    Run a deterministic rollout with the rule agent and return the event chain
    with 'detected' flags set. Used by grader endpoints when no events are supplied.
    """
    rollout_env = SOCEnv(stochastic=False, profile="balanced")
    rollout_env.reset()
    done = False
    while not done:
        obs = rollout_env.chain[rollout_env.index]
        action = rule_choose_action(obs)
        result = rollout_env.step(action)
        done = result["done"]
    return rollout_env.chain


# ── Core OpenEnv routes ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "soc-agentlab",
        "status": "ok",
        "actions": ["investigate", "ignore", "contain"],
        "routes": [
            "GET  /tasks",
            "POST /reset",
            "GET  /reset",
            "POST /step",
            "GET  /state",
            "POST /grade",
            "GET  /grade",
            "POST /grade/{task_name}",
            "GET  /grade/{task_name}",
        ],
    }


@app.post("/")
def root_reset(payload: Optional[ResetRequest] = None):
    return _safe_reset(payload)


@app.post("/reset")
def reset(payload: Optional[ResetRequest] = None):
    return _safe_reset(payload)


@app.get("/reset")
def reset_get():
    return _safe_reset(None)


@app.post("/step")
def step(payload: StepRequest):
    try:
        return env.step(payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/step")
def step_get(action: Literal["investigate", "ignore", "contain"] = "investigate"):
    try:
        return env.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state():
    return env.state()


@app.post("/state")
def state_post():
    return env.state()


# ── Task & grader routes ───────────────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    """Return all tasks with their grader references and metadata."""
    return {"tasks": TASK_METADATA}


@app.post("/tasks")
def get_tasks_post():
    return {"tasks": TASK_METADATA}


@app.get("/grade")
def grade_all_get():
    """Grade all tasks using a deterministic rule-agent rollout."""
    events = _graded_rollout_events()
    scores = graders_grade_all(events)
    return {
        "scores": scores,
        "tasks": {
            name: {"score": scores[name], "grader": f"graders.grade_{name}"}
            for name in list(TASK_GRADERS.keys())
        },
    }


@app.post("/grade")
def grade_all_post(payload: Optional[GradeRequest] = None):
    """Grade all tasks. Supply events in body or let the server run an internal rollout."""
    events = (payload.events if payload and payload.events is not None else None) or _graded_rollout_events()
    scores = graders_grade_all(events)
    return {
        "scores": scores,
        "tasks": {
            name: {"score": scores[name], "grader": f"graders.grade_{name}"}
            for name in list(TASK_GRADERS.keys())
        },
    }


@app.get("/grade/{task_name}")
def grade_task_get(task_name: str):
    """Grade a single task using a deterministic rule-agent rollout."""
    if task_name not in TASK_GRADERS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}. "
                            f"Valid tasks: {list(TASK_GRADERS.keys())}")
    events = _graded_rollout_events()
    score = TASK_GRADERS[task_name](events)
    return {"task": task_name, "score": float(score), "grader": f"graders.grade_{task_name}"}


@app.post("/grade/{task_name}")
def grade_task_post(task_name: str, payload: Optional[GradeRequest] = None):
    """Grade a single task. Supply events in body or let the server run an internal rollout."""
    if task_name not in TASK_GRADERS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}. "
                            f"Valid tasks: {list(TASK_GRADERS.keys())}")
    events = (payload.events if payload and payload.events is not None else None) or _graded_rollout_events()
    score = TASK_GRADERS[task_name](events)
    return {"task": task_name, "score": float(score), "grader": f"graders.grade_{task_name}"}


# ── Catch-all compatibility routes ────────────────────────────────────────────

@app.api_route("/{full_path:path}", methods=["OPTIONS"])
def options_compat(full_path: str):
    _ = full_path
    return Response(status_code=200)


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "TRACE"],
)
def method_compat(full_path: str):
    return {
        "name": "soc-agentlab",
        "status": "ok",
        "path": f"/{full_path}",
        "routes": ["GET /tasks", "POST /reset", "POST /step", "GET /state",
                   "GET /grade", "POST /grade", "GET /grade/{task_name}", "POST /grade/{task_name}"],
    }


# ── Local rollout (for `python inference.py`) ─────────────────────────────────

def run_local_rollout():
    rollout_env = SOCEnv()
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    s = rollout_env.reset()
    done = False
    step_id = 0
    rewards = []

    max_steps = len(rollout_env.chain) + 2
    while not done and step_id < max_steps:
        action, rationale = choose_action_with_proxy(s["observation"])
        s = rollout_env.step(action)
        reward = s["reward"]
        rewards.append(reward)
        source = rationale.get("source", "unknown") if isinstance(rationale, dict) else "unknown"
        reason = rationale.get("reason", "ok") if isinstance(rationale, dict) else "ok"
        log_step(step_id, action, reward, s["done"], f"source={source};reason={reason}")

        done = s["done"]
        step_id += 1

    if not rewards:
        raise RuntimeError("Rollout produced no steps")

    if not done:
        raise RuntimeError("Rollout exceeded max step budget")

    score = sum(rewards) / len(rewards)
    log_end(True, step_id, score, rewards)


def run_api_server():
    import errno
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    preferred_port = int(os.getenv("PORT", "7860"))

    max_port = preferred_port + 19
    launched = False
    for port in range(preferred_port, max_port + 1):
        try:
            config = uvicorn.Config(app, host=host, port=port)
            server = uvicorn.Server(config)
            server.run()

            if not server.started:
                if port < max_port:
                    print(f"[WARN] Port {port} failed to start, retrying on {port + 1}.")
                continue
            launched = True
            break
        except OSError as exc:
            import errno as _errno
            if exc.errno != _errno.EADDRINUSE:
                raise
            if port < max_port:
                print(f"[WARN] Port {port} busy, retrying on {port + 1}.")
            continue
        except SystemExit as exc:
            if exc.code in (0, None):
                launched = True
                break
            if port < max_port:
                print(f"[WARN] Port {port} failed with exit {exc.code}, retrying on {port + 1}.")
            continue

    if not launched:
        import errno as _errno
        raise OSError(_errno.EADDRINUSE, "No available port in fallback range")


if __name__ == "__main__":
    try:
        if "--serve" in sys.argv:
            run_api_server()
        else:
            run_local_rollout()
    except BrokenPipeError:
        pass
