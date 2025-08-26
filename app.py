"""
FastAPI application providing a simple web GUI and API endpoints for a multi-model
router based on Ollama and local GGUF models. This application exposes a
single-page interface for listing available models, downloading new models,
adjusting configuration parameters, querying the router, and inspecting the
router’s long‑term memory. It leverages a minimal front end built with
Jinja2 templates and vanilla JavaScript to remain compatible with the
restricted execution environment.

Note: The application assumes an Ollama server is reachable on
`http://localhost:11434`. Model downloads and chat operations are forwarded to
Ollama’s REST API. In environments where Ollama is unavailable, the
application will gracefully report errors rather than crash. Memory is
persisted on disk to `memory/knowledge.jsonl`.

This module can be launched with `uvicorn app:app --reload --port 8000`.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Multi‑Model Router GUI", version="0.1.0")

# Directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MEM_DIR = BASE_DIR / "memory"
MEM_FILE = MEM_DIR / "knowledge.jsonl"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
for d in (MODELS_DIR, MEM_DIR, TEMPLATES_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Register templates and static files
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Regex helpers for parsing model metadata
SIZE_RE = re.compile(r"(?i)(\d{1,3})b")
QUANT_RE = re.compile(r"(?i)(q\d(?:_[k]?_?[a-z]?)?)")


def parse_model_meta_from_filename(path: Path) -> Dict[str, Any]:
    """Extract approximate model size (in billions) and quantization scheme from
    a GGUF filename. Returns a dictionary with keys `name`, `size_b` (int or
    None) and `quant` (str or None).
    """
    name = path.stem
    m_size = SIZE_RE.search(name)
    m_quant = QUANT_RE.search(name)
    size_b = int(m_size.group(1)) if m_size else None
    quant = m_quant.group(1).lower() if m_quant else None
    return {"name": name, "size_b": size_b, "quant": quant}


def list_local_models() -> List[Dict[str, Any]]:
    """Enumerate GGUF files in the models directory and return metadata.
    Each item contains the filename, display name, size, quantization and path.
    """
    models: List[Dict[str, Any]] = []
    for gguf in sorted(MODELS_DIR.glob("*.gguf")):
        meta = parse_model_meta_from_filename(gguf)
        models.append({
            "filename": gguf.name,
            "display_name": meta["name"],
            "size_b": meta.get("size_b"),
            "quant": meta.get("quant"),
            "path": gguf.as_posix(),
        })
    return models


def append_memory(record: Dict[str, Any]) -> None:
    """Append a session record to the memory log."""
    with MEM_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_memory(limit: int = 100) -> List[Dict[str, Any]]:
    """Load the last `limit` entries from the memory log. Returns a list of
    dictionaries. If the memory file does not exist, returns an empty list.
    """
    if not MEM_FILE.exists():
        return []
    lines = MEM_FILE.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def memory_digest(max_chars: int = 2000) -> str:
    """Generate a short textual digest of recent memory entries. Each advisor
    response and final answer is truncated to keep the digest within
    `max_chars` characters.
    """
    entries = load_memory(limit=50)
    bullets = []
    for entry in entries:
        final = entry.get("final", "").strip()
        if final:
            bullets.append(f"- FINAL: {final[:256]}")
        advisors = entry.get("advisors", [])
        for adv in advisors:
            ans = adv.get("answer", "").strip()
            model = adv.get("model", "advisor")
            bullets.append(f"- {model}: {ans[:256]}")
    text = "Memory Digest:\n" + "\n".join(bullets)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def ensure_ollama_model_exists(model: str) -> bool:
    """Check if a model tag exists in the local Ollama instance. Returns True
    if the model is present, False otherwise. If the Ollama API is not
    reachable, raises a HTTPException.
    """
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        tags = {m.get("name") for m in data.get("models", []) if m.get("name")}
        return model in tags
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Unable to contact Ollama: {e}")


def download_model_via_ollama(model: str) -> str:
    """Trigger a model download using Ollama's pull API. Returns a status
    message. If the model is already present, a message is returned. Any
    exceptions from Ollama are propagated as HTTP exceptions.
    """
    # First check if present
    present = ensure_ollama_model_exists(model)
    if present:
        return f"Model {model} is already available."
    # Kick off download
    try:
        resp = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model},
            timeout=600,  # allow long download
        )
        resp.raise_for_status()
        # The pull API streams progress lines; we don't parse them here.
        return f"Download of {model} completed."
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download model: {e}")


#############################################
# Multi‑Model Router
#############################################

# We adapt a simplified version of the multi‑model router. In this
# implementation, the router uses a primary model and optional advisor
# models. Each advisor is queried sequentially via Ollama. The final
# synthesis is performed by the primary model. The results are stored in
# memory and returned to the API caller. Parameters for context length,
# batch size, and temperature can be overridden via config.


class RouterConfig:
    """Configuration for router operations. The values are stored in a
    JSON file to persist across restarts. Users can update them via the
    API. Defaults are chosen to suit typical consumer GPUs.
    """

    DEFAULTS = {
        "num_ctx": 8192,
        "num_batch": 320,
        "num_threads": max(2, os.cpu_count() or 4),
        "temperature_primary": 0.6,
        "temperature_advisor": 0.2,
    }
    CONFIG_PATH = BASE_DIR / "router_config.json"

    def __init__(self):
        if self.CONFIG_PATH.exists():
            try:
                with self.CONFIG_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.values = {**self.DEFAULTS, **data}
            except Exception:
                self.values = self.DEFAULTS.copy()
        else:
            self.values = self.DEFAULTS.copy()

    def save(self) -> None:
        with self.CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(self.values, f, ensure_ascii=False, indent=2)

    def get(self) -> Dict[str, Any]:
        return self.values.copy()

    def update(self, updates: Dict[str, Any]) -> None:
        for k, v in updates.items():
            if k in self.DEFAULTS and isinstance(v, (int, float)):
                self.values[k] = v
        self.save()


router_config = RouterConfig()


def ollama_chat(model: str, messages: List[Dict[str, str]], keep_alive: str = "0s",
                options: Optional[Dict[str, Any]] = None) -> str:
    """Send a chat request to the Ollama API and return the assistant's
    response content. Raises HTTPException on failure.
    """
    payload = {
        "model": model,
        "messages": messages,
        "keep_alive": keep_alive,
    }
    if options:
        payload["options"] = options
    try:
        resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama chat failed: {e}")


def run_router_session(question: str, primary_model: str, advisor_models: List[str],
                       force_advisors: bool) -> Dict[str, Any]:
    """Run a single router session. Queries advisors if requested and then
    synthesizes a final answer with the primary model. The session is
    recorded in memory. Returns the session record.
    """
    mem_block = memory_digest()
    use_advisors = force_advisors and bool(advisor_models)
    advisor_results: List[Dict[str, Any]] = []

    # Build and send advisor prompts
    if use_advisors:
        for adv in advisor_models:
            adv_msgs = [
                {"role": "system", "content": "Du bist ein fachlicher Berater. Antworte präzise."},
                {"role": "user", "content": f"Memory Digest:\n{mem_block}\n\nNutzerfrage:\n{question}"},
            ]
            opts = {
                "num_ctx": router_config.values["num_ctx"],
                "num_batch": router_config.values["num_batch"],
                "num_threads": router_config.values["num_threads"],
                "temperature": router_config.values["temperature_advisor"],
            }
            try:
                ans = ollama_chat(model=adv, messages=adv_msgs, keep_alive="0s", options=opts)
            except HTTPException as e:
                ans = f"[Advisor error: {e.detail}]"
            advisor_results.append({"model": adv, "role": "advisor", "answer": ans})

    # Build synthesis prompt for primary model
    advisor_block = "\n".join(
        f"- Advisor [{ar['model']}]:\n{ar['answer']}\n" for ar in advisor_results
    ) if advisor_results else "(keine Beraterantworten)"
    user_content = (
        f"Memory Digest:\n{mem_block}\n\n"
        f"Nutzerfrage:\n{question}\n\n"
        f"Berater-Ergebnisse:\n{advisor_block}\n\n"
        "Bitte liefere jetzt die finale, konsolidierte Antwort."
    )
    primary_msgs = [
        {"role": "system", "content": "Du bist das Leitmodell. Antworte klar, schlüssig und gut strukturiert."},
        {"role": "user", "content": user_content},
    ]
    opts_primary = {
        "num_ctx": router_config.values["num_ctx"],
        "num_batch": router_config.values["num_batch"],
        "num_threads": router_config.values["num_threads"],
        "temperature": router_config.values["temperature_primary"],
    }
    try:
        final_ans = ollama_chat(model=primary_model, messages=primary_msgs, keep_alive="0s", options=opts_primary)
    except HTTPException as e:
        final_ans = f"[Primary model error: {e.detail}]"

    session = {
        "prompt": question,
        "primary": primary_model,
        "advisors": advisor_results,
        "used_advisors": use_advisors,
        "final": final_ans,
        "timestamp": int(time.time()),
    }
    append_memory(session)
    return session


#############################################
# API Endpoints
#############################################


@app.get("/models")
async def models_endpoint():
    """Return the list of local GGUF models with metadata."""
    return list_local_models()


@app.post("/download_model")
async def download_model_endpoint(payload: Dict[str, Any]):
    """Download a model via Ollama if not present. Expects JSON with key
    `model` (the tag to pull). Returns a message upon completion."""
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Field 'model' is required")
    msg = download_model_via_ollama(model)
    return {"message": msg}


@app.post("/query")
async def query_endpoint(payload: Dict[str, Any]):
    """Run a router session. Expects JSON with keys `question`,
    `primary_model`, `advisor_models` (list) and `force_advisors` (bool).
    Returns the session record."""
    question = payload.get("question")
    primary_model = payload.get("primary_model")
    advisor_models = payload.get("advisor_models") or []
    force_advisors = bool(payload.get("force_advisors"))
    if not question:
        raise HTTPException(status_code=400, detail="Field 'question' is required")
    if not primary_model:
        raise HTTPException(status_code=400, detail="Field 'primary_model' is required")
    # Verify primary model exists in Ollama
    if not ensure_ollama_model_exists(primary_model):
        raise HTTPException(status_code=400, detail=f"Primary model '{primary_model}' not found in Ollama")
    # Verify advisor models exist (ignore missing advisors)
    existing_advisors = []
    for adv in advisor_models:
        try:
            if ensure_ollama_model_exists(adv):
                existing_advisors.append(adv)
        except HTTPException:
            continue
    session = run_router_session(question, primary_model, existing_advisors, force_advisors)
    return session


@app.get("/memory")
async def memory_endpoint():
    """Return the full memory log (limited to last 100 entries) and a digest."""
    entries = load_memory(limit=100)
    return {"entries": entries, "digest": memory_digest()}


@app.get("/config")
async def get_config_endpoint():
    """Return current router configuration."""
    return router_config.get()


@app.post("/config")
async def update_config_endpoint(payload: Dict[str, Any]):
    """Update the router configuration. Expects numeric fields matching the
    default config keys. Unsupported keys are ignored."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Config update payload must be a JSON object")
    router_config.update(payload)
    return router_config.get()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})
