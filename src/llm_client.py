"""Centralized LLM client — single point of contact for all AI API calls.

Uses NVIDIA NIM (OpenAI-compatible API) as the AI backbone.
Provides retry logic with exponential backoff, robust JSON parsing,
token usage tracking, and a consistent interface for all modules.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger("interviewer.llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

# Config file for persisting the active model across restarts
_CONFIG_PATH = Path("data") / "model_config.json"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


def _load_active_model() -> str:
    """Load the active model from config file, .env, or default."""
    # 1. Check config file (set by owner dashboard)
    try:
        if _CONFIG_PATH.exists():
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if cfg.get("model"):
                return cfg["model"]
    except Exception:
        pass
    # 2. Fall back to .env
    return os.getenv("NVIDIA_MODEL", DEFAULT_MODEL)


def get_active_model() -> str:
    """Return the currently active model ID."""
    return _load_active_model()


def set_active_model(model_id: str):
    """Persist a new active model to config file."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(
        json.dumps({"model": model_id}, indent=2),
        encoding="utf-8",
    )
    logger.info("Active model changed to: %s", model_id)


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Tracks cumulative token usage across all LLM calls in a session."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    errors: int = 0
    retries: int = 0

    def record(self, prompt_tok: int, completion_tok: int):
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_tokens += prompt_tok + completion_tok
        self.call_count += 1

    def summary(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "errors": self.errors,
            "retries": self.retries,
        }


# Module-level singleton usage tracker
_usage = TokenUsage()


def get_usage() -> TokenUsage:
    """Return the global token usage tracker."""
    return _usage


def reset_usage():
    """Reset the global token usage tracker."""
    global _usage
    _usage = TokenUsage()


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def parse_json_object(text: str) -> dict:
    """Extract and parse a JSON object from potentially noisy LLM output."""
    cleaned = strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find outermost { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON object from LLM response:\n{text[:300]}")


def parse_json_array(text: str) -> list:
    """Extract and parse a JSON array from potentially noisy LLM output."""
    cleaned = strip_fences(text)
    start = cleaned.find("[")
    if start == -1:
        raise ValueError(f"No JSON array found in LLM response:\n{text[:300]}")

    # Walk forward to find matching closing bracket
    depth = 0
    in_string = False
    escape_next = False
    end = -1
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError(f"Unclosed JSON array in LLM response:\n{text[:300]}")

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON array: {exc}") from exc


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified interface for NVIDIA NIM API calls with retry, parsing, and tracking.

    Uses the OpenAI-compatible API provided by NVIDIA NIM.

    Usage:
        client = LLMClient()
        result = client.generate("Your prompt here")           # raw text
        data   = client.generate_json("Return JSON", mode="object")  # parsed dict
        chunks = client.generate_stream("Streaming prompt")    # generator of text chunks
    """

    def __init__(self):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "NVIDIA_API_KEY not found in .env file.\n"
                "Get a free key at https://build.nvidia.com\n"
                "Then add it to your .env file: NVIDIA_API_KEY=nvapi-your-key-here"
            )
        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
        )
        self.model = _load_active_model()

    # -- Core generation methods -------------------------------------------

    def generate(self, prompt: str) -> str:
        """Generate text with automatic retry on transient failures.

        Returns the raw text response.
        """
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                self._track_usage(response)
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                _usage.errors += 1
                if attempt < MAX_RETRIES - 1 and self._is_retryable(exc):
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    _usage.retries += 1
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    break
        raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc

    def generate_json(self, prompt: str, mode: str = "object") -> dict | list:
        """Generate and parse a JSON response.

        Args:
            prompt: The prompt to send.
            mode: "object" to parse a JSON object, "array" to parse a JSON array.

        Returns:
            Parsed dict or list.
        """
        text = self.generate(prompt)
        if mode == "array":
            return parse_json_array(text)
        return parse_json_object(text)

    def generate_stream(self, prompt: str):
        """Stream text chunks from the LLM. Yields text strings.

        Retries on transient failure before first chunk.
        """
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content
                # Track usage after stream completes
                _usage.call_count += 1
                return
            except Exception as exc:
                last_exc = exc
                _usage.errors += 1
                if attempt < MAX_RETRIES - 1 and self._is_retryable(exc):
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    _usage.retries += 1
                    logger.warning(
                        "LLM stream failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, MAX_RETRIES, delay, exc,
                    )
                    time.sleep(delay)
                else:
                    break
        raise RuntimeError(f"LLM stream failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc

    # -- Internals ---------------------------------------------------------

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Determine if an exception is transient and worth retrying."""
        exc_str = str(exc).lower()
        retryable_signals = ["429", "rate limit", "quota", "503", "timeout", "unavailable"]
        return any(signal in exc_str for signal in retryable_signals)

    @staticmethod
    def _track_usage(response):
        """Extract and record token usage from an OpenAI-compatible response."""
        try:
            usage = response.usage
            if usage:
                prompt_tok = usage.prompt_tokens or 0
                completion_tok = usage.completion_tokens or 0
                _usage.record(prompt_tok, completion_tok)
                return
        except (AttributeError, TypeError):
            pass
        # Fallback: count as 1 call with unknown tokens
        _usage.call_count += 1


def verify_nvidia_connection() -> tuple[bool, str]:
    """Test that the NVIDIA API key and model are reachable.

    Returns (success: bool, message: str).
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return False, "NVIDIA_API_KEY not set in .env"

    model = _load_active_model()
    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        if response.choices and response.choices[0].message.content:
            return True, f"Model: {model} — Ready"
        return False, f"Model {model} returned empty response"
    except Exception as exc:
        return False, f"Nvidia API key invalid or model unreachable. Check your .env file. Error: {exc}"
