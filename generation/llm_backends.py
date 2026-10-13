"""llm_backends.py — which LLM writes the single agent's turns: GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Pro.

The single agent (single_agent.py) makes every call through two_agents.call_json(),
i.e. `client.chat.completions.create(model=..., temperature=..., max_tokens=...,
response_format={"type": "json_object"}, messages=[system, user])` and reads
`response.choices[0].message.content`. make_client() returns, per model, a client with
exactly that surface, so the prompt, the payoff calculator, validation and retries are
byte-identical across models -- only the transport differs:

  gpt-5.4 (default)  OpenAI SDK, JSON mode. GPT-5.x rejects `max_tokens`; the request
                     is sent with `max_completion_tokens` instead (as the classifier
                     already does for gpt-5.4), with headroom in case the model spends
                     reasoning tokens, which count against that limit.
  gpt-4.1            OpenAI SDK, unchanged -- kept only so the runs made with it
                     (the files without a model suffix) can be resumed and reported.
  claude-sonnet-4.6  Anthropic SDK, model "claude-sonnet-4-6". No assistant prefill
                     (rejected on 4.6+); the system prompt already demands JSON only,
                     so the reply text is parsed as JSON (a ```json fence is tolerated)
                     and anything unparseable raises ValueError, which single_agent
                     treats as an invalid attempt and retries. Thinking is left off, so
                     Claude answers without a separate reasoning phase.
  gemini-2.5-pro     google-genai SDK, JSON MIME type. 2.5 Pro cannot turn thinking off;
                     the budget is set to its minimum (128) and max_output_tokens is
                     raised by that plus a margin, because thinking tokens count against
                     the output limit and would otherwise truncate the JSON.

Classification is NOT affected: every corpus, whichever model wrote it, is coded by the
same GPT-5.4 classifier (classify_<stem>.py, CLASSIFIER_MODEL in step6_classify.py), so
model differences are never a coder confound.

Credentials (environment):
  OPENAI_API_KEY                  gpt-5.4 / gpt-4.1 generation, and classification for every model
  ANTHROPIC_API_KEY               claude-sonnet-4.6 (or an `ant auth login` profile)
  GEMINI_API_KEY / GOOGLE_API_KEY gemini-2.5-pro
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class Backend:
    provider: str      # openai | anthropic | google
    model_id: str      # exact API model string
    label: str         # for reports and tables


MODELS = {
    "gpt-5.4": Backend("openai", "gpt-5.4", "GPT-5.4"),
    "claude-sonnet-4.6": Backend("anthropic", "claude-sonnet-4-6", "Claude Sonnet 4.6"),
    "gemini-2.5-pro": Backend("google", "gemini-2.5-pro", "Gemini 2.5 Pro"),
    "gpt-4.1": Backend("openai", "gpt-4.1", "GPT-4.1"),      # legacy: earlier runs only
}
DEFAULT_MODEL = "gpt-5.4"

GPT5_COMPLETION_HEADROOM = 2048       # reasoning tokens count against max_completion_tokens

GEMINI_THINKING_BUDGET = 128          # minimum allowed on 2.5 Pro (thinking cannot be disabled)
GEMINI_OUTPUT_MARGIN = 1024
LLM_TIMEOUT_MS = 120_000           # fail fast instead of hanging if the backend never responds


def model_id(model: str) -> str:
    return MODELS[model].model_id


def label(model: str) -> str:
    return MODELS[model].label


# ---------------------------------------------------------------------------
# OpenAI-shaped response objects
# ---------------------------------------------------------------------------

def _response(content: str, prompt_tokens: int | None, completion_tokens: int | None):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=usage)


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def extract_json_text(text: str) -> str:
    """The JSON object in a reply, tolerating a markdown fence or a short preamble.
    Raises ValueError when there is none, so the caller retries the attempt."""
    if text is None:
        raise ValueError("empty reply")
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        json.loads(candidate)          # raises JSONDecodeError (a ValueError) if still invalid
        return candidate
    raise ValueError(f"no JSON object in reply: {text[:120]!r}")


def _split(messages):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    return system, user


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class _ChatShim:
    """Gives an adapter the `client.chat.completions.create(...)` attribute path."""

    def __init__(self, create):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class OpenAIGpt5Chat(_ChatShim):
    """GPT-5.x: same call, but `max_tokens` becomes `max_completion_tokens`."""

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI()
        super().__init__(self._create)

    def _create(self, *, model, messages, max_tokens=300, **kw):
        resp = self._client.chat.completions.create(
            model=model, messages=messages,
            max_completion_tokens=max_tokens + GPT5_COMPLETION_HEADROOM, **kw)
        if not (resp.choices and resp.choices[0].message.content):
            raise ValueError(f"empty completion (finish_reason "
                             f"{resp.choices[0].finish_reason if resp.choices else None})")
        return resp


class AnthropicChat(_ChatShim):
    def __init__(self):
        import anthropic
        # Same reasoning as GeminiChat's timeout: observed a call hang with zero
        # progress rather than erroring out, previously only guarded against on
        # the Gemini side. A client-side timeout turns a silent hang into a
        # normal, retryable failure here too.
        self._client = anthropic.Anthropic(max_retries=4, timeout=LLM_TIMEOUT_MS / 1000)
        super().__init__(self._create)

    def _create(self, *, model, messages, temperature=1.0, max_tokens=300, response_format=None, **_):
        system, user = _split(messages)
        # anthropic SDK 1.x removed `temperature` from messages.create() (passing it is a
        # TypeError), but Sonnet 4.6 still honours it at the API. extra_body merges it into
        # the request JSON as-is, so every model runs at the same temperature.
        resp = self._client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
            extra_body={"temperature": temperature},
        )
        if resp.stop_reason == "refusal":
            raise ValueError("model refused")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if response_format and response_format.get("type") == "json_object":
            text = extract_json_text(text)
        return _response(text, resp.usage.input_tokens, resp.usage.output_tokens)


class GeminiChat(_ChatShim):
    def __init__(self):
        from google import genai
        from google.genai import types
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for gemini-2.5-pro.")
        # No timeout is set by default -- observed hanging indefinitely (13+ minutes,
        # zero progress) when the backend accepts a connection but never responds,
        # as opposed to erroring out quickly (which it also does sometimes, as a
        # RemoteProtocolError the caller's own retry loop already handles). A
        # client-side timeout turns a silent hang into a normal, retryable failure.
        http_options = types.HttpOptions(timeout=LLM_TIMEOUT_MS)
        self._client, self._types = genai.Client(api_key=key, http_options=http_options), types
        super().__init__(self._create)

    def _create(self, *, model, messages, temperature=1.0, max_tokens=300, response_format=None, **_):
        system, user = _split(messages)
        json_mode = bool(response_format and response_format.get("type") == "json_object")
        cfg = self._types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            max_output_tokens=max_tokens + GEMINI_THINKING_BUDGET + GEMINI_OUTPUT_MARGIN,
            response_mime_type="application/json" if json_mode else None,
            thinking_config=self._types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
            # no tools are used; turning AFC off also silences a per-call SDK warning
            automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._client.models.generate_content(model=model, contents=user, config=cfg)
        text = resp.text
        if json_mode:
            text = extract_json_text(text)
        u = resp.usage_metadata
        return _response(text, getattr(u, "prompt_token_count", None),
                         (getattr(u, "candidates_token_count", 0) or 0)
                         + (getattr(u, "thoughts_token_count", 0) or 0))


def make_client(model: str = DEFAULT_MODEL):
    """A client exposing `.chat.completions.create(...)` for the chosen model."""
    if model not in MODELS:
        raise SystemExit(f"unknown model {model!r}; choose from {list(MODELS)}")
    provider = MODELS[model].provider
    if provider == "openai":
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY in your environment.")
        return OpenAIGpt5Chat() if MODELS[model].model_id.startswith("gpt-5") else OpenAI()
    if provider == "anthropic":
        return AnthropicChat()
    return GeminiChat()
