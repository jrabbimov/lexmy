"""LLM client with backend toggle (NIM cloud or LM Studio local)."""

import os
import time
from openai import OpenAI, RateLimitError

from .prompt import SYSTEM_PROMPT, SUMMARY_PROMPT

# ── Backend config ────────────────────────────────────────────────────────────

NIM_BASE = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "deepseek-ai/deepseek-v4-flash"
LMSTUDIO_BASE = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "qwen/qwen3.5-2b"


def make_client(backend: str = "nim", api_key: str = "") -> tuple:
    """
    Returns (OpenAI client, model_name, disable_thinking_flag).
    backend: 'nim' or 'lmstudio'
    """
    if backend == "nim":
        key = api_key or os.environ.get("NIM_API_KEY", "") or "no-key-set"
        return OpenAI(base_url=NIM_BASE, api_key=key), NIM_MODEL, True
    elif backend == "lmstudio":
        return (
            OpenAI(base_url=LMSTUDIO_BASE, api_key="lm-studio"),
            LMSTUDIO_MODEL,
            False,
        )
    raise ValueError(f"Unknown backend: {backend}")


# ── Core call ─────────────────────────────────────────────────────────────────


def llm_call(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str = SYSTEM_PROMPT,
    disable_thinking: bool = False,
    max_tokens: int = 1024,
    temp: float = 0.0,
    top_p: float = 0.95,
) -> str:
    """
    Stream a completion. Retries forever on 429 (rate limit). Never returns None.
    """
    while True:
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
            if disable_thinking:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}

            parts = []
            for chunk in client.chat.completions.create(**kwargs):
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    parts.append(delta.content)
            return "".join(parts).strip()
        except RateLimitError:
            time.sleep(60)


def stream_call(
    client: OpenAI,
    model: str,
    prompt: str,
    system: str = SYSTEM_PROMPT,
    disable_thinking: bool = False,
    max_tokens: int = 1024,
    temp: float = 0.0,
    top_p: float = 0.95,
):
    """
    Yield chunks one by one (generator). For Streamlit `st.write_stream`.
    Retries once on 429 then re-raises.
    """
    for attempt in range(2):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
            if disable_thinking:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": False}}

            for chunk in client.chat.completions.create(**kwargs):
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
            return
        except RateLimitError:
            if attempt == 0:
                time.sleep(60)
            else:
                raise


def summarise(
    client: OpenAI, model: str, history: list, disable_thinking: bool = False
) -> str:
    """Compress full Q&A history into a short rolling summary."""
    if not history:
        return ""
    text = "\n\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
    return llm_call(
        client,
        model,
        SUMMARY_PROMPT.format(history=text),
        system="You summarise legal-advice conversations precisely.",
        disable_thinking=disable_thinking,
        max_tokens=300,
        temp=0.0,
    )
